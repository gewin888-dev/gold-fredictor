#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

APP_DATA_DIR="${GOLD_FREDICTOR_APP_DATA_DIR:-$ROOT_DIR}"
VENV_DIR="${GOLD_FREDICTOR_VENV_DIR:-$ROOT_DIR/.venv}"
ENV_PATH="${GOLD_FREDICTOR_ENV_PATH:-$ROOT_DIR/.env}"
DB_PATH="${GOLD_FREDICTOR_DB_PATH:-$APP_DATA_DIR/gold_monitor.db}"
LOG_DIR="${GOLD_FREDICTOR_LOG_DIR:-$APP_DATA_DIR/logs}"
RUNTIME_DIR="${GOLD_FREDICTOR_RUNTIME_DIR:-$APP_DATA_DIR/.runtime}"
API_PORT="${GOLD_FREDICTOR_API_PORT:-8000}"
DASHBOARD_PORT="${GOLD_FREDICTOR_DASHBOARD_PORT:-8501}"

PYTHON="$VENV_DIR/bin/python"
UVICORN="$VENV_DIR/bin/uvicorn"
STREAMLIT="$VENV_DIR/bin/streamlit"

mkdir -p "$APP_DATA_DIR" "$LOG_DIR" "$RUNTIME_DIR" "$(dirname "$ENV_PATH")"

if [[ ! -x "$PYTHON" && "${GOLD_FREDICTOR_BOOTSTRAP_VENV:-}" == "1" ]]; then
  if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required for first-run setup."
    exit 1
  fi
  python3 -m venv "$VENV_DIR"
  "$PYTHON" -m pip install --upgrade pip
  "$PYTHON" -m pip install -r "$ROOT_DIR/requirements.txt"
fi

if [[ ! -f "$ENV_PATH" && -f "$ROOT_DIR/.env.example" ]]; then
  cp "$ROOT_DIR/.env.example" "$ENV_PATH"
fi
if [[ -f "$ENV_PATH" ]] && ! grep -q '^DATABASE_URL=' "$ENV_PATH"; then
  printf '\nDATABASE_URL=sqlite:///%s\n' "$DB_PATH" >> "$ENV_PATH"
fi
if [[ -f "$ENV_PATH" ]] && ! grep -q '^DASHBOARD_API_BASE_URL=' "$ENV_PATH"; then
  printf 'DASHBOARD_API_BASE_URL=http://127.0.0.1:%s\n' "$API_PORT" >> "$ENV_PATH"
fi

export GOLD_FREDICTOR_ENV_PATH="$ENV_PATH"
export DATABASE_URL="sqlite:///$DB_PATH"
export DASHBOARD_API_BASE_URL="${DASHBOARD_API_BASE_URL:-http://127.0.0.1:$API_PORT}"

if [[ ! -x "$PYTHON" || ! -x "$UVICORN" || ! -x "$STREAMLIT" ]]; then
  echo "Virtual environment is not ready. Run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

OPEN_BROWSER=true
if [[ "${1:-}" == "--no-open" || "${GOLD_FREDICTOR_NO_OPEN:-}" == "1" ]]; then
  OPEN_BROWSER=false
fi

"$PYTHON" scripts/system_health_check.py >/tmp/gold_fredictor_health.json || true

api_healthy() {
  "$PYTHON" - <<'PY'
import sys
import os
import urllib.request

try:
    port = os.environ.get("GOLD_FREDICTOR_API_PORT", "8000")
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    raise SystemExit(0 if resp.status == 200 and '"status":"ok"' in body.replace(" ", "") else 1)
except Exception:
    raise SystemExit(1)
PY
}

stop_port() {
  local port="$1"
  local pids
  pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    kill $pids 2>/dev/null || true
    sleep 1
  fi
}

if lsof -iTCP:"$API_PORT" -sTCP:LISTEN >/dev/null 2>&1 && ! api_healthy; then
  echo "Port $API_PORT is occupied by an unhealthy API process; restarting it."
  stop_port "$API_PORT"
fi

if ! lsof -iTCP:"$API_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  "$UVICORN" app.main:app --host 127.0.0.1 --port "$API_PORT" > "$LOG_DIR/local-api.out.log" 2> "$LOG_DIR/local-api.err.log" &
  echo $! > "$RUNTIME_DIR/api.pid"
fi

if ! lsof -iTCP:"$DASHBOARD_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  "$STREAMLIT" run dashboard/streamlit_app.py \
    --server.port "$DASHBOARD_PORT" \
    --server.address 127.0.0.1 \
    --server.headless true \
    --browser.gatherUsageStats false \
    > "$LOG_DIR/local-streamlit.out.log" 2> "$LOG_DIR/local-streamlit.err.log" &
  echo $! > "$RUNTIME_DIR/streamlit.pid"
fi

sleep 2
if [[ "$OPEN_BROWSER" == "true" ]]; then
  open "http://127.0.0.1:$DASHBOARD_PORT"
fi
echo "Gold Fredictor is running:"
echo "  Dashboard: http://127.0.0.1:$DASHBOARD_PORT"
echo "  API docs:  http://127.0.0.1:$API_PORT/docs"
echo "  Health:    .venv/bin/python scripts/system_health_check.py"
