#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="$ROOT_DIR/.venv/bin/python"
UVICORN="$ROOT_DIR/.venv/bin/uvicorn"
STREAMLIT="$ROOT_DIR/.venv/bin/streamlit"

if [[ ! -x "$PYTHON" || ! -x "$UVICORN" || ! -x "$STREAMLIT" ]]; then
  echo "Virtual environment is not ready. Run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

mkdir -p logs .runtime

"$PYTHON" scripts/system_health_check.py >/tmp/gold_fredictor_health.json || true

api_healthy() {
  "$PYTHON" - <<'PY'
import sys
import urllib.request

try:
    with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=3) as resp:
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

if lsof -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1 && ! api_healthy; then
  echo "Port 8000 is occupied by an unhealthy API process; restarting it."
  stop_port 8000
fi

if ! lsof -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
  "$UVICORN" app.main:app --host 127.0.0.1 --port 8000 > logs/local-api.out.log 2> logs/local-api.err.log &
  echo $! > .runtime/api.pid
fi

if ! lsof -iTCP:8501 -sTCP:LISTEN >/dev/null 2>&1; then
  "$STREAMLIT" run dashboard/streamlit_app.py --server.port 8501 --server.address 127.0.0.1 > logs/local-streamlit.out.log 2> logs/local-streamlit.err.log &
  echo $! > .runtime/streamlit.pid
fi

sleep 2
open "http://127.0.0.1:8501"
echo "Gold Fredictor is running:"
echo "  Dashboard: http://127.0.0.1:8501"
echo "  API docs:  http://127.0.0.1:8000/docs"
echo "  Health:    .venv/bin/python scripts/system_health_check.py"
