from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.database import SessionLocal, init_db
from app.monitoring.system_health import get_system_health


def main() -> int:
    init_db()
    db = SessionLocal()
    try:
        payload = get_system_health(db)
    finally:
        db.close()

    print(json.dumps(payload, ensure_ascii=False, default=str, indent=2))
    return 0 if payload.get("status") in {"HEALTHY", "DEGRADED", "RISKY"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
