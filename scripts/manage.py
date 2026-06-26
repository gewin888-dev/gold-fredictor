from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config_registry import get_config_audit
from app.database import SessionLocal, init_db
from app.monitoring.system_health import get_system_health
from app.self_healing import run_self_healing_cycle


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, default=str, indent=2))


def cmd_health(_: argparse.Namespace) -> int:
    init_db()
    db = SessionLocal()
    try:
        payload = get_system_health(db)
    finally:
        db.close()
    _print_json(payload)
    return 0 if payload.get("status") in {"HEALTHY", "DEGRADED", "RISKY"} else 2


def cmd_config(args: argparse.Namespace) -> int:
    init_db()
    db = SessionLocal()
    try:
        payload = get_config_audit(db)
    finally:
        db.close()
    if args.format == "table":
        print("KEY\tGROUP\tSOURCE\tSTATUS\tHOT_RELOAD\tDESCRIPTION")
        for row in payload["items"]:
            print(
                f"{row['key']}\t{row['group']}\t{row['source']}\t"
                f"{row['status']}\t{row['hot_reload']}\t{row['description']}"
            )
    else:
        _print_json(payload)
    return 0 if payload["summary"]["warn"] == 0 else 1


def cmd_self_heal(args: argparse.Namespace) -> int:
    init_db()
    db = SessionLocal()
    try:
        payload = run_self_healing_cycle(db, force=args.force, reason="manage_cli")
    finally:
        db.close()
    _print_json(payload)
    return 0 if payload.get("ok") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gold Fredictor maintenance console")
    subparsers = parser.add_subparsers(dest="command", required=True)

    health = subparsers.add_parser("health", help="run system health check")
    health.set_defaults(func=cmd_health)

    config = subparsers.add_parser("config", help="audit runtime configuration")
    config.add_argument("--format", choices=["json", "table"], default="json")
    config.set_defaults(func=cmd_config)

    self_heal = subparsers.add_parser("self-heal", help="run self-healing cycle")
    self_heal.add_argument("--force", action="store_true", help="run even if disabled by switches")
    self_heal.set_defaults(func=cmd_self_heal)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
