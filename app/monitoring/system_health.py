from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.events.calendar import list_macro_events
from app.models import (
    ExternalMarketIndicator,
    GoldPredictionEvaluation,
    GoldPredictionSnapshot,
    IntradaySnapshot,
)
from app.monitoring.collector_health import get_health_summary
from app.monitoring.health import get_data_health
from app.scoring.gold_predictor import prediction_due_status_summary


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _age_seconds(timestamp: datetime | None) -> float | None:
    if timestamp is None:
        return None
    ts = timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
    return round((_utc_now() - ts).total_seconds(), 1)


def _database_health(db: Session) -> dict[str, Any]:
    try:
        db.execute(text("SELECT 1")).scalar()
        return {"status": "healthy", "message": "数据库可读。"}
    except Exception as exc:
        return {"status": "broken", "message": f"数据库读检查失败: {exc}"}


def _recorder_health(db: Session) -> dict[str, Any]:
    row = db.scalar(select(IntradaySnapshot).order_by(IntradaySnapshot.timestamp.desc()))
    age = _age_seconds(row.timestamp if row else None)
    if row is None:
        status = "no_data"
    elif age is not None and age < 120:
        status = "healthy"
    elif age is not None and age < 600:
        status = "degraded"
    else:
        status = "stale"
    return {
        "status": status,
        "last_record": row.timestamp.isoformat() if row and row.timestamp else None,
        "age_seconds": age,
        "last_price": row.price if row else None,
    }


def _prediction_health(db: Session) -> dict[str, Any]:
    due = prediction_due_status_summary(db)
    snapshot_count = db.scalar(select(func.count()).select_from(GoldPredictionSnapshot)) or 0
    evaluation_count = db.scalar(select(func.count()).select_from(GoldPredictionEvaluation)) or 0
    due_pending = int(due.get("due_pending_count") or 0)
    if snapshot_count == 0:
        status = "no_data"
    elif due_pending:
        status = "degraded"
    else:
        status = "healthy"
    return {
        "status": status,
        "snapshot_count": int(snapshot_count),
        "evaluation_count": int(evaluation_count),
        "due_pending_count": due_pending,
        "future_pending_count": int(due.get("future_pending_count") or 0),
        "can_evolve": bool(due.get("can_evolve")),
        "message": due.get("message"),
    }


def _external_indicator_health(db: Session) -> dict[str, Any]:
    latest = db.scalar(select(ExternalMarketIndicator).order_by(ExternalMarketIndicator.timestamp.desc()))
    age = _age_seconds(latest.updated_at if latest else None)
    status = "healthy" if latest and age is not None and age < 3 * 86400 else ("stale" if latest else "no_data")
    return {
        "status": status,
        "latest_indicator": latest.indicator_id if latest else None,
        "latest_updated_at": latest.updated_at.isoformat() if latest and latest.updated_at else None,
        "age_seconds": age,
    }


def _event_health(db: Session) -> dict[str, Any]:
    events = list_macro_events(db, days_ahead=30)
    high_count = sum(1 for event in events if event.importance == "high")
    return {
        "status": "healthy" if events else "no_data",
        "upcoming_count": len(events),
        "high_importance_count": high_count,
    }


def _self_healing_health(db: Session) -> dict[str, Any]:
    try:
        from app.self_healing import get_self_healing_status

        status = get_self_healing_status(db)
    except Exception as exc:
        return {"status": "degraded", "message": f"自动驾驶状态读取失败: {exc}"}
    if not status.get("enabled"):
        return {"status": "degraded", "enabled": False, "message": "自动驾驶闭环未开启。"}
    last_run = status.get("last_run") or {}
    if not last_run:
        return {"status": "no_data", "enabled": True, "message": "自动驾驶闭环尚未运行。"}
    run_status = last_run.get("status")
    if run_status == "failed":
        health = "degraded"
    elif run_status in {"ok", "needs_attention"}:
        health = "healthy"
    else:
        health = "degraded"
    return {
        "status": health,
        "enabled": status.get("enabled"),
        "autofix": status.get("autofix"),
        "last_run_status": run_status,
        "last_finished_at": last_run.get("finished_at"),
    }


def _status_rank(status: str) -> int:
    return {
        "healthy": 0,
        "ok": 0,
        "market_closed": 0,
        "warn": 1,
        "degraded": 1,
        "stale": 2,
        "risky": 2,
        "no_data": 2,
        "error": 3,
        "critical": 3,
        "broken": 3,
    }.get(status, 1)


def _overall_status(components: dict[str, Any], data_health: dict[str, Any]) -> str:
    if _status_rank(components["database"]["status"]) >= 3:
        return "BROKEN"
    collector_overall = components["collectors"].get("overall", "healthy")
    critical_collectors = components["collectors"].get("summary", {}).get("critical_issues", [])
    if data_health.get("status") == "error" or collector_overall == "critical" or critical_collectors:
        return "BROKEN"
    if data_health.get("status") == "warn":
        return "RISKY"
    if any(_status_rank(component.get("status", "healthy")) >= 2 for component in components.values() if isinstance(component, dict)):
        return "RISKY"
    if collector_overall == "degraded" or any(_status_rank(component.get("status", "healthy")) == 1 for component in components.values() if isinstance(component, dict)):
        return "DEGRADED"
    return "HEALTHY"


def get_system_health(db: Session) -> dict[str, Any]:
    data = get_data_health(db)
    components: dict[str, Any] = {
        "database": _database_health(db),
        "data": {"status": data["status"], "ok": data["ok"]},
        "collectors": get_health_summary(),
        "recorder": _recorder_health(db),
        "prediction": _prediction_health(db),
        "external_indicators": _external_indicator_health(db),
        "events": _event_health(db),
        "self_healing": _self_healing_health(db),
    }
    overall = _overall_status(components, data)
    production_grade = overall in {"HEALTHY", "DEGRADED"} and data.get("status") == "ok"
    return {
        "ok": overall not in {"BROKEN"},
        "status": overall,
        "production_grade": production_grade,
        "checked_at": _utc_now().isoformat(),
        "components": components,
        "data_health": data,
        "message": "生产级可用。" if production_grade else "系统可用性或数据可信度需要关注，详见 components。",
    }
