from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auto_evolve import auto_evolve_if_needed
from app.auto_settings import resolved_auto_settings
from app.database import serialized_write
from app.models import AppSetting, GoldScoreSnapshot, ModelActivationAudit
from app.monitoring.collector_health import record_success
from app.monitoring.system_health import get_system_health
from app.scoring.gold_predictor import (
    evaluate_due_predictions,
    predict_gold_prices,
    prediction_due_status_summary,
    rollback_degraded_prediction_model,
)
from app.scoring.gold_score import compute_and_store_gold_score, compute_and_store_gold_score_with_params
from app.scoring.score_optimizer import get_active_params

logger = logging.getLogger(__name__)


def _json_safe(data: Any) -> str:
    return json.dumps(data or {}, ensure_ascii=False, default=str)


def _record_autopilot_event(
    db: Session,
    *,
    action: str,
    status: str,
    reason: str,
    metrics: dict[str, Any] | None = None,
) -> ModelActivationAudit:
    row = ModelActivationAudit(
        model_type="autopilot",
        action=action,
        from_version=None,
        to_version=status,
        operator="self_healing",
        reason=reason,
        metrics_json=_json_safe(metrics or {}),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def _set_runtime_state(db: Session, key: str, value: dict[str, Any]) -> None:
    row = db.get(AppSetting, key)
    payload = _json_safe(value)
    if row is None:
        row = AppSetting(
            key=key,
            value=payload,
            value_type="json",
            description="自动驾驶闭环运行状态",
            source="self_healing",
        )
        db.add(row)
    else:
        row.value = payload
        row.value_type = "json"
        row.source = "self_healing"
    db.commit()


def _latest_runtime_state(db: Session) -> dict[str, Any] | None:
    row = db.get(AppSetting, "self_healing_last_run")
    if row is None or not row.value:
        return None
    try:
        return json.loads(row.value)
    except json.JSONDecodeError:
        return None


def _score_is_stale(db: Session, max_age_hours: float = 2.0) -> bool:
    row = db.scalar(select(GoldScoreSnapshot).order_by(GoldScoreSnapshot.timestamp.desc()))
    if row is None or row.timestamp is None:
        return True
    ts = row.timestamp if row.timestamp.tzinfo else row.timestamp.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - ts).total_seconds() / 3600 > max_age_hours


def _recompute_score(db: Session) -> dict[str, Any]:
    active_params = get_active_params(db)
    if active_params is not None:
        snapshot = compute_and_store_gold_score_with_params(db, active_params, source="self_healing")
    else:
        snapshot = compute_and_store_gold_score(db)
    return {
        "total_score": snapshot.total_score,
        "direction": snapshot.direction,
        "source": snapshot.source,
        "timestamp": snapshot.timestamp.isoformat() if snapshot.timestamp else None,
    }


def run_self_healing_cycle(
    db: Session,
    *,
    force: bool = False,
    reason: str = "scheduled",
) -> dict[str, Any]:
    """Run the unattended evaluation, correction, evolution, and audit loop."""

    settings = resolved_auto_settings(db)
    enabled = bool(settings.get("AUTO_SELF_HEALING_ENABLED", True))
    autofix = bool(settings.get("AUTO_SELF_HEALING_AUTOFIX", True))
    started_at = datetime.now(timezone.utc)
    actions: list[dict[str, Any]] = []

    if not enabled and not force:
        result = {
            "ok": True,
            "status": "disabled",
            "reason": "AUTO_SELF_HEALING_ENABLED=false",
            "actions": actions,
            "checked_at": started_at.isoformat(),
        }
        _set_runtime_state(db, "self_healing_last_run", result)
        return result

    try:
        due_before = prediction_due_status_summary(db)
        with serialized_write():
            evaluated = evaluate_due_predictions(db)
        actions.append({"action": "evaluate_due_predictions", "result": evaluated})

        rollback = rollback_degraded_prediction_model(db)
        actions.append({"action": "rollback_degraded_prediction_model", "result": rollback})

        if autofix and (_score_is_stale(db) or force):
            with serialized_write():
                score = _recompute_score(db)
            record_success("score_engine", f"self_healing score={score.get('total_score')}")
            actions.append({"action": "recompute_score", "result": score})

        future_pending = int((due_before or {}).get("future_pending_count") or 0)
        if autofix and (force or future_pending == 0):
            with serialized_write():
                prediction = predict_gold_prices(db, persist=True)
            record_success("prediction_engine", f"self_healing run={prediction.get('run_id')}")
            actions.append({
                "action": "persist_prediction_snapshot",
                "result": {
                    "ok": prediction.get("ok"),
                    "run_id": prediction.get("run_id"),
                    "model_version": prediction.get("model_version"),
                },
            })
        else:
            actions.append({
                "action": "persist_prediction_snapshot",
                "skipped": True,
                "reason": f"future pending snapshots exist ({future_pending})",
            })

        evolution = auto_evolve_if_needed(db, force=force, settings=settings)
        actions.append({"action": "auto_evolve_if_needed", "result": evolution})

        health = get_system_health(db)
        status = "ok" if health.get("status") in {"HEALTHY", "DEGRADED", "RISKY"} else "needs_attention"
        result = {
            "ok": True,
            "status": status,
            "autofix": autofix,
            "force": force,
            "reason": reason,
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "due_before": due_before,
            "system_health": {
                "status": health.get("status"),
                "production_grade": health.get("production_grade"),
            },
            "actions": actions,
        }
        _record_autopilot_event(db, action="cycle", status=status, reason=reason, metrics=result)
        _set_runtime_state(db, "self_healing_last_run", result)
        return result
    except Exception as exc:
        logger.exception("self healing cycle failed")
        result = {
            "ok": False,
            "status": "failed",
            "reason": reason,
            "error": str(exc),
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "actions": actions,
        }
        try:
            _record_autopilot_event(db, action="cycle_failed", status="failed", reason=str(exc), metrics=result)
            _set_runtime_state(db, "self_healing_last_run", result)
        except Exception:
            logger.debug("failed to persist self healing failure", exc_info=True)
        return result


def get_self_healing_status(db: Session) -> dict[str, Any]:
    settings = resolved_auto_settings(db)
    latest_audit = db.scalar(
        select(ModelActivationAudit)
        .where(ModelActivationAudit.model_type == "autopilot")
        .order_by(ModelActivationAudit.created_at.desc())
    )
    return {
        "ok": True,
        "enabled": bool(settings.get("AUTO_SELF_HEALING_ENABLED", True)),
        "autofix": bool(settings.get("AUTO_SELF_HEALING_AUTOFIX", True)),
        "settings": settings,
        "last_run": _latest_runtime_state(db),
        "last_audit": {
            "id": latest_audit.id,
            "action": latest_audit.action,
            "status": latest_audit.to_version,
            "reason": latest_audit.reason,
            "created_at": latest_audit.created_at,
        } if latest_audit else None,
    }
