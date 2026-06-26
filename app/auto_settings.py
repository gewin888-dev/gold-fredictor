"""自动优化运行开关。

密钥和部署默认值仍来自 .env；页面修改只落库到 AppSetting，避免改写配置文件。
"""
from __future__ import annotations

import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import AppSetting


AUTO_SWITCH_KEYS = (
    "AUTO_EVOLUTION_FULL_AUTO",
    "AUTO_SELF_HEALING_ENABLED",
    "AUTO_SELF_HEALING_AUTOFIX",
    "AUTO_OPTIMIZE_SCORE_PARAMS",
    "AUTO_ACTIVATE_OPTIMIZED_PARAMS",
    "AUTO_OPTIMIZE_PREDICTION_MODEL",
    "AUTO_ACTIVATE_PREDICTION_MODEL",
)

AUTO_SETTING_FIELDS: dict[str, tuple[str, str]] = {
    "AUTO_EVOLUTION_FULL_AUTO": ("auto_evolution_full_auto", "bool"),
    "AUTO_SELF_HEALING_ENABLED": ("auto_self_healing_enabled", "bool"),
    "AUTO_SELF_HEALING_AUTOFIX": ("auto_self_healing_autofix", "bool"),
    "AUTO_OPTIMIZE_SCORE_PARAMS": ("auto_optimize_score_params", "bool"),
    "AUTO_ACTIVATE_OPTIMIZED_PARAMS": ("auto_activate_optimized_params", "bool"),
    "AUTO_OPTIMIZE_PREDICTION_MODEL": ("auto_optimize_prediction_model", "bool"),
    "AUTO_ACTIVATE_PREDICTION_MODEL": ("auto_activate_prediction_model", "bool"),
    "AUTO_OPTIMIZE_MIN_HIT_RATE": ("auto_optimize_min_hit_rate", "float"),
    "AUTO_OPTIMIZE_N_ITER": ("auto_optimize_n_iter", "int"),
    "AUTO_OPTIMIZE_HORIZON_DAYS": ("auto_optimize_horizon_days", "int"),
    "AUTO_PREDICTION_N_ITER": ("auto_prediction_n_iter", "int"),
    "AUTO_PREDICTION_MIN_SCORE": ("auto_prediction_min_score", "float"),
    "AUTO_PREDICTION_MAX_MAPE_PCT": ("auto_prediction_max_mape_pct", "float"),
    "AUTO_PREDICTION_MIN_DIRECTION_ACCURACY": ("auto_prediction_min_direction_accuracy", "float"),
    "AUTO_PREDICTION_MIN_SAMPLES": ("auto_prediction_min_samples", "int"),
    "AUTO_PREDICTION_MIN_VALID_HORIZONS": ("auto_prediction_min_valid_horizons", "int"),
}


def _coerce(value: Any, value_type: str) -> Any:
    if value_type == "bool":
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if value_type == "int":
        return int(value)
    if value_type == "float":
        return float(value)
    return str(value)


def _serialize(value: Any, value_type: str) -> str:
    return json.dumps(_coerce(value, value_type), ensure_ascii=False)


def _deserialize(value: str, value_type: str) -> Any:
    try:
        raw = json.loads(value)
    except json.JSONDecodeError:
        raw = value
    return _coerce(raw, value_type)


def default_auto_settings() -> dict[str, Any]:
    settings = get_settings()
    return {
        key: _coerce(getattr(settings, attr), value_type)
        for key, (attr, value_type) in AUTO_SETTING_FIELDS.items()
    }


def get_auto_settings(db: Session | None = None) -> dict[str, Any]:
    """读取当前自动优化设置；数据库值覆盖 .env 默认值。"""
    result = default_auto_settings()
    if db is None:
        return result
    rows = db.scalars(select(AppSetting).where(AppSetting.key.in_(AUTO_SETTING_FIELDS))).all()
    for row in rows:
        field = AUTO_SETTING_FIELDS.get(row.key)
        if field is None:
            continue
        result[row.key] = _deserialize(row.value, field[1])
    return result


def set_auto_settings(db: Session, payload: dict[str, Any]) -> dict[str, Any]:
    """只允许更新白名单自动优化开关，返回完整设置。"""
    for key, value in payload.items():
        if key not in AUTO_SWITCH_KEYS:
            continue
        _, value_type = AUTO_SETTING_FIELDS[key]
        row = db.get(AppSetting, key)
        if row is None:
            row = AppSetting(
                key=key,
                value=_serialize(value, value_type),
                value_type=value_type,
                description="自动优化页面开关",
                source="API",
            )
            db.add(row)
        else:
            row.value = _serialize(value, value_type)
            row.value_type = value_type
            row.source = "API"
    db.commit()
    return get_auto_settings(db)


def resolved_auto_settings(db: Session | None = None) -> dict[str, Any]:
    """调度器使用的运行配置快照。"""
    return get_auto_settings(db)
