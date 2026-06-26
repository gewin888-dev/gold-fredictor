"""配置目录与运行时审计。

集中描述系统配置的来源、敏感性、热更新能力和维护说明，避免后续把
`.env`、数据库开关、页面配置和文档各维护一份。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auto_settings import AUTO_SETTING_FIELDS, get_auto_settings
from app.config import get_settings
from app.models import AppSetting


KNOWN_STATE_SETTING_KEYS = {
    "collector_health_state",
    "self_healing_last_run",
}


@dataclass(frozen=True)
class ConfigItem:
    key: str
    attr: str | None
    group: str
    value_type: str
    storage: str
    description: str
    secret: bool = False
    hot_reload: bool = False
    recommended: bool = False
    requires_restart: bool = True


CONFIG_CATALOG: dict[str, ConfigItem] = {
    "DATABASE_URL": ConfigItem("DATABASE_URL", "database_url", "core", "str", "env", "数据库连接地址。", recommended=True),
    "DASHBOARD_API_BASE_URL": ConfigItem("DASHBOARD_API_BASE_URL", "dashboard_api_base_url", "ui", "str", "env", "Streamlit 调用的 FastAPI 地址。"),
    "API_KEY": ConfigItem("API_KEY", "api_key", "security", "str", "env", "写操作 API 鉴权密钥；生产环境建议配置。", secret=True, recommended=True),
    "PRODUCTION_MODE": ConfigItem("PRODUCTION_MODE", "production_mode", "core", "bool", "env", "生产模式开关，影响低可信数据展示和生产级判断。"),
    "SHOW_LOW_CONFIDENCE_DATA": ConfigItem("SHOW_LOW_CONFIDENCE_DATA", "show_low_confidence_data", "ui", "bool", "env", "是否展示样本/占位/低可信来源数据。"),
    "PREDICTION_SCORE_SOURCES": ConfigItem("PREDICTION_SCORE_SOURCES", "prediction_score_sources", "prediction", "str", "env", "预测允许使用的评分来源白名单。"),
    "FRED_API_KEY": ConfigItem("FRED_API_KEY", "fred_api_key", "data_source", "str", "env", "FRED 官方 API Key。", secret=True, recommended=True),
    "FRED_OBSERVATION_START": ConfigItem("FRED_OBSERVATION_START", "fred_observation_start", "data_source", "str", "env", "FRED 首次拉取起始日期。"),
    "FRED_VERIFY_SSL": ConfigItem("FRED_VERIFY_SSL", "fred_verify_ssl", "data_source", "bool", "env", "FRED 请求是否校验证书。"),
    "NEWSAPI_KEY": ConfigItem("NEWSAPI_KEY", "newsapi_key", "data_source", "str", "env", "NewsAPI Key；未配置时可降级到 GDELT/跳过。", secret=True),
    "NEWSAPI_DAILY_LIMIT": ConfigItem("NEWSAPI_DAILY_LIMIT", "newsapi_daily_limit", "data_source", "int", "env", "NewsAPI 单次采集上限。"),
    "NEWSAPI_TIMEOUT_SECONDS": ConfigItem("NEWSAPI_TIMEOUT_SECONDS", "newsapi_timeout_seconds", "data_source", "int", "env", "NewsAPI 请求超时秒数。"),
    "NEWSAPI_VERIFY_SSL": ConfigItem("NEWSAPI_VERIFY_SSL", "newsapi_verify_ssl", "data_source", "bool", "env", "NewsAPI 请求是否校验证书。"),
    "FEISHU_WEBHOOK_URL": ConfigItem("FEISHU_WEBHOOK_URL", "feishu_webhook_url", "notification", "str", "env", "飞书机器人 Webhook；为空时跳过推送。", secret=True),
    "FEISHU_SECRET": ConfigItem("FEISHU_SECRET", "feishu_secret", "notification", "str", "env", "飞书机器人签名密钥。", secret=True),
    "DEEPSEEK_API_KEY": ConfigItem("DEEPSEEK_API_KEY", "deepseek_api_key", "ai", "str", "env", "AI 助理模型 API Key。", secret=True),
    "DEEPSEEK_BASE_URL": ConfigItem("DEEPSEEK_BASE_URL", "deepseek_base_url", "ai", "str", "env", "AI 助理模型 API 基础地址。"),
    "DEEPSEEK_MODEL": ConfigItem("DEEPSEEK_MODEL", "deepseek_model", "ai", "str", "env", "AI 助理模型名称。"),
    "AUTO_START_SCHEDULER": ConfigItem("AUTO_START_SCHEDULER", "auto_start_scheduler", "operation", "bool", "env", "API 启动时是否自动启动调度器。"),
    "AUTO_BOOTSTRAP_DATA": ConfigItem("AUTO_BOOTSTRAP_DATA", "auto_bootstrap_data", "operation", "bool", "env", "API 启动时是否后台补充基础数据。"),
}

for _key, (_attr, _value_type) in AUTO_SETTING_FIELDS.items():
    CONFIG_CATALOG[_key] = ConfigItem(
        key=_key,
        attr=_attr,
        group="automation",
        value_type=_value_type,
        storage="db_override",
        description="自动驾驶/自我进化运行参数，可由页面或 API 管理。",
        hot_reload=True,
        requires_restart=False,
    )


def _mask(value: Any, *, secret: bool) -> Any:
    if not secret:
        return value
    text = "" if value is None else str(value)
    if not text:
        return ""
    if len(text) <= 8:
        return "***"
    return f"{text[:4]}***{text[-4:]}"


def _env_source(key: str, value: Any) -> str:
    if key in os.environ:
        return "env"
    if value not in ("", None):
        return "default"
    return "unset"


def _status(item: ConfigItem, value: Any) -> tuple[str, str]:
    missing = value in ("", None)
    if item.recommended and missing:
        return "warn", "建议配置，缺失时系统可运行但能力受限。"
    if item.secret and missing:
        return "empty", "未配置；相关外部能力会跳过或降级。"
    return "ok", "配置正常。"


def get_config_audit(db: Session | None = None) -> dict[str, Any]:
    """返回面向维护的配置快照；敏感值默认脱敏。"""
    settings = get_settings()
    auto_settings = get_auto_settings(db)
    db_rows: dict[str, AppSetting] = {}
    if db is not None:
        db_rows = {row.key: row for row in db.scalars(select(AppSetting)).all()}

    items: list[dict[str, Any]] = []
    for key in sorted(CONFIG_CATALOG):
        item = CONFIG_CATALOG[key]
        if item.storage == "db_override":
            value = auto_settings.get(key)
            source = "database" if key in db_rows else _env_source(key, value)
            updated_at = db_rows[key].updated_at if key in db_rows else None
        else:
            value = getattr(settings, item.attr or "", None)
            source = _env_source(key, value)
            updated_at = None
        status, message = _status(item, value)
        items.append({
            "key": key,
            "group": item.group,
            "storage": item.storage,
            "source": source,
            "value_type": item.value_type,
            "value": _mask(value, secret=item.secret),
            "is_secret": item.secret,
            "hot_reload": item.hot_reload,
            "requires_restart": item.requires_restart,
            "status": status,
            "message": message,
            "description": item.description,
            "updated_at": updated_at,
        })

    known = set(CONFIG_CATALOG)
    unknown_db_settings = sorted(key for key in db_rows if key not in known and key not in KNOWN_STATE_SETTING_KEYS)
    summary = {
        "total": len(items),
        "ok": sum(1 for row in items if row["status"] == "ok"),
        "warn": sum(1 for row in items if row["status"] == "warn"),
        "empty": sum(1 for row in items if row["status"] == "empty"),
        "database_overrides": sum(1 for row in items if row["source"] == "database"),
        "unknown_db_settings": len(unknown_db_settings),
    }
    return {
        "ok": True,
        "summary": summary,
        "items": items,
        "unknown_db_settings": unknown_db_settings,
        "known_state_settings": sorted(key for key in db_rows if key in KNOWN_STATE_SETTING_KEYS),
        "management": {
            "env_file": ".env",
            "runtime_table": "app_settings",
            "safe_hot_reload": [row["key"] for row in items if row["hot_reload"]],
            "restart_required": [row["key"] for row in items if row["requires_restart"]],
        },
    }
