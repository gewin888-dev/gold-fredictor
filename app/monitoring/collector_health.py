"""采集器健康监控与自动告警。

每个采集器独立跟踪：最后成功时间、连续失败次数、最后错误。
内存中维护状态，通过 /health/collectors 端点暴露。

设计原则：
- 采集器调用处主动调用 record_success/record_failure
- 连续失败 ≥3 次自动飞书告警
- /health/collectors 返回所有采集器状态和整体健康评分
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ── 采集器定义 ──────────────────────────────────────────────────
# (name, label, max_age_hours, critical)
COLLECTOR_DEFS: list[dict[str, Any]] = [
    {"name": "gold_price",           "label": "金价日线",     "max_age_hours": 1.0,    "critical": True},
    {"name": "intraday_snapshot",    "label": "日内快照",     "max_age_hours": 2.0,    "critical": True},
    {"name": "fred_data",            "label": "FRED宏观",     "max_age_hours": 25.0,   "critical": True},
    {"name": "cftc_position",        "label": "CFTC仓位",     "max_age_hours": 170.0,  "critical": True},
    {"name": "china_premium",        "label": "中国溢价",     "max_age_hours": 25.0,   "critical": False},
    {"name": "central_bank_gold",    "label": "央行购金",     "max_age_hours": 2160.0, "critical": False},
    {"name": "news_sentiment",       "label": "新闻情绪",     "max_age_hours": 25.0,   "critical": False},
    {"name": "market_structure",     "label": "市场结构",     "max_age_hours": 25.0,   "critical": False},
    {"name": "sp500_snapshot",       "label": "标普500",      "max_age_hours": 25.0,   "critical": False},
    {"name": "silver_snapshot",      "label": "白银价格",     "max_age_hours": 25.0,   "critical": False},
    {"name": "gld_etf",              "label": "GLD ETF",     "max_age_hours": 25.0,   "critical": False},
    {"name": "google_trend",         "label": "搜索热度",     "max_age_hours": 49.0,   "critical": False},
    {"name": "gdx_snapshot",         "label": "GDX矿业",      "max_age_hours": 25.0,   "critical": False},
    {"name": "wti_snapshot",         "label": "WTI原油",      "max_age_hours": 25.0,   "critical": False},
    {"name": "copper_snapshot",      "label": "铜价",         "max_age_hours": 25.0,   "critical": False},
    {"name": "score_engine",         "label": "评分引擎",     "max_age_hours": 2.0,    "critical": True},
    {"name": "prediction_engine",    "label": "预测引擎",     "max_age_hours": 2.0,    "critical": True},
]

MAX_CONSECUTIVE_FAILURES = 3

# ── 内存状态 ────────────────────────────────────────────────────

_collector_state: dict[str, dict[str, Any]] = {}
_alerted: set[str] = set()


def _load_state() -> None:
    """从数据库恢复上次的健康状态，合并到内存中。"""
    try:
        from app.database import SessionLocal
        from app.models import AppSetting
        from sqlalchemy import select as _select
        import json as _json
        db = SessionLocal()
        try:
            row = db.scalar(_select(AppSetting).where(AppSetting.key == "collector_health_state"))
            if row and row.value:
                data = _json.loads(row.value)
                for k, v in data.items():
                    if k not in _collector_state:  # 不覆盖内存中已有的更新数据
                        last_success = v.get("last_success")
                        _collector_state[k] = {
                            "last_success": last_success,
                            "last_success_dt": datetime.fromisoformat(last_success) if last_success else None,
                            "consecutive_failures": v.get("consecutive_failures", 0),
                            "last_error": v.get("last_error"),
                            "last_detail": v.get("last_detail"),
                        }
        finally:
            db.close()
    except Exception:
        pass


# 模块加载时自动恢复状态
_load_state()


def record_success(name: str, detail: str = "") -> None:
    """记录采集器成功。"""
    now = datetime.now(timezone.utc)
    prev = _collector_state.get(name, {})
    _collector_state[name] = {
        "last_success": now.isoformat(),
        "last_success_dt": now,
        "consecutive_failures": 0,
        "last_error": None,
        "last_detail": detail,
    }
    if name in _alerted:
        _alerted.discard(name)
        logger.info("采集器 %s 已恢复", name)
    _persist_state()

def _persist_state() -> None:
    """将当前状态持久化到数据库，防止重启丢失。
    
    使用 1 秒超时快速失败，避免阻塞采集循环。
    """
    try:
        from app.database import SessionLocal
        from app.models import AppSetting
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert
        import json as _json
        db = SessionLocal()
        try:
            # 设置 1 秒超时，避免被主会话锁阻塞
            db.connection().connection.execute("PRAGMA busy_timeout=1000")
            data = {}
            for k, v in _collector_state.items():
                data[k] = {
                    "last_success": v.get("last_success"),
                    "consecutive_failures": v.get("consecutive_failures", 0),
                    "last_error": v.get("last_error"),
                    "last_detail": v.get("last_detail"),
                }
            stmt = sqlite_insert(AppSetting).values(
                key="collector_health_state",
                value=_json.dumps(data, ensure_ascii=False),
                value_type="json",
                description="Collector health tracking state",
                source="system",
                updated_at=datetime.now(timezone.utc),
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=["key"],
                set_={"value": _json.dumps(data, ensure_ascii=False), "updated_at": datetime.now(timezone.utc)},
            )
            db.execute(stmt)
            db.commit()
        finally:
            db.close()
    except Exception:
        pass


def record_failure(name: str, error: str) -> None:
    """记录采集器失败。连续失败达到阈值时触发告警。"""
    now = datetime.now(timezone.utc)
    prev = _collector_state.get(name, {})
    consecutive = prev.get("consecutive_failures", 0) + 1
    _collector_state[name] = {
        "last_success": prev.get("last_success"),
        "last_success_dt": prev.get("last_success_dt"),
        "consecutive_failures": consecutive,
        "last_error": error,
        "last_error_time": now.isoformat(),
    }

    if consecutive >= MAX_CONSECUTIVE_FAILURES and name not in _alerted:
        _alerted.add(name)
        logger.error(
            "采集器 %s 连续失败 %d 次，触发告警。最近错误: %s",
            name, consecutive, error[:200],
        )
        _try_alert(name, consecutive, error)
    _persist_state()


def _try_alert(name: str, consecutive: int, error: str) -> None:
    """尝试发送飞书告警（失败不阻塞）。"""
    try:
        from app.notifications.feishu import send_text_message
        msg = (
            f"⚠️ 采集器告警: {name}\n"
            f"连续失败 {consecutive} 次\n"
            f"错误: {error[:200]}\n"
            f"时间: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )
        send_text_message(msg)
    except Exception:
        pass


def get_health_summary() -> dict[str, Any]:
    """返回所有采集器的健康状态摘要。"""
    now = datetime.now(timezone.utc)
    collectors_status = []

    for cdef in COLLECTOR_DEFS:
        name = cdef["name"]
        state = _collector_state.get(name, {})
        last_success_dt = state.get("last_success_dt")
        consecutive = state.get("consecutive_failures", 0)

        if last_success_dt is not None:
            age_hours = (now - last_success_dt).total_seconds() / 3600
            if age_hours > cdef["max_age_hours"]:
                status = "stale"
            elif consecutive > 0:
                status = "degraded"
            else:
                status = "healthy"
        else:
            age_hours = None
            status = "no_data"

        collectors_status.append({
            "name": name,
            "label": cdef["label"],
            "status": status,
            "critical": cdef["critical"],
            "last_success": state.get("last_success"),
            "age_hours": round(age_hours, 2) if age_hours is not None else None,
            "max_age_hours": cdef["max_age_hours"],
            "consecutive_failures": consecutive,
            "last_error": state.get("last_error"),
        })

    critical_issues = [c for c in collectors_status if c["critical"] and c["status"] in ("stale", "no_data")]
    degraded = [c for c in collectors_status if c["status"] == "degraded"]
    stale = [c for c in collectors_status if c["status"] == "stale"]
    no_data = [c for c in collectors_status if c["status"] == "no_data"]

    if critical_issues:
        overall = "critical"
    elif degraded or stale:
        overall = "degraded"
    elif no_data:
        overall = "initializing"
    else:
        overall = "healthy"

    return {
        "overall": overall,
        "collectors": collectors_status,
        "summary": {
            "total": len(collectors_status),
            "healthy": sum(1 for c in collectors_status if c["status"] == "healthy"),
            "degraded": len(degraded),
            "stale": len(stale),
            "no_data": len(no_data),
            "critical_issues": [c["name"] for c in critical_issues],
        },
        "checked_at": now.isoformat(),
    }
