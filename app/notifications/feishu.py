from __future__ import annotations

import base64
import hashlib
import hmac
import time
from typing import Any
import json

import requests

from app.config import get_settings
from app.models import GoldScoreSnapshot


def _sign(timestamp: str, secret: str) -> str:
    string_to_sign = f"{timestamp}\n{secret}"
    digest = hmac.new(secret.encode("utf-8"), string_to_sign.encode("utf-8"), digestmod=hashlib.sha256).digest()
    return base64.b64encode(digest).decode("utf-8")


def send_text_message(text: str) -> dict[str, Any]:
    settings = get_settings()
    if not settings.feishu_webhook_url:
        return {"ok": True, "skipped": True, "reason": "FEISHU_WEBHOOK_URL is not configured."}

    payload: dict[str, Any] = {"msg_type": "text", "content": {"text": text}}
    if settings.feishu_secret:
        timestamp = str(int(time.time()))
        payload["timestamp"] = timestamp
        payload["sign"] = _sign(timestamp, settings.feishu_secret)

    response = requests.post(settings.feishu_webhook_url, json=payload, timeout=15)
    response.raise_for_status()
    return {"ok": True, "skipped": False, "response": response.json()}


def send_score_alert(score_snapshot: GoldScoreSnapshot) -> dict[str, Any]:
    return send_text_message(build_score_alert_text(score_snapshot))


def build_score_alert_text(
    score_snapshot: GoldScoreSnapshot,
    data_health: dict[str, Any] | None = None,
    upcoming_events: list[dict[str, Any]] | None = None,
    collector_status: str = "",
) -> str:
    from datetime import datetime, timedelta, timezone

    ts = score_snapshot.timestamp
    utc_str = ts.strftime("%Y-%m-%d %H:%M UTC") if hasattr(ts, "strftime") else str(ts)
    bj_ts = ts + timedelta(hours=8) if ts.tzinfo else ts.replace(tzinfo=timezone.utc) + timedelta(hours=8)
    bj_str = bj_ts.strftime("%Y-%m-%d %H:%M") if hasattr(bj_ts, "strftime") else str(bj_ts)

    factor_scores_raw = json.loads(score_snapshot.factor_scores)
    # Handle v2 format: {"scores": {...}, "details": {...}}
    factor_scores = factor_scores_raw.get("scores", factor_scores_raw) if isinstance(factor_scores_raw, dict) else {}
    risk_flags = json.loads(score_snapshot.risk_flags)
    factor_lines = "\n".join(
        f"- {name}: {value}" for name, value in sorted(factor_scores.items(), key=lambda item: abs(item[1]) if isinstance(item[1], (int, float)) else 0, reverse=True)
    )
    risk_lines = "\n".join(f"- {flag}" for flag in risk_flags)

    health_status = "未检查"
    if data_health:
        health_status = data_health.get("status", "unknown")
    event_lines = ""
    if upcoming_events:
        event_lines = "\n".join(
            f"- {event.get('timestamp')}: {event.get('name')} ({event.get('importance')})"
            for event in upcoming_events[:5]
        )

    return (
        f"📊 黄金走势监控报告\n"
        f"UTC  {utc_str}\n"
        f"北京 {bj_str}\n\n"
        f"评分: {score_snapshot.total_score}\n"
        f"方向: {score_snapshot.direction}\n"
        f"数据健康: {health_status}\n"
        f"采集器: {collector_status or '未检查'}\n"
        f"摘要: {score_snapshot.summary}\n\n"
        f"主要因子:\n"
        f"{factor_lines or '- 暂无因子'}\n\n"
        f"风险提示:\n"
        f"{risk_lines or '- 暂无风险提示'}\n\n"
        f"未来事件:\n"
        f"{event_lines or '- 暂无未来事件'}\n\n"
        f"说明: 本消息用于验证 AI 黄金预测系统可行性，不用于黄金买卖参考。"
    )


def send_score_alert_with_health(
    score_snapshot: GoldScoreSnapshot,
    data_health: dict[str, Any] | None = None,
    upcoming_events: list[dict[str, Any]] | None = None,
    collector_status: str = "",
) -> dict[str, Any]:
    return send_text_message(
        build_score_alert_text(
            score_snapshot,
            data_health=data_health,
            upcoming_events=upcoming_events,
            collector_status=collector_status,
        )
    )
