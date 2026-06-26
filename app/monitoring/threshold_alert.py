"""因子阈值告警：监测评分/因子突变，超阈值立刻推送。

集成方式：
- 每次 compute_gold_score 后调用 check_threshold_alerts
- 对比上次评分，任一因子或总分变化超过阈值 → 推送告警
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import GoldScoreSnapshot
from app.notifications.feishu import send_text_message


# 告警阈值配置
ALERT_THRESHOLDS = {
    "total_score": 15.0,  # 总分变化超过 15 分
    "factor_change": 8.0,  # 任一因子变化超过 8 分
    "direction_flip": True,  # 方向翻转（偏多 ↔ 偏空）
    "risk_flags_new": True,  # 新增风险提示
}


def _load_previous_score(db: Session) -> GoldScoreSnapshot | None:
    """获取倒数第二条评分快照（最近的是当前）。"""
    rows = db.scalars(
        select(GoldScoreSnapshot)
        .order_by(GoldScoreSnapshot.timestamp.desc())
        .limit(2)
    ).all()
    if len(rows) < 2:
        return None
    return rows[1]  # 倒数第二条 = 上一次


def check_threshold_alerts(
    db: Session,
    current: GoldScoreSnapshot,
) -> list[str]:
    """检查评分是否触发阈值告警，返回告警消息列表。"""
    alerts: list[str] = []
    previous = _load_previous_score(db)

    if previous is None:
        return alerts  # 首次评分不告警

    # 总分突变
    score_delta = abs(current.total_score - previous.total_score)
    if score_delta >= ALERT_THRESHOLDS["total_score"]:
        direction_str = "上升" if current.total_score > previous.total_score else "下降"
        alerts.append(
            f"⚠️ 黄金多空评分{direction_str} {score_delta:.0f} 分 "
            f"({previous.total_score:.0f} → {current.total_score:.0f})"
        )

    # 方向翻转
    if ALERT_THRESHOLDS["direction_flip"] and previous.direction != current.direction:
        if previous.direction == "偏多" and current.direction == "偏空":
            alerts.append("🔴 评分方向翻转：偏多 → 偏空")
        elif previous.direction == "偏空" and current.direction == "偏多":
            alerts.append("🟢 评分方向翻转：偏空 → 偏多")

    # 因子突变
    try:
        prev_factors = json.loads(previous.factor_scores)
        curr_factors = json.loads(current.factor_scores)
        # Handle v2 format: {"scores": {...}, "details": {...}}
        prev_scores = prev_factors.get("scores", prev_factors) if isinstance(prev_factors, dict) else {}
        curr_scores = curr_factors.get("scores", curr_factors) if isinstance(curr_factors, dict) else {}
        for name, curr_val in curr_scores.items():
            prev_val = prev_scores.get(name, 0) if isinstance(prev_scores, dict) else 0
            delta = abs(curr_val - prev_val)
            if delta >= ALERT_THRESHOLDS["factor_change"]:
                direction_str = "↑" if curr_val > prev_val else "↓"
                alerts.append(
                    f"📊 因子「{name}」突变 {direction_str}{delta:.1f} 分 "
                    f"({prev_val:.1f} → {curr_val:.1f})"
                )
    except (json.JSONDecodeError, KeyError):
        pass

    # 新增风险提示
    if ALERT_THRESHOLDS["risk_flags_new"]:
        try:
            prev_risks = set(json.loads(previous.risk_flags))
            curr_risks = set(json.loads(current.risk_flags))
            new_risks = curr_risks - prev_risks
            for risk in new_risks:
                if "未触发" not in risk:
                    alerts.append(f"🚨 新增风险：{risk}")
        except (json.JSONDecodeError, KeyError):
            pass

    return alerts


def send_threshold_alerts(db: Session, snapshot: GoldScoreSnapshot) -> dict[str, Any]:
    """检查并发送阈值告警。"""
    alerts = check_threshold_alerts(db, snapshot)
    if not alerts:
        return {"ok": True, "alerts_sent": 0, "message": "No alerts triggered."}

    alert_text = "🔔 黄金走势监控 — 因子告警\n\n" + "\n".join(alerts) + "\n\n⚠️ 用于验证 AI 黄金预测系统可行性，不用于黄金买卖参考。"

    result = send_text_message(alert_text)

    return {
        "ok": True,
        "alerts_sent": len(alerts),
        "alerts": alerts,
        "sent": result.get("skipped", False) is False,
    }
