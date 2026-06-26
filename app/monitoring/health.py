from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.data.fred_client import FRED_SERIES
from app.data_quality import source_quality
from app.models import (
    CentralBankGold,
    CftcPosition,
    ChinaGoldPremium,
    GoldPrice,
    GoldScoreSnapshot,
    MacroObservation,
    NewsSentiment,
)

CRITICAL_FRED_SERIES = {"DGS10", "DFII10", "T10YIE", "FEDFUNDS", "VIXCLS", "DTWEXBGS"}
FREQUENCY_THRESHOLDS_DAYS = {
    "daily": (7, 30),
    "weekly": (21, 45),
    "monthly": (60, 95),
    "quarterly": (140, 220),
    "annual": (430, 760),
}


def _utc_naive_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _age_days(timestamp: datetime | None) -> float | None:
    if timestamp is None:
        return None
    normalized = timestamp.replace(tzinfo=None) if timestamp.tzinfo else timestamp
    return round((_utc_naive_now() - normalized).total_seconds() / 86400, 2)


def _status_from_age(age_days: float | None, warn_after_days: int, error_after_days: int) -> str:
    if age_days is None:
        return "error"
    if age_days > error_after_days:
        return "error"
    if age_days > warn_after_days:
        return "warn"
    return "ok"


def _overall_status(items: list[dict[str, Any]]) -> str:
    statuses = {item["status"] for item in items if item.get("critical", True)}
    if "error" in statuses:
        return "error"
    if "warn" in statuses:
        return "warn"
    if any(item["status"] == "warn" for item in items):
        return "warn"
    return "ok"


def _quality_fields(source: str | None) -> dict[str, Any]:
    quality = source_quality(source)
    return {
        "quality_tier": quality.tier,
        "quality_label": quality.label,
        "can_score": quality.can_score,
    }


def get_data_health(db: Session) -> dict[str, Any]:
    items: list[dict[str, Any]] = []

    # FRED 指标
    for series in FRED_SERIES:
        row = db.scalar(
            select(MacroObservation)
            .where(MacroObservation.series_id == series.series_id)
            .order_by(MacroObservation.timestamp.desc())
        )
        age = _age_days(row.timestamp if row else None)
        warn_after, error_after = FREQUENCY_THRESHOLDS_DAYS.get(
            series.frequency or "daily",
            FREQUENCY_THRESHOLDS_DAYS["daily"],
        )
        status = _status_from_age(age, warn_after_days=warn_after, error_after_days=error_after)
        source = row.source if row else None
        critical = series.series_id in CRITICAL_FRED_SERIES
        items.append(
            {
                "name": series.name,
                "kind": "FRED",
                "key": series.series_id,
                "status": status,
                "latest_timestamp": row.timestamp if row else None,
                "age_days": age,
                "source": source,
                "message": "数据正常" if status == "ok" else "FRED 指标缺失或过期",
                "critical": critical,
                **_quality_fields(source),
            }
        )

    # 黄金价格
    gold = db.scalar(select(GoldPrice).order_by(GoldPrice.date.desc()))
    gold_age = _age_days(gold.date if gold else None)
    gold_status = _status_from_age(gold_age, warn_after_days=5, error_after_days=14)
    items.append(
        {
            "name": "黄金价格",
            "kind": "GOLD_PRICE",
            "key": "GC=F",
            "status": gold_status,
            "latest_timestamp": gold.date if gold else None,
            "age_days": gold_age,
            "source": gold.source if gold else None,
            "message": "数据正常" if gold_status == "ok" else "黄金价格缺失或过期",
            "critical": True,
            **_quality_fields(gold.source if gold else None),
        }
    )

    # CFTC
    cftc = db.scalar(select(CftcPosition).order_by(CftcPosition.timestamp.desc()))
    cftc_age = _age_days(cftc.timestamp if cftc else None)
    cftc_status = _status_from_age(cftc_age, warn_after_days=14, error_after_days=35)
    items.append(
        {
            "name": "CFTC 黄金期货持仓",
            "kind": "CFTC",
            "key": "088691",
            "status": cftc_status,
            "latest_timestamp": cftc.timestamp if cftc else None,
            "age_days": cftc_age,
            "source": cftc.source if cftc else None,
            "message": "数据正常" if cftc_status == "ok" else "CFTC 持仓缺失或过期",
            "critical": True,
            **_quality_fields(cftc.source if cftc else None),
        }
    )

    # 中国黄金溢价
    premium = db.scalar(select(ChinaGoldPremium).order_by(ChinaGoldPremium.timestamp.desc()))
    prem_age = _age_days(premium.timestamp if premium else None)
    prem_status = "warn" if premium is None else _status_from_age(prem_age, warn_after_days=3, error_after_days=10)
    premium_quality = _quality_fields(premium.source if premium else None)
    premium_can_score = (premium.source if premium else "").upper() in {"SGE", "LBMA", "TEST"}
    premium_message = "数据正常" if prem_status == "ok" else "中国溢价未接入或过期"
    if premium and not premium_can_score:
        prem_status = "warn"
        premium_message = "中国溢价为非官方估算，暂不参与评分"
    items.append(
        {
            "name": "中国黄金溢价",
            "kind": "PREMIUM",
            "key": "china_premium",
            "status": prem_status,
            "latest_timestamp": premium.timestamp if premium else None,
            "age_days": prem_age,
            "source": premium.source if premium else None,
            "message": premium_message,
            "critical": False,
            **premium_quality,
            "can_score": premium_can_score,
        }
    )

    # 央行购金
    cb = db.scalar(
        select(CentralBankGold)
        .where(CentralBankGold.country == "Global")
        .order_by(CentralBankGold.timestamp.desc())
    )
    cb_age = _age_days(cb.timestamp if cb else None)
    cb_status = "warn" if cb is None else _status_from_age(cb_age, warn_after_days=120, error_after_days=200)
    cb_quality = _quality_fields(cb.source if cb else None)
    if cb and not cb_quality["can_score"]:
        cb_status = "warn"
    items.append(
        {
            "name": "全球央行购金",
            "kind": "CB_GOLD",
            "key": "global_cb_gold",
            "status": cb_status,
            "latest_timestamp": cb.timestamp if cb else None,
            "age_days": cb_age,
            "source": cb.source if cb else None,
            "message": "数据正常" if cb_status == "ok" else "央行购金未接入、过期或来源不可评分",
            "critical": False,
            **cb_quality,
        }
    )

    # 新闻情绪
    sent = db.scalar(select(NewsSentiment).order_by(NewsSentiment.timestamp.desc()))
    sent_age = _age_days(sent.timestamp if sent else None)
    sent_status = "warn" if sent is None else _status_from_age(sent_age, warn_after_days=2, error_after_days=7)
    sent_quality = _quality_fields(sent.source if sent else None)
    if sent and not sent_quality["can_score"]:
        sent_status = "warn"
    items.append(
        {
            "name": "新闻情绪",
            "kind": "SENTIMENT",
            "key": "news_sentiment",
            "status": sent_status,
            "latest_timestamp": sent.timestamp if sent else None,
            "age_days": sent_age,
            "source": sent.source if sent else None,
            "message": "数据正常（NewsAPI/GDELT）" if sent_status == "ok" else "新闻情绪未接入、过期或来源不可评分",
            "critical": False,
            **sent_quality,
        }
    )

    # 评分
    score = db.scalar(select(GoldScoreSnapshot).order_by(GoldScoreSnapshot.timestamp.desc()))
    score_age = _age_days(score.timestamp if score else None)
    score_status = _status_from_age(score_age, warn_after_days=2, error_after_days=7)
    items.append(
        {
            "name": "黄金多空评分",
            "kind": "SCORE",
            "key": "gold_score",
            "status": score_status,
            "latest_timestamp": score.timestamp if score else None,
            "age_days": score_age,
            "source": score.source if score else None,
            "message": "评分正常" if score_status == "ok" else "评分缺失或过期",
            "critical": True,
            **_quality_fields(score.source if score else None),
        }
    )

    return {
        "ok": _overall_status(items) != "error",
        "status": _overall_status(items),
        "checked_at": _utc_naive_now(),
        "items": items,
    }
