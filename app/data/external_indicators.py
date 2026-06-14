"""外部市场结构指标的目录与入库工具。"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import ExternalMarketIndicator


INDICATOR_CATALOG: list[dict[str, str | bool]] = [
    {
        "indicator_id": "GLD_FLOW_TONNES",
        "name": "SPDR GLD ETF 资金流",
        "category": "ETF",
        "unit": "tonnes",
        "scored": True,
        "reason": "官方或可信来源入库后参与评分，反映黄金 ETF 边际配置需求。",
    },
    {
        "indicator_id": "COMEX_REGISTERED_GOLD_OZ",
        "name": "COMEX 注册库存",
        "category": "期货结构",
        "unit": "oz",
        "scored": False,
        "reason": "CME 官方页面可能有访问限制；当前可展示，后续接入授权源或手动录入后再评分。",
    },
    {
        "indicator_id": "COMEX_GOLD_FRONT_SPREAD_PCT",
        "name": "COMEX 期限结构",
        "category": "期货结构",
        "unit": "%",
        "scored": False,
        "reason": "需要稳定期货曲线数据源；当前可手动维护，完善后用于观察升贴水和资金成本。",
    },
    {
        "indicator_id": "GEO_RISK_INTENSITY",
        "name": "地缘风险强度",
        "category": "风险事件",
        "unit": "score",
        "scored": False,
        "reason": "需要事件强度和持续时间模型；后续可接入新闻事件或人工评分。",
    },
    {
        "indicator_id": "INDIA_CHINA_PHYSICAL_DEMAND",
        "name": "中印实物需求",
        "category": "实物需求",
        "unit": "score",
        "scored": False,
        "reason": "需要进口、节庆和本地溢价数据；当前可手动录入作为阶段性观察。",
    },
    {
        "indicator_id": "GOLD_OPTION_IV_30D",
        "name": "黄金期权 30D 隐含波动率",
        "category": "期权",
        "unit": "%",
        "scored": True,
        "reason": "可信来源入库后参与评分，反映尾部行情定价。",
    },
    {
        "indicator_id": "GOLD_OPTION_SKEW_25D",
        "name": "黄金期权 25D 偏度",
        "category": "期权",
        "unit": "vol pts",
        "scored": True,
        "reason": "可信来源入库后参与评分，反映极端上涨/下跌保护需求。",
    },
]

_CATALOG_BY_ID = {item["indicator_id"]: item for item in INDICATOR_CATALOG}


def latest_external_indicators(db: Session, limit: int = 200) -> list[ExternalMarketIndicator]:
    return db.scalars(
        select(ExternalMarketIndicator)
        .order_by(ExternalMarketIndicator.timestamp.desc())
        .limit(max(1, min(int(limit), 1000)))
    ).all()


def upsert_external_indicator(
    db: Session,
    indicator_id: str,
    timestamp: datetime,
    value: float,
    *,
    source: str,
    name: str | None = None,
    category: str | None = None,
    unit: str | None = None,
    note: str = "",
) -> ExternalMarketIndicator:
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    meta = _CATALOG_BY_ID.get(indicator_id, {})
    row = db.scalar(
        select(ExternalMarketIndicator).where(
            ExternalMarketIndicator.indicator_id == indicator_id,
            ExternalMarketIndicator.timestamp == timestamp,
        )
    )
    if row is None:
        row = ExternalMarketIndicator(
            indicator_id=indicator_id,
            timestamp=timestamp,
            value=float(value),
            source=source,
            name=name or str(meta.get("name") or indicator_id),
            category=category or str(meta.get("category") or ""),
            unit=unit or str(meta.get("unit") or ""),
            note=note,
        )
        db.add(row)
    else:
        row.value = float(value)
        row.source = source
        row.name = name or row.name or str(meta.get("name") or indicator_id)
        row.category = category or row.category or str(meta.get("category") or "")
        row.unit = unit or row.unit or str(meta.get("unit") or "")
        row.note = note
    return row
