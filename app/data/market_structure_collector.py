"""ETF/COMEX/期权等市场结构指标采集入口。

免费公开源对 COMEX 库存和期货曲线经常有访问限制，因此这里的原则是：
能稳定取得的数据才入库；不能取得时返回结构化原因，供前端灰色展示和手动录入。
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.data.external_indicators import upsert_external_indicator
from app.models import ExternalMarketIndicator


def _latest_indicator(db: Session, indicator_id: str) -> ExternalMarketIndicator | None:
    return db.scalar(
        select(ExternalMarketIndicator)
        .where(ExternalMarketIndicator.indicator_id == indicator_id)
        .order_by(ExternalMarketIndicator.timestamp.desc())
    )


def _derive_gld_flow_from_holdings(db: Session) -> dict[str, object]:
    """从手动/授权录入的 GLD_HOLDINGS_TONNES 推导最近一次吨数变化。"""
    rows = db.scalars(
        select(ExternalMarketIndicator)
        .where(ExternalMarketIndicator.indicator_id == "GLD_HOLDINGS_TONNES")
        .order_by(ExternalMarketIndicator.timestamp.desc())
        .limit(2)
    ).all()
    if len(rows) < 2:
        latest_flow = _latest_indicator(db, "GLD_FLOW_TONNES")
        if latest_flow:
            return {
                "ok": True,
                "indicator_id": "GLD_FLOW_TONNES",
                "value": latest_flow.value,
                "timestamp": latest_flow.timestamp.isoformat() if latest_flow.timestamp else None,
                "source": latest_flow.source,
                "note": "使用最近已入库的 GLD 资金流数据。",
            }
        return {
            "ok": False,
            "indicator_id": "GLD_FLOW_TONNES",
            "reason": "尚无可推导资金流的 GLD 持仓历史；可通过 /external/indicators 手动录入 GLD_FLOW_TONNES。",
        }

    newest, previous = rows[0], rows[1]
    flow = float(newest.value) - float(previous.value)
    row = upsert_external_indicator(
        db,
        "GLD_FLOW_TONNES",
        newest.timestamp,
        flow,
        source=newest.source,
        note=f"由 GLD_HOLDINGS_TONNES 相邻两期推导：{previous.value:.2f} -> {newest.value:.2f} 吨。",
    )
    return {
        "ok": True,
        "indicator_id": "GLD_FLOW_TONNES",
        "value": row.value,
        "timestamp": row.timestamp.isoformat() if row.timestamp else None,
        "source": row.source,
        "note": row.note,
    }


def collect_market_structure_data(db: Session) -> dict[str, dict[str, object]]:
    """采集/推导市场结构指标。

    返回每个指标的独立状态；任何单项失败都不会影响系统整体运行。
    """
    now = datetime.now(timezone.utc)
    results: dict[str, dict[str, object]] = {
        "gld_flow": _derive_gld_flow_from_holdings(db),
        "comex_inventory": {
            "ok": False,
            "indicator_id": "COMEX_REGISTERED_GOLD_OZ",
            "timestamp": now.isoformat(),
            "reason": "CME 官方库存数据当前未配置稳定免费接口；可接入授权源或手动录入后展示。",
        },
        "comex_term_structure": {
            "ok": False,
            "indicator_id": "COMEX_GOLD_FRONT_SPREAD_PCT",
            "timestamp": now.isoformat(),
            "reason": "期货期限结构需要可靠曲线数据；当前预留手动录入口，不参与自动评分。",
        },
        "geopolitical_risk": {
            "ok": False,
            "indicator_id": "GEO_RISK_INTENSITY",
            "timestamp": now.isoformat(),
            "reason": "地缘风险强度模型尚未接入事件持续时间和强度评分；当前仅灰色展示。",
        },
        "physical_demand": {
            "ok": False,
            "indicator_id": "INDIA_CHINA_PHYSICAL_DEMAND",
            "timestamp": now.isoformat(),
            "reason": "中印实物需求需要进口、节庆和本地溢价数据；当前预留手动录入。",
        },
    }
    db.commit()
    return results
