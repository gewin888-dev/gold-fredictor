from __future__ import annotations

from datetime import timedelta
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import configured_prediction_sources
from app.data.utils import gold_price_frame
from app.models import GoldPrice, GoldScoreSnapshot


def _direction_signal(score: float) -> int:
    if score >= 30:
        return 1
    if score <= -30:
        return -1
    return 0


def _score_frame(db: Session, score_sources: set[str] | None = None) -> pd.DataFrame:
    rows = db.scalars(select(GoldScoreSnapshot).order_by(GoldScoreSnapshot.timestamp.asc())).all()
    frame = pd.DataFrame(
        [
            {
                "timestamp": row.timestamp,
                "total_score": row.total_score,
                "direction": row.direction,
                "source": row.source,
            }
            for row in rows
        ]
    )
    if frame.empty:
        return frame
    allowed = score_sources or configured_prediction_sources()
    return frame[frame["source"].isin(allowed)].copy()


def run_score_backtest(
    db: Session,
    horizon_days: int = 20,
    include_trades: bool = True,
    limit: int = 100,
    offset: int = 0,
    score_sources: set[str] | None = None,
) -> dict[str, Any]:
    if horizon_days <= 0:
        raise ValueError("horizon_days must be greater than 0.")
    limit = max(1, min(int(limit), 500))
    offset = max(0, int(offset))

    allowed_sources = score_sources or configured_prediction_sources()
    scores = _score_frame(db, allowed_sources)
    prices = gold_price_frame(db)
    if scores.empty or prices.empty:
        return {
            "ok": False,
            "reason": "Need both score snapshots and gold prices to run backtest.",
            "horizon_days": horizon_days,
            "score_sources": sorted(allowed_sources),
            "trades": [],
            "summary": {},
        }

    scores["timestamp"] = pd.to_datetime(scores["timestamp"])
    prices["timestamp"] = pd.to_datetime(prices["timestamp"])
    scores = scores.sort_values("timestamp")
    prices = prices.sort_values("timestamp")

    entry = pd.merge_asof(scores, prices, on="timestamp", direction="backward")
    entry = entry.dropna(subset=["gold_price"]).rename(columns={"gold_price": "entry_price"})
    if entry.empty:
        return {
            "ok": False,
            "reason": "No score snapshots can be aligned to historical gold prices.",
            "horizon_days": horizon_days,
            "score_sources": sorted(allowed_sources),
            "trades": [],
            "summary": {},
        }

    exit_prices = prices.rename(columns={"timestamp": "target_timestamp", "gold_price": "exit_price"})
    entry["target_timestamp"] = entry["timestamp"] + timedelta(days=horizon_days)
    merged = pd.merge_asof(
        entry.sort_values("target_timestamp"),
        exit_prices.sort_values("target_timestamp"),
        on="target_timestamp",
        direction="forward",
    )
    merged = merged.dropna(subset=["exit_price"])
    if merged.empty:
        return {
            "ok": False,
            "reason": "No score snapshots have enough future gold price data for this horizon.",
            "horizon_days": horizon_days,
            "score_sources": sorted(allowed_sources),
            "trades": [],
            "summary": {},
        }

    merged["future_return_pct"] = (merged["exit_price"] / merged["entry_price"] - 1) * 100
    merged["signal"] = merged["total_score"].apply(_direction_signal)
    directional = merged[merged["signal"] != 0].copy()
    directional["hit"] = directional.apply(
        lambda row: (row["signal"] > 0 and row["future_return_pct"] > 0)
        or (row["signal"] < 0 and row["future_return_pct"] < 0),
        axis=1,
    )

    trades = [
        {
            "timestamp": row.timestamp,
            "target_timestamp": row.target_timestamp,
            "total_score": round(float(row.total_score), 2),
            "direction": row.direction,
            "entry_price": round(float(row.entry_price), 4),
            "exit_price": round(float(row.exit_price), 4),
            "future_return_pct": round(float(row.future_return_pct), 4),
            "signal": int(row.signal),
        }
        for row in merged.itertuples(index=False)
    ]

    summary = {
        "sample_count": int(len(merged)),
        "directional_sample_count": int(len(directional)),
        "average_future_return_pct": round(float(merged["future_return_pct"].mean()), 4),
        "median_future_return_pct": round(float(merged["future_return_pct"].median()), 4),
        "hit_rate": round(float(directional["hit"].mean()), 4) if not directional.empty else None,
        "long_count": int((merged["signal"] == 1).sum()),
        "short_count": int((merged["signal"] == -1).sum()),
        "neutral_count": int((merged["signal"] == 0).sum()),
        "trade_count": int(len(trades)),
    }
    paged_trades = trades[offset: offset + limit] if include_trades else []

    return {
        "ok": True,
        "horizon_days": horizon_days,
        "score_sources": sorted(allowed_sources),
        "summary": summary,
        "pagination": {
            "include_trades": include_trades,
            "limit": limit,
            "offset": offset,
            "returned": len(paged_trades),
            "total": len(trades),
        },
        "trades": paged_trades,
    }
