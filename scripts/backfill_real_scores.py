from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import pandas as pd
from sqlalchemy import delete, select

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.database import SessionLocal, init_db
from app.models import CftcPosition, GoldPrice, GoldScoreSnapshot, MacroObservation
from app.scoring.gold_score import DOLLAR, INFLATION_EXPECTATION, NOMINAL_RATE, REAL_RATE, VIX, _clamp, _direction

SOURCE = "backfill_real_v2"
REQUIRED = [REAL_RATE, NOMINAL_RATE, INFLATION_EXPECTATION, VIX, DOLLAR]


def _series_frame(db, series_id: str) -> pd.DataFrame:
    rows = db.scalars(
        select(MacroObservation)
        .where(MacroObservation.series_id == series_id, MacroObservation.source == "FRED")
        .order_by(MacroObservation.timestamp.asc())
    ).all()
    frame = pd.DataFrame([{"timestamp": r.timestamp, series_id: r.value} for r in rows])
    if not frame.empty:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


def _latest_cftc_score(cftc_rows: list[CftcPosition], timestamp: datetime) -> tuple[float | None, str | None]:
    ts = pd.Timestamp(timestamp).to_pydatetime()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    candidates = []
    for row in cftc_rows:
        row_ts = row.timestamp
        if row_ts.tzinfo is None:
            row_ts = row_ts.replace(tzinfo=timezone.utc)
        if row_ts <= ts:
            candidates.append(row)
    if not candidates:
        return None, None
    latest = candidates[-1]
    latest_ts = latest.timestamp if latest.timestamp.tzinfo else latest.timestamp.replace(tzinfo=timezone.utc)
    age_days = (ts - latest_ts).total_seconds() / 86400
    if age_days > 35 or latest.open_interest <= 0:
        return None, None
    net_ratio = latest.noncommercial_net / latest.open_interest
    return round(_clamp(net_ratio * 30.0, -15, 15), 2), f"CFTC 非商业净持仓占总持仓约 {net_ratio:.1%}。"


def _score_row(window: pd.DataFrame, cftc_rows: list[CftcPosition]) -> tuple[float, str, dict[str, float], list[str]]:
    latest = window.iloc[-1]
    factor_scores: dict[str, float] = {}
    risk_flags: list[str] = []

    factor_scores["实际利率"] = round(_clamp(-(latest[REAL_RATE] - window[REAL_RATE].iloc[-21]) * 30, -25, 25), 2)
    factor_scores["名义利率"] = round(_clamp(-(latest[NOMINAL_RATE] - window[NOMINAL_RATE].iloc[-21]) * 40, -20, 20), 2)
    if latest[NOMINAL_RATE] >= 4.5:
        risk_flags.append("10年美债收益率处于高位，黄金持有机会成本较高。")

    dollar_change_pct = (latest[DOLLAR] / window[DOLLAR].iloc[-21] - 1) * 100
    factor_scores["美元指数"] = round(_clamp(-dollar_change_pct * 4, -20, 20), 2)

    factor_scores["避险情绪"] = round(_clamp((latest[VIX] - window[VIX].iloc[-21]) * 1.2, -15, 15), 2)
    if latest[VIX] >= 25:
        risk_flags.append("VIX 处于较高水平，市场避险波动上升。")

    factor_scores["通胀预期"] = round(
        _clamp((latest[INFLATION_EXPECTATION] - window[INFLATION_EXPECTATION].iloc[-21]) * 25, -15, 15),
        2,
    )

    gold_ma20 = window["gold_price"].tail(20).mean()
    gold_ma60 = window["gold_price"].tail(60).mean()
    trend_score = 8 if latest["gold_price"] > gold_ma20 else -8
    trend_score += 12 if gold_ma20 > gold_ma60 else -12
    factor_scores["黄金趋势"] = float(trend_score)

    cftc_score, cftc_note = _latest_cftc_score(cftc_rows, latest["timestamp"])
    if cftc_score is not None:
        factor_scores["CFTC投机仓位"] = cftc_score
    if cftc_note:
        risk_flags.append(cftc_note)

    if latest[REAL_RATE] >= 2.0:
        risk_flags.append("实际利率处于较高水平，可能压制无息资产估值。")
    if abs(dollar_change_pct) >= 2:
        risk_flags.append("美元指数近 20 个交易日波动较大，需关注汇率因子扰动。")
    if not risk_flags:
        risk_flags.append("当前未触发显著宏观风险阈值。")

    total = round(_clamp(sum(factor_scores.values()), -100, 100), 2)
    return total, _direction(total), factor_scores, risk_flags


def backfill_real_scores(days: int = 730) -> int:
    init_db()
    with SessionLocal() as db:
        frames = [_series_frame(db, sid) for sid in REQUIRED]
        if any(frame.empty for frame in frames):
            missing = [sid for sid, frame in zip(REQUIRED, frames) if frame.empty]
            raise RuntimeError(f"Missing real FRED series: {', '.join(missing)}")

        merged = frames[0].sort_values("timestamp")
        for frame in frames[1:]:
            merged = pd.merge_asof(merged, frame.sort_values("timestamp"), on="timestamp", direction="backward")

        gold_rows = db.scalars(
            select(GoldPrice)
            .where(GoldPrice.source == "YAHOO")
            .order_by(GoldPrice.date.asc())
        ).all()
        if not gold_rows:
            raise RuntimeError("Missing YAHOO gold prices.")

        gold = pd.DataFrame([{"timestamp": row.date, "gold_price": row.close} for row in gold_rows])
        gold["timestamp"] = pd.to_datetime(gold["timestamp"], utc=True)
        cutoff = pd.Timestamp(datetime.now(timezone.utc) - timedelta(days=days))
        gold = gold[gold["timestamp"] >= cutoff].sort_values("timestamp")

        merged = pd.merge_asof(gold, merged.sort_values("timestamp"), on="timestamp", direction="backward").dropna()
        if len(merged) < 80:
            raise RuntimeError(f"Not enough aligned real rows: {len(merged)}")

        cftc_rows = db.scalars(
            select(CftcPosition)
            .where(CftcPosition.source == "CFTC")
            .order_by(CftcPosition.timestamp.asc())
        ).all()

        db.execute(delete(GoldScoreSnapshot).where(GoldScoreSnapshot.source == SOURCE))
        count = 0
        for index in range(59, len(merged)):
            window = merged.iloc[: index + 1]
            timestamp = window.iloc[-1]["timestamp"]
            total, direction, factors, risks = _score_row(window, cftc_rows)
            db.add(
                GoldScoreSnapshot(
                    timestamp=timestamp,
                    total_score=total,
                    direction=direction,
                    factor_scores=json.dumps(factors, ensure_ascii=False),
                    risk_flags=json.dumps(risks, ensure_ascii=False),
                    summary=f"黄金多空评分为 {total}，方向为{direction}。该结果仅用于数据分析和风险提示。",
                    source=SOURCE,
                )
            )
            count += 1
        db.commit()
        return count


if __name__ == "__main__":
    count = backfill_real_scores()
    print(f"Backfilled {count} real score snapshots.")
