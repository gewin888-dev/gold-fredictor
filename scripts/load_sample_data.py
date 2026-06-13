from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy import select

from app.data.cftc_client import CftcPositionRecord
from app.data.cftc_collector import upsert_cftc_position
from app.data.cb_gold_collector import load_sample_cb_gold
from app.data.fred_client import FRED_SERIES
from app.data.sentiment_collector import load_sample_sentiment
from app.data.sge_collector import load_sample_china_premium
from app.database import SessionLocal, init_db
from app.events.calendar import load_sample_macro_events
from app.models import GoldPrice, GoldScoreSnapshot, MacroObservation, MacroSeries


def sample_value(series_id: str, index: int) -> float:
    values = {
        "GOLDAMGBD228NLBM": 2350 + index * 2.3,
        "DGS10": 4.4 - index * 0.004,
        "DFII10": 2.1 - index * 0.005,
        "T10YIE": 2.2 + index * 0.002,
        "FEDFUNDS": 5.3 - index * 0.001,
        "VIXCLS": 14 + index * 0.05,
        "DTWEXBGS": 122 - index * 0.045,
    }
    return float(values[series_id])


def upsert_macro_series(db) -> None:
    for series in FRED_SERIES:
        stmt = sqlite_insert(MacroSeries).values(
            series_id=series.series_id,
            name=series.name,
            frequency=series.frequency,
            unit=series.unit,
            source="SAMPLE",
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["series_id"],
            set_={
                "name": series.name,
                "frequency": series.frequency,
                "unit": series.unit,
                "source": "SAMPLE",
            },
        )
        db.execute(stmt)


def upsert_observation(db, series_id: str, timestamp: datetime, value: float) -> None:
    stmt = sqlite_insert(MacroObservation).values(
        series_id=series_id,
        timestamp=timestamp,
        value=value,
        source="SAMPLE",
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=["series_id", "timestamp"],
        set_={"value": value, "source": "SAMPLE"},
    )
    db.execute(stmt)


def upsert_gold_price(db, timestamp: datetime, close: float) -> None:
    stmt = sqlite_insert(GoldPrice).values(
        date=timestamp,
        open=close - 3,
        high=close + 8,
        low=close - 8,
        close=close,
        source="SAMPLE",
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=["date"],
        set_={
            "open": close - 3,
            "high": close + 8,
            "low": close - 8,
            "close": close,
            "source": "SAMPLE",
        },
    )
    db.execute(stmt)


def insert_sample_scores(db, start: datetime) -> None:
    score_rows = [
        (60, 35.0, "偏多"),
        (70, 42.0, "偏多"),
        (80, 12.0, "中性"),
        (90, -34.0, "偏空"),
        (100, 38.0, "偏多"),
    ]
    for day_offset, score, direction in score_rows:
        timestamp = start + timedelta(days=day_offset)
        existing = db.scalar(
            select(GoldScoreSnapshot).where(
                GoldScoreSnapshot.timestamp == timestamp,
                GoldScoreSnapshot.source == "SAMPLE",
            )
        )
        if existing:
            existing.total_score = score
            existing.direction = direction
            continue
        db.add(
            GoldScoreSnapshot(
                timestamp=timestamp,
                total_score=score,
                direction=direction,
                factor_scores="{}",
                risk_flags="[]",
                summary="样例评分快照，仅用于本地演示。",
                source="SAMPLE",
            )
        )


if __name__ == "__main__":
    init_db()
    db = SessionLocal()
    try:
        upsert_macro_series(db)
        start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=119)
        for series in FRED_SERIES:
            for index in range(120):
                timestamp = start + timedelta(days=index)
                upsert_observation(
                    db,
                    series.series_id,
                    timestamp,
                    sample_value(series.series_id, index),
                )
        for index in range(120):
            timestamp = start + timedelta(days=index)
            upsert_gold_price(db, timestamp, sample_value("GOLDAMGBD228NLBM", index))
        upsert_cftc_position(
            db,
            CftcPositionRecord(
                market_name="GOLD - COMMODITY EXCHANGE INC.",
                contract_market_code="088691",
                exchange_code="CMX",
                timestamp=datetime(2025, 12, 30, tzinfo=timezone.utc),
                open_interest=326052,
                noncommercial_long=206096,
                noncommercial_short=30076,
                noncommercial_spreading=22449,
                commercial_long=53851,
                commercial_short=260196,
            ),
            source="SAMPLE",
        )
        insert_sample_scores(db, start)
        load_sample_macro_events(db)
        load_sample_china_premium(db)
        load_sample_cb_gold(db)
        load_sample_sentiment(db)
        db.commit()
    finally:
        db.close()
    print("Sample macro data loaded. You can now run: python scripts/compute_score.py")
