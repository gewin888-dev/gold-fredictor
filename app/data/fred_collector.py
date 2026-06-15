from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.data.fred_client import FRED_SERIES, FredClient, FredSeriesConfig
from app.models import MacroObservation, MacroSeries

logger = logging.getLogger(__name__)


def upsert_series(db: Session, series: FredSeriesConfig) -> MacroSeries:
    existing = db.scalar(select(MacroSeries).where(MacroSeries.series_id == series.series_id))
    if existing:
        existing.name = series.name
        existing.frequency = series.frequency
        existing.unit = series.unit
        existing.source = "FRED"
        existing.updated_at = datetime.now(timezone.utc)
        return existing

    record = MacroSeries(
        series_id=series.series_id,
        name=series.name,
        frequency=series.frequency,
        unit=series.unit,
        source="FRED",
    )
    db.add(record)
    return record


def upsert_observation(db: Session, series_id: str, timestamp, value: float) -> None:
    timestamp_value = timestamp.to_pydatetime() if hasattr(timestamp, "to_pydatetime") else timestamp
    stmt = sqlite_insert(MacroObservation).values(
        series_id=series_id,
        timestamp=timestamp_value,
        value=float(value),
        source="FRED",
        updated_at=datetime.now(timezone.utc),
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=["series_id", "timestamp"],
        set_={
            "value": float(value),
            "source": "FRED",
            "updated_at": datetime.now(timezone.utc),
        },
    )
    db.execute(stmt)


def collect_fred_data(db: Session, observation_start: str | None = None) -> dict[str, int]:
    """采集 FRED 数据。每个系列独立重试，单点失败不阻塞其他系列。"""
    settings = get_settings()
    start = observation_start or settings.fred_observation_start
    counts: dict[str, int] = {}
    failed = 0
    total = len(FRED_SERIES)

    for series in FRED_SERIES:
        try:
            client = FredClient(timeout=8)
            upsert_series(db, series)
            df = client.get_observations(series.series_id, observation_start=start)
            for row in df.itertuples(index=False):
                upsert_observation(db, series.series_id, row.timestamp, row.value)
            counts[series.series_id] = len(df)
        except Exception as e:
            failed += 1
            logger.warning("FRED series %s failed: %s", series.series_id, str(e)[:100])

    if failed > 0:
        raise RuntimeError(
            f"FRED collection: {total - failed}/{total} succeeded, {failed} failed "
            f"(first error likely Python 3.9 + LibreSSL SSLEOFError)"
        )

    return counts
