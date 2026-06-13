from datetime import datetime, timezone

from sqlalchemy import func, select

from app.data.fred_collector import upsert_observation
from app.models import MacroObservation


def test_upsert_observation_updates_existing_row(db_session):
    timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)

    upsert_observation(db_session, "DFII10", timestamp, 2.1)
    upsert_observation(db_session, "DFII10", timestamp, 1.9)
    db_session.commit()

    count = db_session.scalar(select(func.count()).select_from(MacroObservation))
    row = db_session.scalar(select(MacroObservation).where(MacroObservation.series_id == "DFII10"))

    assert count == 1
    assert row.value == 1.9
    assert row.source == "FRED"
