from datetime import datetime, timezone

from app.events.calendar import list_macro_events, load_sample_macro_events


def test_load_sample_macro_events_is_queryable(db_session):
    count = load_sample_macro_events(db_session, base_time=datetime(2026, 6, 11, tzinfo=timezone.utc))

    events = list_macro_events(db_session, days_ahead=40)

    assert count == 4
    assert len(events) == 4
    assert events[0].name == "美国 CPI 数据"


def test_load_sample_macro_events_is_idempotent(db_session):
    base = datetime(2026, 6, 11, tzinfo=timezone.utc)

    load_sample_macro_events(db_session, base_time=base)
    load_sample_macro_events(db_session, base_time=base)
    events = list_macro_events(db_session, days_ahead=40)

    assert len(events) == 4
