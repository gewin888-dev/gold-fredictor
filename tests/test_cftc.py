from datetime import datetime, timezone

from sqlalchemy import func, select

from app.data.cftc_client import parse_legacy_futures_only
from app.data.cftc_collector import upsert_cftc_position
from app.models import CftcPosition


SAMPLE_CFTC_TEXT = (
    '"GOLD - COMMODITY EXCHANGE INC.",260602,2026-06-02,088691,CMX ,01,088 ,'
    '  326052,  206096,   30076,   22449,   53851,  260196,  282396,  312721,'
    '   43656,   13331,  326052\n'
)


def test_parse_legacy_futures_only_gold_row():
    record = parse_legacy_futures_only(SAMPLE_CFTC_TEXT)

    assert record.market_name == "GOLD - COMMODITY EXCHANGE INC."
    assert record.contract_market_code == "088691"
    assert record.timestamp == datetime(2026, 6, 2, tzinfo=timezone.utc)
    assert record.open_interest == 326052
    assert record.noncommercial_net == 176020


def test_upsert_cftc_position_updates_existing_row(db_session):
    record = parse_legacy_futures_only(SAMPLE_CFTC_TEXT)

    upsert_cftc_position(db_session, record, source="TEST")
    upsert_cftc_position(db_session, record, source="TEST")
    db_session.commit()

    count = db_session.scalar(select(func.count()).select_from(CftcPosition))
    row = db_session.scalar(select(CftcPosition))

    assert count == 1
    assert row.noncommercial_net == 176020
    assert row.source == "TEST"
