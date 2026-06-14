from datetime import datetime, timedelta, timezone

from app import database
from app.data import gold_price_collector
from app.models import IntradaySnapshot


def test_fetch_gold_intraday_returns_full_timestamps_and_coverage(db_session, monkeypatch):
    start = datetime.now(timezone.utc) - timedelta(hours=2)
    for index, price in enumerate([2400.0, 2401.0, 2402.0]):
        db_session.add(
            IntradaySnapshot(
                timestamp=start + timedelta(minutes=index * 5),
                price=price,
                high=price,
                low=price,
                source="TEST",
            )
        )
    db_session.commit()

    monkeypatch.setattr(database, "SessionLocal", lambda: db_session)

    result = gold_price_collector.fetch_gold_intraday(interval_minutes=5)

    assert result["ok"] is True
    assert result["point_count"] == 3
    assert result["coverage_hours"] > 0
    assert result["is_flat"] is False
    assert "timestamp" in result["points"][0]
    assert result["points"][0]["time"].count(":") == 1


def test_comex_market_closed_weekend_window():
    assert gold_price_collector.is_comex_market_closed(
        datetime(2026, 6, 13, 16, 0, tzinfo=timezone.utc)
    )
    assert gold_price_collector.is_comex_market_closed(
        datetime(2026, 6, 14, 12, 0, tzinfo=timezone.utc)
    )
    assert not gold_price_collector.is_comex_market_closed(
        datetime(2026, 6, 14, 22, 0, tzinfo=timezone.utc)
    )
