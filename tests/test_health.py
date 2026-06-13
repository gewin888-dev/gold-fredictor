from datetime import datetime, timezone

from app.models import GoldScoreSnapshot
from app.data.cb_gold_collector import load_sample_cb_gold
from app.data.sentiment_collector import load_sample_sentiment
from app.data.sge_collector import load_sample_china_premium
from app.monitoring.health import get_data_health
from conftest import insert_cftc_position, insert_gold_prices, insert_required_macro_observations


def test_data_health_reports_error_on_empty_database(db_session):
    result = get_data_health(db_session)

    assert result["ok"] is False
    assert result["status"] == "error"
    assert any(item["status"] == "error" for item in result["items"])


def test_data_health_reports_ok_with_current_sample_data(db_session):
    now = datetime.now(timezone.utc)
    insert_required_macro_observations(db_session, end_at=now)
    insert_gold_prices(db_session, end_at=now)
    insert_cftc_position(db_session, timestamp=now)
    load_sample_china_premium(db_session, days=3)
    load_sample_cb_gold(db_session, quarters=2)
    load_sample_sentiment(db_session, days=3)
    db_session.add(
        GoldScoreSnapshot(
            timestamp=datetime.now(timezone.utc),
            total_score=12.3,
            direction="中性",
            factor_scores="{}",
            risk_flags="[]",
            summary="测试评分。",
            source="TEST",
        )
    )
    db_session.commit()

    result = get_data_health(db_session)

    assert result["ok"] is True
    assert result["status"] == "warn"
    assert all(item["status"] == "ok" for item in result["items"] if item.get("critical", True))
