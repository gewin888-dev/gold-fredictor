from datetime import datetime, timedelta, timezone

from app.backtesting.score_backtest import run_score_backtest
from app.models import GoldScoreSnapshot
from conftest import insert_gold_prices, insert_required_macro_observations


def insert_score_snapshots(db) -> None:
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    for index, score in enumerate([35, 40, 10, -35, 45]):
        db.add(
            GoldScoreSnapshot(
                timestamp=start + timedelta(days=60 + index * 3),
                total_score=score,
                direction="偏多" if score >= 30 else "偏空" if score <= -30 else "中性",
                factor_scores="{}",
                risk_flags="[]",
                summary="测试评分。",
                source="rule_v2",
            )
        )
    db.commit()


def test_run_score_backtest_returns_summary(db_session):
    insert_required_macro_observations(db_session, days=100)
    insert_gold_prices(db_session, days=100)
    insert_score_snapshots(db_session)

    result = run_score_backtest(db_session, horizon_days=10)

    assert result["ok"] is True
    assert result["horizon_days"] == 10
    assert result["summary"]["sample_count"] > 0
    assert result["summary"]["directional_sample_count"] > 0
    assert result["trades"]


def test_run_score_backtest_requires_data(db_session):
    result = run_score_backtest(db_session)

    assert result["ok"] is False
    assert "Need both score snapshots and gold prices" in result["reason"]


def test_run_score_backtest_rejects_invalid_horizon(db_session):
    try:
        run_score_backtest(db_session, horizon_days=0)
    except ValueError as exc:
        assert "horizon_days" in str(exc)
    else:
        raise AssertionError("Expected invalid horizon error")
