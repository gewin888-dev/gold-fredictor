from datetime import datetime, timedelta, timezone

from app.models import GoldPredictionEvaluation, GoldPredictionSnapshot, GoldPrice, GoldScoreSnapshot
from app.scoring.gold_predictor import (
    evaluate_due_predictions,
    predict_gold_prices,
    prediction_evaluation_summary,
)
from conftest import insert_gold_prices


def insert_score_history(db, days: int = 100) -> None:
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    for index in range(days):
        score = -10 + (index % 20)
        db.add(
            GoldScoreSnapshot(
                timestamp=start + timedelta(days=index),
                total_score=float(score),
                direction="中性",
                factor_scores="{}",
                risk_flags="[]",
                summary="测试评分。",
                source="rule_v2",
            )
        )
    db.commit()


def test_prediction_persist_creates_snapshots(db_session):
    insert_gold_prices(db_session, days=120)
    insert_score_history(db_session, days=120)

    result = predict_gold_prices(db_session, persist=True)

    assert result["ok"] is True
    assert result["persisted_run_id"]
    rows = db_session.query(GoldPredictionSnapshot).all()
    assert len(rows) == 6
    assert {row.horizon_days for row in rows} == {1, 7, 30, 90, 180, 360}


def test_due_prediction_evaluation_records_error(db_session):
    now = datetime(2025, 6, 1, tzinfo=timezone.utc)
    db_session.add(
        GoldPredictionSnapshot(
            run_id="test_run",
            timestamp=now - timedelta(days=10),
            target_timestamp=now - timedelta(days=1),
            horizon_days=7,
            current_price=2400.0,
            predicted_price=2500.0,
            expected_return_pct=4.0,
            confidence_low=2350.0,
            confidence_high=2550.0,
            reliability=0.8,
            method="test",
            model_version="predictor_v2_baseline",
            score_value=10.0,
            score_source="rule_v2",
            input_summary_json="{}",
            note="test",
        )
    )
    db_session.add(
        GoldPrice(
            date=now,
            open=2440.0,
            high=2460.0,
            low=2430.0,
            close=2450.0,
            source="TEST",
        )
    )
    db_session.commit()

    result = evaluate_due_predictions(db_session)
    summary = prediction_evaluation_summary(db_session)

    assert result["evaluated"] == 1
    row = db_session.query(GoldPredictionEvaluation).one()
    assert row.abs_error_price == 50.0
    assert row.direction_hit is True
    assert summary["summary"]["evaluated_count"] == 1
    assert summary["by_horizon"][0]["horizon_days"] == 7
