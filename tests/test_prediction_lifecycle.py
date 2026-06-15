from datetime import datetime, timedelta, timezone

from app.models import GoldPredictionEvaluation, GoldPredictionSnapshot, GoldPrice, GoldScoreSnapshot, PredictionModelVersion
from app.scoring.gold_predictor import (
    DEFAULT_MODEL_VERSION,
    evaluate_due_predictions,
    ensure_default_prediction_model,
    predict_gold_prices,
    prediction_due_status_summary,
    prediction_evaluation_summary,
    prediction_model_activation_decision,
    rollback_degraded_prediction_model,
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


def test_prediction_due_status_groups_short_horizons(db_session):
    now = datetime.now(timezone.utc)
    for horizon in [1, 7, 30]:
        snap = GoldPredictionSnapshot(
            run_id=f"run_{horizon}",
            timestamp=now - timedelta(days=horizon + 2),
            target_timestamp=now - timedelta(days=1),
            horizon_days=horizon,
            current_price=2400.0,
            predicted_price=2410.0,
            expected_return_pct=0.4,
            confidence_low=2300.0,
            confidence_high=2500.0,
            reliability=0.8,
            method="test",
            model_version=DEFAULT_MODEL_VERSION,
            score_value=1.0,
            score_source="rule_v2",
            input_summary_json="{}",
            note="test",
        )
        db_session.add(snap)
    future = GoldPredictionSnapshot(
        run_id="future",
        timestamp=now,
        target_timestamp=now + timedelta(days=30),
        horizon_days=30,
        current_price=2400.0,
        predicted_price=2500.0,
        expected_return_pct=4.0,
        confidence_low=2300.0,
        confidence_high=2600.0,
        reliability=0.8,
        method="test",
        model_version=DEFAULT_MODEL_VERSION,
        score_value=1.0,
        score_source="rule_v2",
        input_summary_json="{}",
        note="test",
    )
    db_session.add(future)
    db_session.commit()

    status = prediction_due_status_summary(db_session, min_samples=120)
    by_h = {row["horizon_days"]: row for row in status["by_horizon"]}

    assert status["can_evolve"] is False
    assert status["due_pending_count"] == 3
    assert by_h[1]["due_pending_count"] == 1
    assert by_h[30]["future_pending_count"] == 1
    assert status["cannot_evolve_reasons"]


def _candidate_result(**overrides):
    base = {
        "ok": True,
        "optimization_score": 50.0,
        "weighted_mape_price_pct": 5.0,
        "weighted_mae_price": 100.0,
        "weighted_direction_accuracy": 0.60,
        "weighted_recent_direction_accuracy": 0.58,
        "valid_horizons": 3,
        "sample_count": 150,
        "target_horizons": [1, 7, 30],
        "horizon_metrics": {
            "1": {"ok": True, "sample_count": 50},
            "7": {"ok": True, "sample_count": 50},
            "30": {"ok": True, "sample_count": 50},
        },
    }
    base.update(overrides)
    return base


def test_prediction_activation_rejects_insufficient_samples():
    candidate = _candidate_result(sample_count=50)
    baseline = _candidate_result(weighted_direction_accuracy=0.52, weighted_mape_price_pct=5.0)

    decision = prediction_model_activation_decision(candidate, baseline)

    assert decision["eligible"] is False
    assert any("sample_count" in reason for reason in decision["reasons"])


def test_prediction_activation_accepts_guarded_strong_candidate():
    candidate = _candidate_result()
    baseline = _candidate_result(weighted_direction_accuracy=0.55, weighted_mape_price_pct=5.2)

    decision = prediction_model_activation_decision(candidate, baseline)

    assert decision["eligible"] is True
    assert decision["baseline_lift"] == 0.05


def test_prediction_activation_rejects_mape_degradation():
    candidate = _candidate_result(weighted_mape_price_pct=7.0)
    baseline = _candidate_result(weighted_direction_accuracy=0.55, weighted_mape_price_pct=5.0)

    decision = prediction_model_activation_decision(candidate, baseline)

    assert decision["eligible"] is False
    assert any("MAPE" in reason for reason in decision["reasons"])


def test_rollback_degraded_prediction_model_restores_baseline(db_session):
    ensure_default_prediction_model(db_session)
    candidate = PredictionModelVersion(
        version="candidate_active",
        method="multi_signal_ensemble_v2",
        params_json="{}",
        is_active=True,
        direction_accuracy=0.8,
        evaluated_count=20,
    )
    default = db_session.query(PredictionModelVersion).filter_by(version=DEFAULT_MODEL_VERSION).one()
    default.is_active = False
    db_session.add(candidate)
    db_session.commit()
    now = datetime.now(timezone.utc)
    for index in range(6):
        db_session.add(
            GoldPredictionEvaluation(
                prediction_id=1000 + index,
                evaluated_at=now - timedelta(days=index),
                actual_timestamp=now - timedelta(days=index),
                actual_price=2400.0,
                predicted_price=2500.0,
                error_price=100.0,
                abs_error_price=100.0,
                abs_pct_error=4.0,
                predicted_return_pct=2.0,
                actual_return_pct=-2.0,
                direction_hit=False,
                horizon_days=7,
                model_version="candidate_active",
            )
        )
        db_session.add(
            GoldPredictionEvaluation(
                prediction_id=2000 + index,
                evaluated_at=now - timedelta(days=index),
                actual_timestamp=now - timedelta(days=index),
                actual_price=2400.0,
                predicted_price=2410.0,
                error_price=10.0,
                abs_error_price=10.0,
                abs_pct_error=0.4,
                predicted_return_pct=1.0,
                actual_return_pct=1.5,
                direction_hit=True,
                horizon_days=7,
                model_version=DEFAULT_MODEL_VERSION,
            )
        )
    db_session.commit()

    result = rollback_degraded_prediction_model(db_session, min_observations=5)

    assert result["rolled_back"] is True
    assert db_session.query(PredictionModelVersion).filter_by(version=DEFAULT_MODEL_VERSION).one().is_active is True
