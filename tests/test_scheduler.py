from app.scheduler import create_scheduler


def test_create_scheduler_registers_daily_job():
    scheduler = create_scheduler()
    jobs = scheduler.get_jobs()

    assert {job.id for job in jobs} == {
        "hourly_collect_and_score",
        "daily_full_report",
        "cb_gold_15day",
        "weekly_auto_optimize",
    }


def test_settings_include_prediction_auto_optimize_fields():
    from app.config import Settings

    settings = Settings()

    assert settings.auto_evolution_full_auto is True
    assert settings.auto_optimize_prediction_model is True
    assert settings.auto_activate_prediction_model is True
    assert settings.auto_prediction_n_iter >= 80
    assert settings.auto_prediction_min_samples >= 120
    assert settings.auto_prediction_min_score >= 40
    assert settings.auto_prediction_max_mape_pct <= 8
    assert settings.auto_prediction_min_direction_accuracy >= 0.52


def test_auto_optimize_job_skips_when_switches_off(db_session, monkeypatch):
    from app import scheduler

    class _SessionFactory:
        def __call__(self):
            return db_session

    monkeypatch.setattr(scheduler, "SessionLocal", _SessionFactory())
    monkeypatch.setattr(
        scheduler,
        "resolved_auto_settings",
        lambda db: {
            "AUTO_EVOLUTION_FULL_AUTO": False,
            "AUTO_OPTIMIZE_SCORE_PARAMS": False,
            "AUTO_OPTIMIZE_PREDICTION_MODEL": False,
            "AUTO_ACTIVATE_OPTIMIZED_PARAMS": False,
            "AUTO_ACTIVATE_PREDICTION_MODEL": False,
            "AUTO_OPTIMIZE_N_ITER": 1,
            "AUTO_OPTIMIZE_HORIZON_DAYS": 20,
            "AUTO_OPTIMIZE_MIN_HIT_RATE": 0.55,
            "AUTO_PREDICTION_N_ITER": 1,
            "AUTO_PREDICTION_MIN_SCORE": 40,
            "AUTO_PREDICTION_MAX_MAPE_PCT": 8,
            "AUTO_PREDICTION_MIN_DIRECTION_ACCURACY": 0.52,
            "AUTO_PREDICTION_MIN_SAMPLES": 120,
            "AUTO_PREDICTION_MIN_VALID_HORIZONS": 3,
        },
    )
    monkeypatch.setattr(
        scheduler,
        "optimize_score_params",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not optimize score")),
    )
    monkeypatch.setattr(
        scheduler,
        "optimize_prediction_model_params",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not optimize prediction")),
    )

    scheduler.auto_optimize_job()
