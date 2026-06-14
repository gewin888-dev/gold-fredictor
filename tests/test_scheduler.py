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

    assert settings.auto_optimize_prediction_model is False
    assert settings.auto_activate_prediction_model is False
    assert settings.auto_prediction_n_iter > 0
    assert settings.auto_prediction_min_samples >= 1


def test_auto_optimize_job_skips_when_switches_off(db_session, monkeypatch):
    from app import scheduler

    class _SessionFactory:
        def __call__(self):
            return db_session

    monkeypatch.setattr(scheduler, "SessionLocal", _SessionFactory())
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
