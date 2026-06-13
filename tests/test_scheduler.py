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
