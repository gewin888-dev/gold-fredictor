import logging

from apscheduler.schedulers.background import BackgroundScheduler

logger = logging.getLogger(__name__)

from app.config import get_settings
from app.data.cb_gold_collector import collect_central_bank_gold
from app.data.cftc_collector import collect_cftc_gold_position
from app.data.fred_collector import collect_fred_data
from app.data.sentiment_collector import collect_news_sentiment
from app.data.sge_collector import collect_china_gold_premium
from app.database import SessionLocal, serialized_write
from app.events.calendar import list_macro_events
from app.monitoring.health import get_data_health
from app.monitoring.threshold_alert import send_threshold_alerts
from app.notifications.feishu import send_score_alert_with_health
from app.scoring.gold_score import compute_and_store_gold_score, compute_and_store_gold_score_with_params
from app.scoring.gold_predictor import evaluate_due_predictions, predict_gold_prices
from app.scoring.score_optimizer import activate_version, get_active_params, optimize_score_params, save_best_params


def _collect_sp500_snapshot(db: SessionLocal) -> None:
    """从新浪采集标普 500 指数快照，存入 MacroObservation 表。"""
    import requests
    from datetime import datetime, timezone
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert
    from app.models import MacroObservation, MacroSeries

    try:
        url = "https://hq.sinajs.cn/list=int_sp500"
        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn/"}
        r = requests.get(url, headers=headers, timeout=10)
        r.encoding = "gbk"
        fields = r.text.strip().split('"')[1].split(",")
        price = float(fields[1])

        now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        # 确保 series 存在
        stmt = sqlite_insert(MacroSeries).values(
            series_id="SP500", name="标普 500 指数", frequency="daily", unit="index", source="SINA"
        )
        stmt = stmt.on_conflict_do_update(index_elements=["series_id"], set_={"source": "SINA"})
        db.execute(stmt)
        # 存当日快照
        stmt = sqlite_insert(MacroObservation).values(
            series_id="SP500", timestamp=now, value=price, source="SINA"
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["series_id", "timestamp"],
            set_={"value": price, "source": "SINA"},
        )
        db.execute(stmt)
    except Exception:
        pass


def _collect_silver_snapshot(db: SessionLocal) -> None:
    """从新浪采集白银价格快照。"""
    import requests
    from datetime import datetime, timezone
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert
    from app.models import MacroObservation, MacroSeries
    try:
        url = "https://hq.sinajs.cn/list=hf_SI"
        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn/"}
        r = requests.get(url, headers=headers, timeout=10)
        r.encoding = "gbk"
        fields = r.text.strip().split('"')[1].split(",")
        price = float(fields[3])
        now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        for sid, name in [("SILVER", "白银价格"), ("SILVER_SOURCE", "白银数据源")]:
            stmt = sqlite_insert(MacroSeries).values(series_id=sid, name=name, frequency="daily", unit="USD/oz", source="SINA")
            stmt = stmt.on_conflict_do_update(index_elements=["series_id"], set_={"source": "SINA"})
            db.execute(stmt)
        stmt = sqlite_insert(MacroObservation).values(series_id="SILVER", timestamp=now, value=price, source="SINA")
        stmt = stmt.on_conflict_do_update(index_elements=["series_id", "timestamp"], set_={"value": price, "source": "SINA"})
        db.execute(stmt)
    except Exception:
        pass


def _collect_gld_snapshot(db: SessionLocal) -> None:
    """从新浪采集 GLD ETF 价格快照。"""
    import requests
    from datetime import datetime, timezone
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert
    from app.models import MacroObservation, MacroSeries
    try:
        url = "https://hq.sinajs.cn/list=gb_gld"
        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn/"}
        r = requests.get(url, headers=headers, timeout=10)
        r.encoding = "gbk"
        fields = r.text.strip().split('"')[1].split(",")
        price = float(fields[1])
        now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        stmt = sqlite_insert(MacroSeries).values(series_id="GLD_ETF", name="SPDR Gold Trust ETF", frequency="daily", unit="USD", source="SINA")
        stmt = stmt.on_conflict_do_update(index_elements=["series_id"], set_={"source": "SINA"})
        db.execute(stmt)
        stmt = sqlite_insert(MacroObservation).values(series_id="GLD_ETF", timestamp=now, value=price, source="SINA")
        stmt = stmt.on_conflict_do_update(index_elements=["series_id", "timestamp"], set_={"value": price, "source": "SINA"})
        db.execute(stmt)
    except Exception:
        pass


def _collect_google_trend_snapshot(db: SessionLocal) -> None:
    """从 Google Trends 采集 gold price 搜索热度。"""
    from datetime import datetime, timezone
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert
    from app.models import MacroObservation, MacroSeries
    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl='en-US', tz=360, timeout=15)
        pytrends.build_payload(['gold price'], timeframe='today 3-m', geo='')
        df = pytrends.interest_over_time()
        if df.empty:
            return
        stmt = sqlite_insert(MacroSeries).values(
            series_id="GOOGLE_TREND", name="Google Trends 搜索热度", frequency="daily", unit="index", source="GOOGLE_TRENDS"
        )
        stmt = stmt.on_conflict_do_update(index_elements=["series_id"], set_={"source": "GOOGLE_TRENDS"})
        db.execute(stmt)
        for idx, row in df.iterrows():
            day = idx.to_pydatetime().replace(tzinfo=timezone.utc)
            val = float(row['gold price'])
            stmt = sqlite_insert(MacroObservation).values(
                series_id="GOOGLE_TREND", timestamp=day, value=val, source="GOOGLE_TRENDS"
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=["series_id", "timestamp"],
                set_={"value": val, "source": "GOOGLE_TRENDS"},
            )
            db.execute(stmt)
    except Exception:
        pass


def collect_and_score_job() -> None:
    """定时任务：采集所有数据 → 评分 → 阈值告警 → 每日报告推送。"""
    db = SessionLocal()
    try:
        with serialized_write():
            # 采集全部数据源（每个独立 try，单点失败不影响其他）
            for name, collector in [
                ("FRED", collect_fred_data),
                ("CFTC", collect_cftc_gold_position),
                ("中国溢价", collect_china_gold_premium),
                ("央行购金", collect_central_bank_gold),
                ("新闻情绪", collect_news_sentiment),
                ("SP500", _collect_sp500_snapshot),
                ("白银", _collect_silver_snapshot),
                ("GLD_ETF", _collect_gld_snapshot),
                ("搜索热度", _collect_google_trend_snapshot),
                ("GDX", _collect_gdx_snapshot),
                ("WTI", _collect_wti_snapshot),
                ("铜", _collect_copper_snapshot),
            ]:
                try:
                    collector(db)
                except Exception:
                    logger.warning("采集器 %s 失败（已跳过，不中断整体流程）", name, exc_info=True)

            # 优先使用激活的优化参数
            active_params = get_active_params(db)
            if active_params is not None:
                snapshot = compute_and_store_gold_score_with_params(db, active_params, source="optimized_active")
            else:
                snapshot = compute_and_store_gold_score(db)

            # 记录一次预测快照，并评估已到期的历史预测。
            predict_gold_prices(db, persist=True)
            evaluate_due_predictions(db)

        # 阈值告警（因子突变即时推送）
        send_threshold_alerts(db, snapshot)

        # 每小时推送精简版，每日完整版（UTC 22:00 是北京时间 06:00）
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        is_daily_report = now.hour == 22

        if is_daily_report:
            events = [
                {
                    "timestamp": row.timestamp,
                    "name": row.name,
                    "importance": row.importance,
                }
                for row in list_macro_events(db, days_ahead=30)
            ]
            send_score_alert_with_health(snapshot, get_data_health(db), events)
    finally:
        db.close()


def auto_optimize_job() -> None:
    """受控自我优化：保存候选参数；默认不自动激活。"""
    settings = get_settings()
    if not settings.auto_optimize_score_params:
        return

    db = SessionLocal()
    try:
        with serialized_write():
            results = optimize_score_params(
                db,
                n_iter=settings.auto_optimize_n_iter,
                horizon_days=settings.auto_optimize_horizon_days,
            )
            if not results or not results[0].get("ok"):
                return

            best = results[0]
            from datetime import datetime, timezone

            version = f"auto_opt_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}"
            saved = save_best_params(
                db,
                best,
                version=version,
                horizon_days=settings.auto_optimize_horizon_days,
                notes="定时任务生成的候选参数；默认仅保存，达到阈值且显式授权后才自动激活。",
            )
            hit_rate = best.get("hit_rate")
            if (
                settings.auto_activate_optimized_params
                and hit_rate is not None
                and float(hit_rate) >= settings.auto_optimize_min_hit_rate
                and saved.sample_count
                and saved.sample_count >= 80
            ):
                activate_version(db, version)
    finally:
        db.close()


def create_scheduler() -> BackgroundScheduler:
    scheduler = BackgroundScheduler(timezone="UTC")

    # 每小时整点过5分执行（FRED/CFTC 日频数据会缓存，不会重复拉）
    scheduler.add_job(
        collect_and_score_job,
        "cron",
        hour="*",
        minute=5,
        id="hourly_collect_and_score",
    )

    # 央行购金：每 15 天执行一次（UTC 周一 08:00）
    scheduler.add_job(
        collect_and_score_job,
        "cron",
        day="*/15",
        hour=8,
        minute=0,
        id="cb_gold_15day",
    )

    # 每日完整报告（北京时间早 6 点）
    scheduler.add_job(
        collect_and_score_job,
        "cron",
        hour=22,
        minute=0,
        id="daily_full_report",
    )

    # 每周自我优化候选参数；默认由 AUTO_OPTIMIZE_SCORE_PARAMS 控制，不会擅自激活。
    scheduler.add_job(
        auto_optimize_job,
        "cron",
        day_of_week="sun",
        hour=21,
        minute=30,
        id="weekly_auto_optimize",
    )

    return scheduler


def _collect_gdx_snapshot(db: SessionLocal) -> None:
    import requests
    from datetime import datetime, timezone
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert
    from app.models import MacroObservation, MacroSeries
    try:
        url = "https://hq.sinajs.cn/list=gb_gdx"
        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn/"}
        r = requests.get(url, headers=headers, timeout=10)
        r.encoding = "gbk"
        price = float(r.text.strip().split('"')[1].split(",")[1])
        now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        stmt = sqlite_insert(MacroSeries).values(series_id="GDX", name="黄金矿业股ETF", frequency="daily", unit="USD", source="SINA")
        stmt = stmt.on_conflict_do_update(index_elements=["series_id"], set_={"source": "SINA"})
        db.execute(stmt)
        stmt = sqlite_insert(MacroObservation).values(series_id="GDX", timestamp=now, value=price, source="SINA")
        stmt = stmt.on_conflict_do_update(index_elements=["series_id", "timestamp"], set_={"value": price, "source": "SINA"})
        db.execute(stmt)
    except Exception:
        pass


def _collect_wti_snapshot(db: SessionLocal) -> None:
    import requests
    from datetime import datetime, timezone
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert
    from app.models import MacroObservation, MacroSeries
    try:
        url = "https://hq.sinajs.cn/list=hf_CL"
        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn/"}
        r = requests.get(url, headers=headers, timeout=10)
        r.encoding = "gbk"
        price = float(r.text.strip().split('"')[1].split(",")[3])
        now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        stmt = sqlite_insert(MacroSeries).values(series_id="WTI", name="WTI原油", frequency="daily", unit="USD/bbl", source="SINA")
        stmt = stmt.on_conflict_do_update(index_elements=["series_id"], set_={"source": "SINA"})
        db.execute(stmt)
        stmt = sqlite_insert(MacroObservation).values(series_id="WTI", timestamp=now, value=price, source="SINA")
        stmt = stmt.on_conflict_do_update(index_elements=["series_id", "timestamp"], set_={"value": price, "source": "SINA"})
        db.execute(stmt)
    except Exception:
        pass


def _collect_copper_snapshot(db: SessionLocal) -> None:
    import requests
    from datetime import datetime, timezone
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert
    from app.models import MacroObservation, MacroSeries
    try:
        url = "https://hq.sinajs.cn/list=hf_HG"
        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn/"}
        r = requests.get(url, headers=headers, timeout=10)
        r.encoding = "gbk"
        price = float(r.text.strip().split('"')[1].split(",")[3])
        now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        stmt = sqlite_insert(MacroSeries).values(series_id="COPPER", name="铜价", frequency="daily", unit="USD/lb", source="SINA")
        stmt = stmt.on_conflict_do_update(index_elements=["series_id"], set_={"source": "SINA"})
        db.execute(stmt)
        stmt = sqlite_insert(MacroObservation).values(series_id="COPPER", timestamp=now, value=price, source="SINA")
        stmt = stmt.on_conflict_do_update(index_elements=["series_id", "timestamp"], set_={"value": price, "source": "SINA"})
        db.execute(stmt)
    except Exception:
        pass
