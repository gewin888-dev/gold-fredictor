import logging

from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

from sqlalchemy import select

from app.config import get_settings
from app.auto_settings import resolved_auto_settings
from app.monitoring.collector_health import record_success, record_failure
from app.data.cb_gold_collector import collect_central_bank_gold
from app.data.cftc_collector import collect_cftc_gold_position
from app.data.fred_collector import collect_fred_data
from app.data.gold_price_collector import fetch_gold_history
from app.data.market_structure_collector import collect_market_structure_data
from app.data.sentiment_collector import collect_news_sentiment
from app.data.sge_collector import collect_china_gold_premium
from app.database import SessionLocal, serialized_write
from app.events.calendar import list_macro_events
from app.monitoring.health import get_data_health
from app.monitoring.threshold_alert import send_threshold_alerts
from app.notifications.feishu import send_score_alert_with_health
from app.scoring.gold_score import compute_and_store_gold_score, compute_and_store_gold_score_with_params
from app.scoring.gold_predictor import (
    evaluate_due_predictions,
    optimize_prediction_model_params,
    predict_gold_prices,
    rollback_degraded_prediction_model,
)
from app.scoring.score_optimizer import activate_version, get_active_params, optimize_score_params, save_best_params


def _collect_sp500_snapshot(db: Session) -> None:
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
        logger.warning("采集 SP500 快照失败", exc_info=True)


def _collect_silver_snapshot(db: Session) -> None:
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
        logger.warning("采集白银快照失败", exc_info=True)


def _collect_gld_snapshot(db: Session) -> None:
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
        logger.warning("采集 GLD ETF 快照失败", exc_info=True)


def _collect_google_trend_snapshot(db: Session) -> None:
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
        logger.warning("采集 Google Trends 快照失败", exc_info=True)


def collect_and_score_job() -> None:
    """定时任务：采集所有数据 → 评分 → 阈值告警 → 每日报告推送。"""
    # 先刷新金价日线（独立 DB 会话，失败不阻塞后续采集）
    try:
        fetch_gold_history(days=1)
        record_success("gold_price", "hourly refresh")
    except Exception:
        logger.warning("金价日线刷新失败（已跳过，不中断整体流程）", exc_info=True)
        record_failure("gold_price", "fetch_gold_history failed")

    db = SessionLocal()
    try:
        # 采集全部数据源（每个独立 try，单点失败不影响其他）
        # (显示名, 健康名, 采集函数)
        _collectors: list[tuple[str, str, object]] = [
            ("FRED", "fred_data", collect_fred_data),
            ("CFTC", "cftc_position", collect_cftc_gold_position),
            ("中国溢价", "china_premium", collect_china_gold_premium),
            ("央行购金", "central_bank_gold", collect_central_bank_gold),
            ("新闻情绪", "news_sentiment", collect_news_sentiment),
            ("市场结构", "market_structure", collect_market_structure_data),
            ("SP500", "sp500_snapshot", _collect_sp500_snapshot),
            ("白银", "silver_snapshot", _collect_silver_snapshot),
            ("GLD_ETF", "gld_etf", _collect_gld_snapshot),
            ("搜索热度", "google_trend", _collect_google_trend_snapshot),
            ("GDX", "gdx_snapshot", _collect_gdx_snapshot),
            ("WTI", "wti_snapshot", _collect_wti_snapshot),
            ("铜", "copper_snapshot", _collect_copper_snapshot),
        ]
        for label, health_name, collector in _collectors:
            try:
                collector(db)
                record_success(health_name, "ok")
            except Exception:
                logger.warning("采集器 %s 失败（已跳过，不中断整体流程）", label, exc_info=True)
                record_failure(health_name, f"collector {label} failed")

        # 所有采集器完成后统一提交，写入操作受串行锁保护
        with serialized_write():
            db.commit()

        # 优先使用激活的优化参数
        active_params = get_active_params(db)
        if active_params is not None:
            snapshot = compute_and_store_gold_score_with_params(db, active_params, source="optimized_active")
        else:
            snapshot = compute_and_store_gold_score(db)

        # 记录一次预测快照，并评估已到期的历史预测。
        predict_gold_prices(db, persist=True)
        evaluate_due_predictions(db)
        rollback_degraded_prediction_model(db)

        # 安全网：如果激活模型方向准确率接近0且存在更好的候选，自动切换
        try:
            from app.models import PredictionModelVersion
            from app.scoring.gold_predictor import _active_prediction_model, activate_prediction_model_version
            active_model = _active_prediction_model(db)
            if active_model is not None and (active_model.direction_accuracy or 0.0) < 0.05 \
                    and (active_model.evaluated_count or 0) >= 5:
                # 查找更好的候选
                candidates = db.scalars(
                    select(PredictionModelVersion)
                    .where(
                        PredictionModelVersion.is_active == False,  # noqa: E712
                        PredictionModelVersion.direction_accuracy > 0.2,
                        PredictionModelVersion.evaluated_count >= 10,
                    )
                    .order_by(PredictionModelVersion.direction_accuracy.desc())
                ).all()
                if candidates:
                    best = candidates[0]
                    logger.warning(
                        "安全网触发：激活模型 %s 方向准确率 %.1f%%，自动切换到 %s (%.1f%%)",
                        active_model.version,
                        (active_model.direction_accuracy or 0) * 100,
                        best.version,
                        (best.direction_accuracy or 0) * 100,
                    )
                    activate_prediction_model_version(db, best.version)
                    record_failure("prediction_engine",
                                   f"安全网切换: {active_model.version}({active_model.direction_accuracy}) -> {best.version}({best.direction_accuracy})")
            else:
                record_success("prediction_engine", f"model={active_model.version}" if active_model else "no_model")
        except Exception:
            logger.debug("预测模型安全网检查跳过", exc_info=True)

        # 自动进化桥接：预测不准 → 搜索更优参数
        try:
            from app.auto_evolve import auto_evolve_if_needed
            auto_evolve_if_needed(db)
        except Exception:
            logger.debug("自动进化检查跳过", exc_info=True)

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
            # 采集器健康
            collector_status = ""
            try:
                from app.monitoring.collector_health import get_health_summary
                h = get_health_summary()
                issues = h.get("summary", {}).get("critical_issues", [])
                if issues:
                    collector_status = f"⚠️ 异常采集器：{', '.join(issues)}"
                else:
                    collector_status = f"✅ 采集器正常（{h.get('summary',{}).get('healthy',0)}/{h.get('summary',{}).get('total',0)}）"
            except Exception:
                collector_status = "采集器状态：检查失败"
            send_score_alert_with_health(snapshot, get_data_health(db), events, collector_status)
    finally:
        db.close()


def auto_optimize_job() -> None:
    """受控自我优化：保存候选评分/预测参数；默认不自动激活。"""
    db = SessionLocal()
    try:
        settings = resolved_auto_settings(db)
        if not settings["AUTO_OPTIMIZE_SCORE_PARAMS"] and not settings["AUTO_OPTIMIZE_PREDICTION_MODEL"]:
            return

        if settings["AUTO_OPTIMIZE_SCORE_PARAMS"]:
            # 参数搜索（纯计算，不持锁）
            results = optimize_score_params(
                db,
                n_iter=int(settings["AUTO_OPTIMIZE_N_ITER"]),
                horizon_days=int(settings["AUTO_OPTIMIZE_HORIZON_DAYS"]),
            )
            if not results or not results[0].get("ok"):
                results = []

            if results:
                best = results[0]
                from datetime import datetime, timezone

                version = f"auto_opt_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}"
                hit_rate = best.get("hit_rate")
                activation_check = best.get("activation_check") or {}
                # 仅写库操作持锁
                with serialized_write():
                    saved = save_best_params(
                        db,
                        best,
                        version=version,
                        horizon_days=int(settings["AUTO_OPTIMIZE_HORIZON_DAYS"]),
                        notes="定时任务生成的候选评分参数；默认仅保存，达到阈值且显式授权后才自动激活。",
                    )
                    if (
                        settings["AUTO_ACTIVATE_OPTIMIZED_PARAMS"]
                        and hit_rate is not None
                        and float(hit_rate) >= float(settings["AUTO_OPTIMIZE_MIN_HIT_RATE"])
                        and saved.sample_count
                        and saved.sample_count >= 120
                        and activation_check.get("eligible")
                    ):
                        activate_version(db, version)

        if settings["AUTO_OPTIMIZE_PREDICTION_MODEL"]:
            # 预测模型搜索同样先跑计算，再持锁写库
            optimize_prediction_model_params(
                db,
                n_iter=int(settings["AUTO_PREDICTION_N_ITER"]),
                top_k=5,
                random_seed=42,
                save_best=True,
                auto_activate=bool(settings["AUTO_ACTIVATE_PREDICTION_MODEL"]),
                activation_thresholds={
                    "min_score": settings["AUTO_PREDICTION_MIN_SCORE"],
                    "max_mape_price_pct": settings["AUTO_PREDICTION_MAX_MAPE_PCT"],
                    "min_direction_accuracy": settings["AUTO_PREDICTION_MIN_DIRECTION_ACCURACY"],
                    "min_samples": settings["AUTO_PREDICTION_MIN_SAMPLES"],
                    "min_valid_horizons": 3,
                    "min_baseline_lift": 0.03,
                    "max_mape_worse_ratio": 1.2,
                    "max_recent_degradation": 0.05,
                },
            )
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
        max_instances=1,
        coalesce=True,
        misfire_grace_time=300,
    )

    # 央行购金：每 15 天执行一次（UTC 周一 08:00）
    scheduler.add_job(
        collect_and_score_job,
        "cron",
        day="*/15",
        hour=8,
        minute=0,
        id="cb_gold_15day",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=600,
    )

    # 每日完整报告（北京时间早 6 点）
    scheduler.add_job(
        collect_and_score_job,
        "cron",
        hour=22,
        minute=0,
        id="daily_full_report",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=600,
    )

    # 每周自我优化候选参数；默认由 AUTO_OPTIMIZE_SCORE_PARAMS 控制，不会擅自激活。
    scheduler.add_job(
        auto_optimize_job,
        "cron",
        day_of_week="sun",
        hour=21,
        minute=30,
        id="weekly_auto_optimize",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=1800,
    )

    return scheduler


def _collect_gdx_snapshot(db: Session) -> None:
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
        logger.warning("采集 GDX 快照失败", exc_info=True)


def _collect_wti_snapshot(db: Session) -> None:
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
        logger.warning("采集 WTI 快照失败", exc_info=True)


def _collect_copper_snapshot(db: Session) -> None:
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
        logger.warning("采集铜价快照失败", exc_info=True)
