from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.backtesting.score_backtest import run_score_backtest
from app.data.cb_gold_collector import collect_central_bank_gold, load_sample_cb_gold
from app.data.cftc_collector import collect_cftc_gold_position
from app.data.fred_collector import collect_fred_data
from app.data.gold_price_collector import (
    fetch_gold_history,
    fetch_gold_intraday,
    fetch_gold_price,
    start_intraday_recorder,
    stop_intraday_recorder,
)
from app.data.sentiment_collector import collect_news_sentiment
from app.data.sge_collector import collect_china_gold_premium
from app.database import get_db, init_db, serialized_write
from app.config import get_settings
from app.events.calendar import list_macro_events
from app.models import (
    CentralBankGold,
    CftcPosition,
    ChinaGoldPremium,
    GoldScoreSnapshot,
    MacroObservation,
    NewsSentiment,
    PredictionModelVersion,
    ScoreParamsVersion,
)
from app.monitoring.health import get_data_health
from app.monitoring.threshold_alert import send_threshold_alerts
from app.notifications.feishu import send_score_alert_with_health, send_text_message
from app.scoring.gold_score import (
    compute_and_store_gold_score,
    compute_and_store_gold_score_with_params,
)
from app.scoring.gold_predictor import (
    ensure_default_prediction_model,
    evaluate_due_predictions,
    predict_gold_prices,
    prediction_evaluation_summary,
    refresh_prediction_model_metrics,
)
from app.scoring.score_optimizer import (
    ScoreParams,
    activate_version,
    optimize_score_params,
    save_best_params,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    settings = get_settings()
    scheduler = None

    if settings.auto_start_scheduler:
        from app.scheduler import create_scheduler

        scheduler = create_scheduler()
        scheduler.start()

    if settings.auto_bootstrap_data:
        from app.database import SessionLocal

        db = SessionLocal()
        try:
            fetch_gold_history(days=200)
            if settings.fred_api_key:
                collect_fred_data(db)
            db.commit()
        except Exception:
            db.rollback()
        finally:
            db.close()

    # 启动日内金价快照记录（用于 24 小时走势图）
    start_intraday_recorder()

    try:
        yield
    finally:
        stop_intraday_recorder()
        if scheduler is not None:
            scheduler.shutdown(wait=False)


app = FastAPI(title="黄金走势实时监控与预测系统", lifespan=lifespan)

# ── WebSocket 实时推送管理 ────────────────────────────────────────


class ScoreBroadcaster:
    """管理所有 WebSocket 连接，评分更新时广播。"""

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._connections:
            self._connections.remove(ws)

    async def broadcast(self, data: dict) -> None:
        dead: list[WebSocket] = []
        for ws in self._connections:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._connections.remove(ws)


broadcaster = ScoreBroadcaster()


def auto_score_and_broadcast(db: Session, broadcast: bool = False) -> GoldScoreSnapshot:
    """采集数据后自动评分、阈值告警。

    当 broadcast=True 时（仅在异步上下文中），同时通过 WebSocket 广播。
    """
    from app.scoring.score_optimizer import get_active_params

    active_params = get_active_params(db)
    if active_params is not None:
        snapshot = compute_and_store_gold_score_with_params(db, active_params, source="auto_trigger")
    else:
        snapshot = compute_and_store_gold_score(db)

    # 阈值告警
    send_threshold_alerts(db, snapshot)

    # WebSocket 广播（仅异步上下文可用）
    if broadcast:
        data = serialize_score(snapshot)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(broadcaster.broadcast(data))
        except RuntimeError:
            pass

    return snapshot


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/gold/price")
def gold_price() -> dict[str, object]:
    """获取实时金价（Yahoo Finance，约 15 分钟延迟）。"""
    return fetch_gold_price()


@app.get("/gold/intraday")
def gold_intraday(interval_minutes: int = 5) -> dict[str, object]:
    """获取日内金价走势（1分钟原始数据聚合为指定分钟OHLC）。"""
    return fetch_gold_intraday(interval_minutes=interval_minutes)


@app.get("/predict/gold")
def predict_gold(persist: bool = False, db: Session = Depends(get_db)) -> dict[str, object]:
    """预测未来 1/7/30/90/180/360 天金价。

    persist=false 时只计算展示；persist=true 时会保存本次预测快照。
    """
    if persist:
        with serialized_write():
            return predict_gold_prices(db, persist=True)
    return predict_gold_prices(db)


@app.post("/predict/gold/snapshot")
def create_prediction_snapshot(db: Session = Depends(get_db)) -> dict[str, object]:
    """保存一次当前预测快照，用于未来到期验证。"""
    with serialized_write():
        return predict_gold_prices(db, persist=True)


@app.post("/predict/evaluate")
def evaluate_predictions(limit: int = 500, db: Session = Depends(get_db)) -> dict[str, object]:
    """评估已经到期的历史预测。"""
    with serialized_write():
        return evaluate_due_predictions(db, limit=limit)


@app.get("/predict/evaluation")
def prediction_evaluation(db: Session = Depends(get_db)) -> dict[str, object]:
    """查看预测历史误差和方向命中率汇总。"""
    return prediction_evaluation_summary(db)


@app.get("/predict/models")
def prediction_models(db: Session = Depends(get_db)) -> dict[str, object]:
    """列出预测模型版本及其历史评估指标。"""
    ensure_default_prediction_model(db)
    refresh_prediction_model_metrics(db)
    rows = db.scalars(
        select(PredictionModelVersion).order_by(PredictionModelVersion.created_at.desc())
    ).all()
    return {
        "ok": True,
        "data": [
            {
                "version": row.version,
                "method": row.method,
                "is_active": bool(row.is_active),
                "mae_price": row.mae_price,
                "mape_price_pct": row.mape_price_pct,
                "direction_accuracy": row.direction_accuracy,
                "evaluated_count": row.evaluated_count,
                "created_at": row.created_at,
                "updated_at": row.updated_at,
                "notes": row.notes,
            }
            for row in rows
        ],
    }


@app.post("/collect/gold_history")
def collect_gold_history(days: int = 200, db: Session = Depends(get_db)) -> dict[str, object]:
    """下载历史金价日线（Yahoo Finance），存入 gold_prices 表。"""
    with serialized_write():
        results = fetch_gold_history(days=days)
        result = results[0] if results else {"ok": False, "error": "no data"}
        if result.get("ok"):
            db.commit()
    return result


@app.websocket("/ws/score")
async def ws_score(websocket: WebSocket) -> None:
    """WebSocket 实时评分推送。

    连接后，每次评分更新自动推送最新分数、方向和因子。
    """
    await broadcaster.connect(websocket)
    try:
        # 连接后立即推送当前最新评分
        db = next(get_db())
        try:
            latest = db.scalar(select(GoldScoreSnapshot).order_by(GoldScoreSnapshot.timestamp.desc()))
            if latest:
                await websocket.send_json(serialize_score(latest))
        finally:
            db.close()
        # 保持连接，等待广播
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        broadcaster.disconnect(websocket)


@app.get("/health/data")
def data_health(db: Session = Depends(get_db)) -> dict[str, object]:
    return get_data_health(db)


@app.post("/collect/fred")
def collect_fred(db: Session = Depends(get_db)) -> dict[str, object]:
    with serialized_write():
        counts = collect_fred_data(db)
        db.commit()
        snapshot = auto_score_and_broadcast(db)
    return {"ok": True, "counts": counts, "score_updated": serialize_score(snapshot)}


@app.post("/collect/cftc")
def collect_cftc(db: Session = Depends(get_db)) -> dict[str, object]:
    with serialized_write():
        record = collect_cftc_gold_position(db)
        snapshot = auto_score_and_broadcast(db)
    return {"ok": True, "data": serialize_cftc_position(record), "score_updated": serialize_score(snapshot)}


# ── 新增数据源采集 API ──────────────────────────────────────────


@app.post("/collect/china_premium")
def collect_china_premium(db: Session = Depends(get_db)) -> dict[str, object]:
    """采集中国黄金溢价数据。"""
    with serialized_write():
        record = collect_china_gold_premium(db)
    if record is None:
        return {"ok": False, "reason": "No data available."}
    return {
        "ok": True,
        "data": {
            "timestamp": record.timestamp.isoformat() if record.timestamp else None,
            "premium_pct": record.premium_pct,
            "sge_price_cny": record.sge_price_cny,
            "source": record.source,
        },
    }


@app.post("/collect/cb_gold")
def collect_cb_gold(db: Session = Depends(get_db)) -> dict[str, object]:
    """采集央行黄金储备数据。"""
    with serialized_write():
        count = collect_central_bank_gold(db)
    return {"ok": True, "records_collected": count}


@app.post("/collect/sentiment")
def collect_sentiment(
    days_back: int = 3, max_records: int = 50, db: Session = Depends(get_db)
) -> dict[str, object]:
    """采集黄金相关新闻情绪。"""
    with serialized_write():
        count = collect_news_sentiment(db, days_back=days_back, max_records=max_records)
    return {"ok": True, "records_collected": count}


@app.get("/china_premium/latest")
def china_premium_latest(db: Session = Depends(get_db)) -> dict[str, object]:
    """查询最新中国黄金溢价。"""
    row = db.scalar(select(ChinaGoldPremium).order_by(ChinaGoldPremium.timestamp.desc()))
    if not row:
        return {"ok": False, "reason": "No China premium data found."}
    return {
        "ok": True,
        "data": {
            "timestamp": row.timestamp.isoformat() if row.timestamp else None,
            "premium_pct": row.premium_pct,
            "sge_price_cny": row.sge_price_cny,
            "lbma_price_usd": row.lbma_price_usd,
            "usdcny": row.usdcny,
            "source": row.source,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        },
    }


@app.get("/cb_gold/latest")
def cb_gold_latest(db: Session = Depends(get_db)) -> dict[str, object]:
    """查询最新央行购金数据（全球汇总）。"""
    rows = db.scalars(
        select(CentralBankGold)
        .where(CentralBankGold.country == "Global")
        .order_by(CentralBankGold.timestamp.desc())
        .limit(4)
    ).all()
    settings = get_settings()
    if (
        settings.production_mode
        and rows
        and all(str(r.source).upper() in {"SAMPLE", "ESTIMATE", "MANUAL", "JSON"} for r in rows)
        and not settings.show_low_confidence_data
    ):
        return {
            "ok": True,
            "status": "pending_source",
            "reason": "央行购金当前只有占位或不可评分来源，生产模式下已隐藏。",
            "data": [],
        }
    return {
        "ok": True,
        "data": [
            {
                "period": r.period,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                "net_change_tonnes": r.net_change_tonnes,
                "reserves_tonnes": r.reserves_tonnes,
                "source": r.source,
            }
            for r in rows
        ],
    }


@app.get("/sentiment/latest")
def sentiment_latest(db: Session = Depends(get_db)) -> dict[str, object]:
    """查询最近新闻情绪。"""
    from app.data.sentiment_collector import get_recent_sentiment

    avg = get_recent_sentiment(db)
    recent = db.scalars(
        select(NewsSentiment)
        .where(NewsSentiment.source.in_(["NEWSAPI", "GDELT"]))
        .order_by(NewsSentiment.timestamp.desc())
        .limit(10)
    ).all()
    return {
        "ok": True,
        "average_sentiment_7d": avg,
        "recent": [
            {
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                "title": r.title,
                "sentiment_score": r.sentiment_score,
                "source_url": r.source_url,
                "source": r.source,
            }
            for r in recent
        ],
    }


@app.post("/admin/load_sample_data")
def load_all_sample_data(db: Session = Depends(get_db)) -> dict[str, object]:
    """生产模式已禁用样例数据加载。所有数据源已接入真实数据。"""
    return {"ok": False, "reason": "生产模式已禁用样例数据。所有因子均使用真实数据源。"}


@app.post("/score/compute")
def compute_score(db: Session = Depends(get_db)) -> dict[str, object]:
    with serialized_write():
        snapshot = auto_score_and_broadcast(db)
    return serialize_score(snapshot)


@app.post("/score/compute/ws")
async def compute_score_ws() -> dict[str, object]:
    """评分并 WebSocket 广播（异步端点，供前端实时推送用）。"""
    from app.database import SessionLocal

    db = SessionLocal()
    try:
        with serialized_write():
            snapshot = auto_score_and_broadcast(db, broadcast=True)
        return serialize_score(snapshot)
    finally:
        db.close()


@app.get("/backtest/score")
def score_backtest(
    horizon_days: int = 20,
    include_trades: bool = False,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return run_score_backtest(
        db,
        horizon_days=horizon_days,
        include_trades=include_trades,
        limit=limit,
        offset=offset,
    )


@app.get("/macro/latest")
def macro_latest(db: Session = Depends(get_db)) -> dict[str, object]:
    rows = db.scalars(select(MacroObservation).order_by(MacroObservation.timestamp.desc())).all()
    latest: dict[str, dict[str, object]] = {}
    for row in rows:
        if row.series_id not in latest:
            latest[row.series_id] = {
                "timestamp": row.timestamp,
                "value": row.value,
                "source": row.source,
                "updated_at": row.updated_at,
            }
    return {"ok": True, "data": latest}


@app.get("/events/upcoming")
def upcoming_events(days_ahead: int = 30, db: Session = Depends(get_db)) -> dict[str, object]:
    events = list_macro_events(db, days_ahead=days_ahead)
    return {
        "ok": True,
        "days_ahead": days_ahead,
        "data": [
            {
                "event_id": row.event_id,
                "timestamp": row.timestamp,
                "name": row.name,
                "country": row.country,
                "importance": row.importance,
                "description": row.description,
                "source": row.source,
                "updated_at": row.updated_at,
            }
            for row in events
        ],
    }


@app.get("/positions/cftc/latest")
def cftc_latest(db: Session = Depends(get_db)) -> dict[str, object]:
    row = db.scalar(select(CftcPosition).order_by(CftcPosition.timestamp.desc()))
    if not row:
        return {"ok": False, "reason": "No CFTC position found."}
    return {
        "ok": True,
        "data": {
            "market_name": row.market_name,
            "contract_market_code": row.contract_market_code,
            "exchange_code": row.exchange_code,
            "timestamp": row.timestamp,
            "open_interest": row.open_interest,
            "noncommercial_long": row.noncommercial_long,
            "noncommercial_short": row.noncommercial_short,
            "noncommercial_spreading": row.noncommercial_spreading,
            "commercial_long": row.commercial_long,
            "commercial_short": row.commercial_short,
            "noncommercial_net": row.noncommercial_net,
            "source": row.source,
            "updated_at": row.updated_at,
        },
    }


@app.get("/score/latest")
def score_latest(db: Session = Depends(get_db)) -> dict[str, object]:
    snapshot = db.scalar(select(GoldScoreSnapshot).order_by(GoldScoreSnapshot.timestamp.desc()))
    if not snapshot:
        return {"ok": False, "reason": "No score snapshot found."}
    return serialize_score(snapshot)


@app.post("/notify/feishu/test")
def notify_feishu_test(db: Session = Depends(get_db)) -> dict[str, object]:
    snapshot = db.scalar(select(GoldScoreSnapshot).order_by(GoldScoreSnapshot.timestamp.desc()))
    if snapshot:
        events = [
            {
                "timestamp": row.timestamp,
                "name": row.name,
                "importance": row.importance,
            }
            for row in list_macro_events(db, days_ahead=30)
        ]
        return send_score_alert_with_health(snapshot, get_data_health(db), events)
    return send_text_message("黄金走势监控测试消息：飞书机器人接口已连接。")


# ── 评分模型自我优化 API ──────────────────────────────────────────


@app.post("/score/optimize")
def score_optimize(
    n_iter: int = 150,
    horizon_days: int = 20,
    db: Session = Depends(get_db),
) -> dict[str, object]:
    """运行参数搜索，选出命中率最高的参数组合，存入数据库。

    参数：
    - n_iter: 随机搜索的候选参数组数（默认 150）
    - horizon_days: 回测展望期天数（默认 20）
    """
    with serialized_write():
        results = optimize_score_params(db, n_iter=n_iter, horizon_days=horizon_days)

    if not results or not results[0].get("ok"):
        return {"ok": False, "reason": "Optimization failed: no valid results.", "results": results}

    best = results[0]
    baseline = next((r for r in results if r["label"] == "baseline"), None)

    # 保存最优参数
    version = f"optimized_v{len(db.scalars(select(ScoreParamsVersion)).all()) + 1}"
    with serialized_write():
        save_best_params(
            db,
            best,
            version=version,
            horizon_days=horizon_days,
            notes=f"自动搜索最优参数 (n_iter={n_iter}, horizon={horizon_days}d)",
        )

    return {
        "ok": True,
        "version": version,
        "best": {
            "hit_rate": best.get("hit_rate"),
            "sample_count": best.get("sample_count"),
            "signal_count": best.get("signal_count"),
            "params": best["params"],
        },
        "baseline": {
            "hit_rate": baseline.get("hit_rate") if baseline else None,
            "sample_count": baseline.get("sample_count") if baseline else None,
        } if baseline else None,
        "top_5": [
            {
                "label": r["label"],
                "hit_rate": r.get("hit_rate"),
                "sample_count": r.get("sample_count"),
                "signal_ratio": r.get("signal_ratio"),
            }
            for r in results[:5]
            if r.get("ok")
        ],
        "message": f"Optimization complete. Best version: {version}. "
        f"Call POST /score/params/{version}/activate to use it.",
    }


@app.get("/score/params")
def list_score_params(db: Session = Depends(get_db)) -> dict[str, object]:
    """列出所有评分参数版本。"""
    rows = db.scalars(
        select(ScoreParamsVersion).order_by(ScoreParamsVersion.created_at.desc())
    ).all()
    return {
        "ok": True,
        "data": [
            {
                "id": r.id,
                "version": r.version,
                "hit_rate": r.hit_rate,
                "sample_count": r.sample_count,
                "backtest_horizon_days": r.backtest_horizon_days,
                "is_active": r.is_active,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "notes": r.notes,
            }
            for r in rows
        ],
    }


@app.post("/score/params/{version}/activate")
def activate_score_params(version: str, db: Session = Depends(get_db)) -> dict[str, object]:
    """激活指定参数版本，后续评分和调度将使用该版本。"""
    with serialized_write():
        activated = activate_version(db, version)
    if activated is None:
        return {"ok": False, "reason": f"Version '{version}' not found."}
    return {
        "ok": True,
        "version": activated.version,
        "hit_rate": activated.hit_rate,
        "message": f"Version '{version}' activated. Score computation will use these params.",
    }


@app.post("/score/compute/{version}")
def compute_score_with_version(version: str, db: Session = Depends(get_db)) -> dict[str, object]:
    """用指定参数版本计算评分并存储。"""
    row = db.scalar(
        select(ScoreParamsVersion).where(ScoreParamsVersion.version == version)
    )
    if row is None:
        # 支持 'default' 作为默认规则的别名
        if version == "default":
            snapshot = compute_and_store_gold_score(db)
            return serialize_score(snapshot)
        return {"ok": False, "reason": f"Version '{version}' not found."}

    params_dict = json.loads(row.params_json)
    params = ScoreParams.from_dict(params_dict)
    with serialized_write():
        snapshot = compute_and_store_gold_score_with_params(db, params, source=version)
    return serialize_score(snapshot)


# ── 辅助函数 ─────────────────────────────────────────────────────


def serialize_score(snapshot: GoldScoreSnapshot) -> dict[str, object]:
    raw_factors = json.loads(snapshot.factor_scores)
    if isinstance(raw_factors, dict) and "scores" in raw_factors:
        factor_scores = raw_factors.get("scores", {})
        factor_details = raw_factors.get("details", {})
    else:
        factor_scores = raw_factors
        factor_details = {}
    return {
        "ok": True,
        "timestamp": snapshot.timestamp,
        "total_score": snapshot.total_score,
        "direction": snapshot.direction,
        "factor_scores": factor_scores,
        "factor_details": factor_details,
        "risk_flags": json.loads(snapshot.risk_flags),
        "summary": snapshot.summary,
        "source": snapshot.source,
        "updated_at": snapshot.updated_at,
    }


def serialize_cftc_position(record) -> dict[str, object]:
    return {
        "market_name": record.market_name,
        "contract_market_code": record.contract_market_code,
        "exchange_code": record.exchange_code,
        "timestamp": record.timestamp,
        "open_interest": record.open_interest,
        "noncommercial_long": record.noncommercial_long,
        "noncommercial_short": record.noncommercial_short,
        "noncommercial_spreading": record.noncommercial_spreading,
        "commercial_long": record.commercial_long,
        "commercial_short": record.commercial_short,
        "noncommercial_net": record.noncommercial_net,
    }
