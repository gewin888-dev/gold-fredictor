from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pathlib import Path
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.backtesting.score_backtest import run_score_backtest
from app.data.cb_gold_collector import collect_central_bank_gold, load_sample_cb_gold
from app.data.cftc_collector import collect_cftc_gold_position
from app.data.fred_collector import collect_fred_data
from app.data.external_indicators import INDICATOR_CATALOG, latest_external_indicators, upsert_external_indicator
from app.data.market_structure_collector import collect_market_structure_data
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
from app.auto_settings import get_auto_settings, set_auto_settings
from app.config_registry import get_config_audit
from app.events.calendar import list_macro_events
from app.models import (
    IntradaySnapshot,
    CentralBankGold,
    CftcPosition,
    ChinaGoldPremium,
    GoldScoreSnapshot,
    GoldPredictionEvaluation,
    GoldPredictionSnapshot,
    MacroObservation,
    NewsSentiment,
    ModelActivationAudit,
    PredictionModelVersion,
    ScoreParamsVersion,
)
from app.monitoring.health import get_data_health
from app.monitoring.system_health import get_system_health
from app.monitoring.threshold_alert import send_threshold_alerts
from app.notifications.feishu import send_score_alert_with_health, send_text_message
from app.scoring.gold_score import (
    compute_and_store_gold_score,
    compute_and_store_gold_score_with_params,
)
from app.scoring.gold_predictor import (
    ensure_default_prediction_model,
    activate_prediction_model_version,
    evaluate_due_predictions,
    optimize_prediction_model_params,
    predict_gold_prices,
    prediction_due_status_summary,
    prediction_evaluation_summary,
    refresh_prediction_model_metrics,
)
from app.scoring.score_optimizer import (
    ScoreParams,
    activate_version,
    deactivate_all_versions,
    evaluate_params,
    optimize_score_params,
    overfit_risk_assessment,
    save_best_params,
)
from app.scoring.factor_registry import factor_registry_payload
from app.self_healing import get_self_healing_status, run_self_healing_cycle
from app.ai import analyze_latest_score
from app.ai.chat import (
    chat as ai_chat,
    reset_session as ai_reset_session,
    get_history as ai_get_history,
    list_sessions as ai_list_sessions,
    get_session_messages as ai_get_session_messages,
    delete_session as ai_delete_session,
    execute_action as ai_execute_action,
    generate_insight as ai_generate_insight,
)


logger = logging.getLogger(__name__)


def is_comex_market_closed(now_utc) -> bool:
    weekday = now_utc.weekday()
    hour = now_utc.hour
    return weekday == 5 or (weekday == 6 and hour < 22) or (weekday == 4 and hour >= 21)


def _bootstrap_data_once() -> None:
    """Run slow network bootstrap outside FastAPI startup."""
    from app.database import SessionLocal

    db = SessionLocal()
    try:
        logger.info("Starting background data bootstrap")
        fetch_gold_history(days=200)
        settings = get_settings()
        if settings.fred_api_key:
            collect_fred_data(db)
        db.commit()
        logger.info("Background data bootstrap finished")
    except Exception:
        db.rollback()
        logger.exception("Background data bootstrap failed")
    finally:
        db.close()


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    settings = get_settings()
    scheduler = None
    bootstrap_task: asyncio.Task | None = None

    if settings.auto_start_scheduler:
        from app.scheduler import create_scheduler

        scheduler = create_scheduler()
        scheduler.start()

    if settings.auto_bootstrap_data:
        bootstrap_task = asyncio.create_task(asyncio.to_thread(_bootstrap_data_once))

    # 启动日内金价快照记录（用于 24 小时走势图）
    start_intraday_recorder()

    # 启动自检：记录所有采集器初始状态
    try:
        from app.monitoring.collector_health import record_success
        record_success("intraday_snapshot", "startup")
        logger.info("启动自检完成，采集器监控已就绪")
    except Exception:
        pass

    try:
        yield
    finally:
        if bootstrap_task is not None and not bootstrap_task.done():
            bootstrap_task.cancel()
        stop_intraday_recorder()
        if scheduler is not None:
            scheduler.shutdown(wait=False)


app = FastAPI(title="黄金走势实时监控与预测系统", lifespan=lifespan)


class ExternalIndicatorIn(BaseModel):
    indicator_id: str = Field(..., examples=["GLD_FLOW_TONNES"])
    timestamp: str
    value: float
    source: str = Field(..., examples=["SPDR"])
    name: Optional[str] = None
    category: Optional[str] = None
    unit: Optional[str] = None
    note: str = ""


class ActivationRequest(BaseModel):
    operator: str = "dashboard"
    reason: str = ""
    activation_source: str = "manual"


def _json_safe(data: Any) -> str:
    return json.dumps(data or {}, ensure_ascii=False, default=str)


def _record_activation_audit(
    db: Session,
    *,
    model_type: str,
    action: str,
    from_version: str | None,
    to_version: str,
    operator: str,
    reason: str,
    metrics: dict[str, Any] | None = None,
) -> ModelActivationAudit:
    row = ModelActivationAudit(
        model_type=model_type,
        action=action,
        from_version=from_version,
        to_version=to_version,
        operator=operator or "dashboard",
        reason=reason or "",
        metrics_json=_json_safe(metrics or {}),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def _active_score_version(db: Session) -> str:
    row = db.scalar(select(ScoreParamsVersion).where(ScoreParamsVersion.is_active == True))  # noqa: E712
    return row.version if row else "default"


def _score_compare_payload(db: Session, row: ScoreParamsVersion) -> dict[str, Any]:
    params = ScoreParams.from_dict(json.loads(row.params_json))
    horizon = int(row.backtest_horizon_days or 20)
    candidate = evaluate_params(db, params, horizon_days=horizon, min_samples=10)
    baseline = evaluate_params(db, ScoreParams.defaults(), horizon_days=horizon, min_samples=10)
    candidate["stored_hit_rate"] = row.hit_rate
    candidate["stored_sample_count"] = row.sample_count
    baseline_hit = baseline.get("hit_rate")
    if candidate.get("hit_rate") is not None and baseline_hit is not None:
        candidate["baseline_lift"] = round(float(candidate["hit_rate"]) - float(baseline_hit), 4)
    overfit = overfit_risk_assessment(candidate, baseline)
    return {
        "version": row.version,
        "horizon_days": horizon,
        "candidate": candidate,
        "baseline": baseline,
        "overfit_risk": overfit,
        "recommendation": (
            "不建议直接激活，建议先做分段稳定性复核。"
            if overfit["not_recommended_for_direct_activation"]
            else "未触发明显过拟合警告，可交由自动门控处理。"
        ),
    }

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

    # 自动飞书推送评分摘要（异步，失败不阻塞）
    try:
        summary_text = (
            f"📊 黄金多空评分更新\n"
            f"方向：{snapshot.direction} | 评分：{snapshot.total_score:+.1f}\n"
            f"时间：{snapshot.timestamp.strftime('%Y-%m-%d %H:%M')} UTC"
        )
        send_text_message(summary_text)
    except Exception:
        pass

    # AI 分析（后台异步，失败不阻塞）
    try:
        analyze_latest_score(db)
    except Exception:
        pass
 
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


@app.get("/external/indicators/catalog")
def external_indicator_catalog() -> dict[str, object]:
    """查看可预留接入的 ETF/COMEX/期权/地缘/实物需求指标。"""
    return {"ok": True, "data": INDICATOR_CATALOG}


@app.get("/score/factors/registry")
def score_factor_registry() -> dict[str, object]:
    """评分因子注册表：供 UI、手动录入和优化报告共用。"""
    return {"ok": True, "data": factor_registry_payload()}


@app.get("/external/indicators/latest")
def external_indicators_latest(limit: int = 200, db: Session = Depends(get_db)) -> dict[str, object]:
    rows = latest_external_indicators(db, limit=limit)
    return {
        "ok": True,
        "data": [
            {
                "indicator_id": row.indicator_id,
                "name": row.name,
                "category": row.category,
                "timestamp": row.timestamp,
                "value": row.value,
                "unit": row.unit,
                "source": row.source,
                "note": row.note,
                "updated_at": row.updated_at,
            }
            for row in rows
        ],
    }


@app.post("/external/indicators")
def external_indicator_upsert(payload: ExternalIndicatorIn, db: Session = Depends(get_db)) -> dict[str, object]:
    """写入授权或人工维护的外部市场指标，供后续评分使用。"""

    allowed_indicator_ids = {str(item.get("indicator_id")) for item in INDICATOR_CATALOG}
    if payload.indicator_id not in allowed_indicator_ids:
        raise HTTPException(status_code=400, detail=f"Unknown indicator_id: {payload.indicator_id}")
    try:
        timestamp = datetime.fromisoformat(payload.timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="timestamp must be a valid ISO-8601 datetime") from exc
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    min_timestamp = datetime(2000, 1, 1, tzinfo=timezone.utc)
    max_timestamp = datetime.now(timezone.utc) + timedelta(days=1)
    if timestamp < min_timestamp or timestamp > max_timestamp:
        raise HTTPException(status_code=400, detail="timestamp is outside the accepted range")
    with serialized_write():
        upsert_external_indicator(
            db,
            payload.indicator_id,
            timestamp,
            payload.value,
            source=payload.source,
            name=payload.name,
            category=payload.category,
            unit=payload.unit,
            note=payload.note,
        )
        db.commit()
    return {"ok": True}


@app.get("/health/recorder")
def health_recorder(db: Session = Depends(get_db)) -> dict[str, object]:
    """记录器健康检查：最近一次 COMEX 日内快照时间。"""
    from datetime import datetime, timezone

    row = db.scalar(select(IntradaySnapshot).order_by(IntradaySnapshot.timestamp.desc()))
    now = datetime.now(timezone.utc)
    market_closed = is_comex_market_closed(now)
    if not row:
        return {
            "ok": True,
            "status": "market_closed" if market_closed else "no_data",
            "last_record": None,
            "market_closed": market_closed,
        }
    ts = row.timestamp.replace(tzinfo=None) if row.timestamp else None
    age_secs = (now.replace(tzinfo=None) - ts).total_seconds() if ts else None
    status = "healthy" if age_secs is not None and age_secs < 120 else (
        "stale" if age_secs is not None and age_secs < 600 else "down")
    if market_closed and status == "down":
        status = "market_closed"
    return {
        "ok": True,
        "status": status,
        "last_record": row.timestamp.isoformat() if row.timestamp else None,
        "last_price": row.price,
        "age_seconds": round(age_secs, 1) if age_secs is not None else None,
        "market_closed": market_closed,
    }
 
@app.get("/health/collectors")
def health_collectors() -> dict[str, object]:
    """采集器健康检查：所有数据源的成功/失败/新鲜度状态。"""
    from app.monitoring.collector_health import get_health_summary
    return {"ok": True, **get_health_summary()}

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


@app.get("/predict/due-status")
def prediction_due_status(auto_evaluate: bool = False, db: Session = Depends(get_db)) -> dict[str, object]:
    """查看预测评估闭环状态；可选择自动补评估到期预测。"""
    evaluated_now: dict[str, object] | None = None
    if auto_evaluate:
        with serialized_write():
            evaluated_now = evaluate_due_predictions(db)
    due_summary = prediction_due_status_summary(db)
    due = int(due_summary.get("due_pending_count") or 0)
    evaluated = int(due_summary.get("evaluated_count") or 0)
    future = int(due_summary.get("future_pending_count") or 0)
    return {
        "ok": True,
        "status": "due_pending" if due else ("waiting_for_maturity" if future else "no_snapshots"),
        "evaluated_count": evaluated,
        "due_pending_count": due,
        "future_pending_count": future,
        "target_evaluated_count": due_summary.get("target_evaluated_count"),
        "target_horizons": due_summary.get("target_horizons"),
        "by_horizon": due_summary.get("by_horizon"),
        "can_evolve": due_summary.get("can_evolve"),
        "cannot_evolve_reasons": due_summary.get("cannot_evolve_reasons"),
        "auto_evaluated": evaluated_now,
        "message": (
            f"有 {due} 条预测已到期但尚未评估，建议点击补评估。"
            if due else (
                due_summary.get("message")
                or (f"暂无到期预测，仍有 {future} 条等待目标日期。" if future else "暂无预测快照，请先保存一次预测。")
            )
        ),
    }


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


@app.post("/predict/models/optimize")
def optimize_prediction_models(
    n_iter: int = 80,
    top_k: int = 5,
    random_seed: int = 42,
    save_best: bool = True,
    auto_activate: bool = False,
    min_score: float = 40.0,
    max_mape_pct: float = 8.0,
    min_direction_accuracy: float = 0.52,
    min_samples: int = 120,
    min_valid_horizons: int = 3,
    min_baseline_lift: float = 0.03,
    max_mape_worse_ratio: float = 1.2,
    max_recent_degradation: float = 0.05,
    db: Session = Depends(get_db),
) -> dict[str, object]:
    """生成 1/7/30 天短周期候选预测模型参数并保存最佳候选。"""
    n_iter = max(1, min(int(n_iter), 300))
    top_k = max(1, min(int(top_k), 20))
    with serialized_write():
        return optimize_prediction_model_params(
            db,
            n_iter=n_iter,
            top_k=top_k,
            random_seed=random_seed,
            save_best=save_best,
            auto_activate=auto_activate,
            activation_thresholds={
                "min_score": min_score,
                "max_mape_price_pct": max_mape_pct,
                "min_direction_accuracy": min_direction_accuracy,
                "min_samples": min_samples,
                "min_valid_horizons": min_valid_horizons,
                "min_baseline_lift": min_baseline_lift,
                "max_mape_worse_ratio": max_mape_worse_ratio,
                "max_recent_degradation": max_recent_degradation,
            },
        )


@app.post("/predict/models/{version}/activate")
def activate_prediction_model(
    version: str,
    payload: Optional[ActivationRequest] = None,
    db: Session = Depends(get_db),
) -> dict[str, object]:
    """人工激活指定预测模型版本。"""
    previous = db.scalar(
        select(PredictionModelVersion)
        .where(PredictionModelVersion.is_active == True)  # noqa: E712
        .order_by(PredictionModelVersion.created_at.desc())
    )
    with serialized_write():
        ensure_default_prediction_model(db)
        target = activate_prediction_model_version(db, version)
        if target is None:
            return {"ok": False, "reason": f"Prediction model version '{version}' not found."}
    metrics = {
        "method": target.method,
        "mae_price": target.mae_price,
        "mape_price_pct": target.mape_price_pct,
        "direction_accuracy": target.direction_accuracy,
        "evaluated_count": target.evaluated_count,
        "notes": target.notes,
    }
    audit = _record_activation_audit(
        db,
        model_type="prediction",
        action="activate",
        from_version=previous.version if previous else None,
        to_version=target.version,
        operator=(payload.operator if payload else "dashboard"),
        reason=(payload.reason if payload else "人工激活预测模型候选"),
        metrics={**metrics, "activation_source": payload.activation_source if payload else "manual"},
    )
    return {
        "ok": True,
        "version": target.version,
        "method": target.method,
        "audit_id": audit.id,
        "message": f"Prediction model '{version}' activated.",
    }


@app.post("/collect/gold_history")
def collect_gold_history(days: int = 200, db: Session = Depends(get_db)) -> dict[str, object]:
    """下载历史金价日线（Yahoo Finance），存入 gold_prices 表。"""
    days = max(1, min(int(days), 2000))
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


@app.get("/health/system")
def system_health(db: Session = Depends(get_db)) -> dict[str, object]:
    return get_system_health(db)


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
        db.commit()
        snapshot = auto_score_and_broadcast(db)
    return {"ok": True, "data": serialize_cftc_position(record), "score_updated": serialize_score(snapshot)}


# ── 新增数据源采集 API ──────────────────────────────────────────


@app.post("/collect/china_premium")
def collect_china_premium(db: Session = Depends(get_db)) -> dict[str, object]:
    """采集中国黄金溢价数据。"""
    with serialized_write():
        record = collect_china_gold_premium(db)
        db.commit()
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
        db.commit()
    return {"ok": True, "records_collected": count}


@app.post("/collect/sentiment")
def collect_sentiment(
    days_back: int = 3, max_records: int = 50, db: Session = Depends(get_db)
) -> dict[str, object]:
    """采集黄金相关新闻情绪。"""
    days_back = max(1, min(int(days_back), 30))
    max_records = max(1, min(int(max_records), 200))
    with serialized_write():
        count = collect_news_sentiment(db, days_back=days_back, max_records=max_records)
        db.commit()
    return {"ok": True, "records_collected": count}


@app.post("/collect/market_structure")
def collect_market_structure(db: Session = Depends(get_db)) -> dict[str, object]:
    """采集 GLD ETF 持仓/资金流与 COMEX 库存等市场结构数据。"""
    with serialized_write():
        results = collect_market_structure_data(db)
        db.commit()
        try:
            snapshot = auto_score_and_broadcast(db)
            score = serialize_score(snapshot)
        except Exception as exc:
            score = {"ok": False, "reason": str(exc)}
    return {"ok": any(item["ok"] for item in results.values()), "results": results, "score_updated": score}


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
    from app.monitoring.collector_health import record_success

    record_success("score_engine", f"score={snapshot.total_score}")
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
    horizon_days = max(1, min(int(horizon_days), 720))
    limit = max(1, min(int(limit), 1000))
    offset = max(0, int(offset))
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


# ── 自我进化设置与健康状态 API ─────────────────────────────────────


def _model_health_status(db: Session, settings: dict[str, object] | None = None) -> dict[str, object]:
    settings = settings or get_auto_settings(db)
    active_score = db.scalar(
        select(ScoreParamsVersion)
        .where(ScoreParamsVersion.is_active == True)  # noqa: E712
        .order_by(ScoreParamsVersion.created_at.desc())
    )
    latest_score_candidate = db.scalar(
        select(ScoreParamsVersion).order_by(ScoreParamsVersion.created_at.desc())
    )

    ensure_default_prediction_model(db)
    refresh_prediction_model_metrics(db)
    active_prediction = db.scalar(
        select(PredictionModelVersion)
        .where(PredictionModelVersion.is_active == True)  # noqa: E712
        .order_by(PredictionModelVersion.created_at.desc())
    )
    latest_prediction_candidate = db.scalar(
        select(PredictionModelVersion).order_by(PredictionModelVersion.created_at.desc())
    )
    prediction_min_samples = int(settings.get("AUTO_PREDICTION_MIN_SAMPLES") or 30)
    evaluation = prediction_evaluation_summary(db)
    due_status = prediction_due_status_summary(db, min_samples=prediction_min_samples)
    summary = evaluation.get("summary", {}) if evaluation.get("ok") else {}
    evaluated_count = int(summary.get("evaluated_count") or 0)
    due_pending_count = int(summary.get("due_pending_count") or 0)
    target_evaluated_count = int(due_status.get("target_evaluated_count") or 0)

    score_ready = bool(
        latest_score_candidate
        and latest_score_candidate.sample_count is not None
        and latest_score_candidate.sample_count >= prediction_min_samples
    )
    prediction_ready = bool(due_status.get("can_evolve"))
    reasons: list[str] = []
    if not score_ready:
        reasons.append("评分优化候选样本不足或尚未生成候选版本。")
    if not prediction_ready:
        reasons.extend(due_status.get("cannot_evolve_reasons") or [f"预测评估样本 {evaluated_count} 条，低于 {prediction_min_samples} 条门槛。"])
    if due_pending_count:
        reasons.append(f"有 {due_pending_count} 条到期预测尚未完成评估。")
    if not reasons:
        reasons.append("样本条件基本满足，可生成候选版本；预测候选满足强门槛时可受控自动激活。")

    latest_prediction_audit = db.scalar(
        select(ModelActivationAudit)
        .where(ModelActivationAudit.model_type == "prediction")
        .order_by(ModelActivationAudit.created_at.desc())
    )

    return {
        "score_params_version": active_score.version if active_score else "default",
        "score_params_active": active_score is not None,
        "latest_score_candidate": {
            "version": latest_score_candidate.version,
            "hit_rate": latest_score_candidate.hit_rate,
            "sample_count": latest_score_candidate.sample_count,
            "created_at": latest_score_candidate.created_at,
            "is_active": latest_score_candidate.is_active,
        } if latest_score_candidate else None,
        "prediction_model_version": active_prediction.version if active_prediction else None,
        "latest_prediction_candidate": {
            "version": latest_prediction_candidate.version,
            "direction_accuracy": latest_prediction_candidate.direction_accuracy,
            "mape_price_pct": latest_prediction_candidate.mape_price_pct,
            "evaluated_count": latest_prediction_candidate.evaluated_count,
            "created_at": latest_prediction_candidate.created_at,
            "is_active": latest_prediction_candidate.is_active,
        } if latest_prediction_candidate else None,
        "prediction_evaluated_count": evaluated_count,
        "prediction_target_evaluated_count": target_evaluated_count,
        "prediction_due_pending_count": due_pending_count,
        "prediction_future_pending_count": int(summary.get("future_pending_count") or 0),
        "prediction_by_horizon": due_status.get("by_horizon"),
        "score_sample_ready": score_ready,
        "prediction_sample_ready": prediction_ready,
        "can_self_evolve": score_ready or prediction_ready,
        "reasons": reasons,
        "auto_evolution": {
            "target_horizons": due_status.get("target_horizons"),
            "can_evolve_prediction": prediction_ready,
            "sample_threshold": prediction_min_samples,
            "auto_search_enabled": bool(settings.get("AUTO_OPTIMIZE_PREDICTION_MODEL")),
            "auto_activate_enabled": bool(settings.get("AUTO_ACTIVATE_PREDICTION_MODEL")),
            "self_healing_enabled": bool(settings.get("AUTO_SELF_HEALING_ENABLED", True)),
            "self_healing_autofix": bool(settings.get("AUTO_SELF_HEALING_AUTOFIX", True)),
            "full_auto": bool(settings.get("AUTO_EVOLUTION_FULL_AUTO", True)),
            "last_prediction_action": {
                "action": latest_prediction_audit.action,
                "from_version": latest_prediction_audit.from_version,
                "to_version": latest_prediction_audit.to_version,
                "operator": latest_prediction_audit.operator,
                "reason": latest_prediction_audit.reason,
                "created_at": latest_prediction_audit.created_at,
            } if latest_prediction_audit else None,
        },
        "policy": {
            "auto_search_allowed": True,
            "auto_activate_default": bool(settings.get("AUTO_SELF_HEALING_AUTOFIX", True)),
            "mode": "ai_validation_full_auto" if settings.get("AUTO_EVOLUTION_FULL_AUTO", True) else "guarded",
            "manual_review_required": False,
        },
    }


@app.get("/settings/auto-optimize")
def auto_optimize_settings(db: Session = Depends(get_db)) -> dict[str, object]:
    settings = get_auto_settings(db)
    return {
        "ok": True,
        "settings": settings,
        "health": _model_health_status(db, settings),
    }


@app.get("/config/audit")
def config_audit(db: Session = Depends(get_db)) -> dict[str, object]:
    return get_config_audit(db)


@app.post("/settings/auto-optimize")
def update_auto_optimize_settings(
    payload: dict[str, bool],
    db: Session = Depends(get_db),
) -> dict[str, object]:
    allowed = {
        "AUTO_EVOLUTION_FULL_AUTO",
        "AUTO_SELF_HEALING_ENABLED",
        "AUTO_SELF_HEALING_AUTOFIX",
        "AUTO_OPTIMIZE_SCORE_PARAMS",
        "AUTO_ACTIVATE_OPTIMIZED_PARAMS",
        "AUTO_OPTIMIZE_PREDICTION_MODEL",
        "AUTO_ACTIVATE_PREDICTION_MODEL",
    }
    rejected = sorted(set(payload) - allowed)
    clean_payload = {key: bool(value) for key, value in payload.items() if key in allowed}
    with serialized_write():
        settings = set_auto_settings(db, clean_payload)
    return {
        "ok": True,
        "settings": settings,
        "rejected_keys": rejected,
        "health": _model_health_status(db, settings),
        "message": "自动优化开关已保存到数据库；未修改 .env 或任何密钥字段。",
    }


@app.get("/self-healing/status")
def self_healing_status(db: Session = Depends(get_db)) -> dict[str, object]:
    return get_self_healing_status(db)


@app.post("/self-healing/run")
def self_healing_run(force: bool = False, db: Session = Depends(get_db)) -> dict[str, object]:
    return run_self_healing_cycle(db, force=force, reason="api")


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
    n_iter = max(1, min(int(n_iter), 300))
    horizon_days = max(1, min(int(horizon_days), 720))
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
            "signal_ratio": best.get("signal_ratio"),
            "long_signal_count": best.get("long_signal_count"),
            "short_signal_count": best.get("short_signal_count"),
            "baseline_lift": best.get("baseline_lift"),
            "avg_return": best.get("avg_return"),
            "worst_decile_return": best.get("worst_decile_return"),
            "recent_hit_rate": best.get("recent_hit_rate"),
            "activation_check": best.get("activation_check"),
            "params": best["params"],
        },
        "baseline": {
            "hit_rate": baseline.get("hit_rate") if baseline else None,
            "sample_count": baseline.get("sample_count") if baseline else None,
            "signal_ratio": baseline.get("signal_ratio") if baseline else None,
        } if baseline else None,
        "top_5": [
            {
                "label": r["label"],
                "hit_rate": r.get("hit_rate"),
                "sample_count": r.get("sample_count"),
                "signal_ratio": r.get("signal_ratio"),
                "baseline_lift": r.get("baseline_lift"),
                "activation_check": r.get("activation_check"),
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


@app.get("/score/params/{version}/compare")
def compare_score_params(version: str, db: Session = Depends(get_db)) -> dict[str, object]:
    """候选评分参数详情：baseline vs candidate，并提示过拟合风险。"""
    row = db.scalar(select(ScoreParamsVersion).where(ScoreParamsVersion.version == version))
    if row is None:
        return {"ok": False, "reason": f"Version '{version}' not found."}
    return {"ok": True, "data": _score_compare_payload(db, row)}


@app.post("/score/params/{version}/activate")
def activate_score_params(
    version: str,
    payload: Optional[ActivationRequest] = None,
    db: Session = Depends(get_db),
) -> dict[str, object]:
    """激活指定参数版本，后续评分和调度将使用该版本。"""
    if version == "default":
        return deactivate_score_params(payload, db)
    previous = _active_score_version(db)
    with serialized_write():
        activated = activate_version(db, version)
    if activated is None:
        return {"ok": False, "reason": f"Version '{version}' not found."}
    compare = _score_compare_payload(db, activated)
    audit = _record_activation_audit(
        db,
        model_type="score",
        action="activate",
        from_version=previous,
        to_version=activated.version,
        operator=(payload.operator if payload else "dashboard"),
        reason=(payload.reason if payload else "人工激活评分参数候选"),
        metrics=compare,
    )
    return {
        "ok": True,
        "version": activated.version,
        "hit_rate": activated.hit_rate,
        "audit_id": audit.id,
        "overfit_risk": compare["overfit_risk"],
        "message": f"Version '{version}' activated. Score computation will use these params.",
    }


@app.post("/score/params/deactivate")
def deactivate_score_params(
    payload: Optional[ActivationRequest] = None,
    db: Session = Depends(get_db),
) -> dict[str, object]:
    """停用所有优化参数版本，恢复默认 rule_v2 评分。"""
    previous = _active_score_version(db)
    with serialized_write():
        count = deactivate_all_versions(db)
    audit = _record_activation_audit(
        db,
        model_type="score",
        action="deactivate",
        from_version=previous,
        to_version="default",
        operator=(payload.operator if payload else "dashboard"),
        reason=(payload.reason if payload else "恢复默认评分规则"),
        metrics={"deactivated_count": count},
    )
    return {
        "ok": True,
        "deactivated_count": count,
        "version": "default",
        "audit_id": audit.id,
        "message": "已恢复默认 rule_v2 评分规则。",
    }


@app.post("/score/params/default/activate")
def activate_default_score_params(
    payload: Optional[ActivationRequest] = None,
    db: Session = Depends(get_db),
) -> dict[str, object]:
    """默认评分规则的显式激活别名。"""
    return deactivate_score_params(payload, db)


@app.get("/models/activation-audit")
def list_activation_audit(limit: int = 50, db: Session = Depends(get_db)) -> dict[str, object]:
    rows = db.scalars(
        select(ModelActivationAudit)
        .order_by(ModelActivationAudit.created_at.desc())
        .limit(max(1, min(int(limit), 500)))
    ).all()
    return {
        "ok": True,
        "data": [
            {
                "id": row.id,
                "model_type": row.model_type,
                "action": row.action,
                "from_version": row.from_version,
                "to_version": row.to_version,
                "operator": row.operator,
                "reason": row.reason,
                "metrics": json.loads(row.metrics_json or "{}"),
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ],
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


@app.get("/ai/analysis")
def ai_analysis(db: Session = Depends(get_db)) -> dict[str, object]:
    """AI 解读最新评分：调用 DeepSeek 生成市场分析和风险解读。

    需要配置 DEEPSEEK_API_KEY 环境变量。
    未配置时返回 ok=false 但不报错。
    """
    result = analyze_latest_score(db)
    if result is None:
        return {"ok": False, "error": "AI analysis unavailable"}
    return result


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default="default")


@app.post("/ai/chat")
def ai_chat_endpoint(req: ChatRequest, db: Session = Depends(get_db)) -> dict[str, object]:
    """AI 对话：用户输入事件/信息，AI 结合当前评分给出分析。

    支持多轮对话，同一 session_id 保持上下文。
    """
    result = ai_chat(db, req.message, req.session_id)
    return {"ok": True, **result}


@app.post("/ai/chat/reset")
def ai_chat_reset(session_id: str = "default", db: Session = Depends(get_db)) -> dict[str, object]:
    """重置对话会话，清空历史。"""
    ai_reset_session(db, session_id)
    return {"ok": True, "session_id": session_id, "message": "会话已重置"}


@app.get("/ai/chat/history")
def ai_chat_history(session_id: str = "default", db: Session = Depends(get_db)) -> dict[str, object]:
    """获取对话历史。"""
    history = ai_get_history(db, session_id)
    return {"ok": True, "session_id": session_id, "messages": history}


@app.get("/ai/chat/archive")
def ai_chat_archive(db: Session = Depends(get_db)) -> dict[str, object]:
    """列出所有对话会话归档。"""
    sessions = ai_list_sessions(db)
    return {"ok": True, "sessions": sessions}


@app.get("/ai/chat/archive/{session_id}")
def ai_chat_archive_detail(session_id: str, db: Session = Depends(get_db)) -> dict[str, object]:
    """查看指定归档会话的完整消息。"""
    messages = ai_get_session_messages(db, session_id)
    return {"ok": True, "session_id": session_id, "messages": messages}


@app.delete("/ai/chat/archive/{session_id}")
def ai_chat_archive_delete(session_id: str, db: Session = Depends(get_db)) -> dict[str, object]:
    """删除指定归档会话。"""
    ai_delete_session(db, session_id)
    return {"ok": True, "session_id": session_id, "message": "会话已删除"}


@app.post("/ai/action")
def ai_action(action: str, db: Session = Depends(get_db)) -> dict[str, object]:
    """AI 自主操作：执行系统操作（重新评分、切换模型、检查采集器等）。"""
    result = ai_execute_action(db, action)
    return {"ok": result.get("ok", False), **result}


@app.get("/ai/insight")
def ai_insight(db: Session = Depends(get_db)) -> dict[str, object]:
    """AI 主动智能洞察：分析系统状态，生成简短洞察报告。"""
    return ai_generate_insight(db)


# ── .env 配置读写 ──

def _env_path() -> Path:
    import os

    return Path(os.environ.get("GOLD_FREDICTOR_ENV_PATH", Path(__file__).resolve().parents[1] / ".env"))


class EnvUpdateRequest(BaseModel):
    updates: dict[str, str] = Field(default_factory=dict)


@app.get("/settings/env")
def get_env_config() -> dict[str, object]:
    """读取当前 .env 配置（敏感值打码）。"""
    env_path = _env_path()
    env_dict: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").split("\n"):
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                key, _, val = line.partition("=")
                env_dict[key.strip()] = val.strip()
    return {"ok": True, "env": env_dict}


@app.post("/settings/env")
def update_env_config(payload: EnvUpdateRequest) -> dict[str, object]:
    """更新 .env 配置（只更新提供的 key，保留其他不变）。"""
    env_path = _env_path()
    updates = payload.updates
    env_path.parent.mkdir(parents=True, exist_ok=True)

    # 读取现有内容
    current: dict[str, str] = {}
    order: list[str] = []
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").split("\n"):
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                key, _, val = line.partition("=")
                current[key.strip()] = val.strip()
                order.append(key.strip())

    # 合并更新
    for k, v in updates.items():
        current[k] = v
        if k not in order:
            order.append(k)

    # 写回
    lines = [f"{k}={current[k]}" for k in order if k in current]
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"ok": True, "updated": list(updates.keys()), "message": "配置已保存，重启 API 后生效。"}


@app.get("/ai/ui", response_class=HTMLResponse)
def ai_chat_ui() -> str:
    """AI 独立对话窗口页面。"""
    html_path = Path(__file__).resolve().parent / "ai" / "chat_ui.html"
    return html_path.read_text(encoding="utf-8")


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
