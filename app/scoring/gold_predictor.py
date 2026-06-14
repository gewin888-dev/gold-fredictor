"""金价预测模型 v2 — 多信号集成。

方法：
  短期限 (1-30天): 动量趋势 + 评分方向 → 线性回归
  长期限 (90-360天): 宏观环境 + 持仓结构 → 分组均值
  置信度: 基于样本量和标准差，0-1 评分

注意：长期预测不确定性极大，仅供参考。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from uuid import uuid4
import json
import random
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import configured_prediction_sources
from app.models import (
    CftcPosition,
    GoldPredictionEvaluation,
    GoldPredictionSnapshot,
    GoldPrice,
    GoldScoreSnapshot,
    MacroObservation,
    PredictionModelVersion,
)
from app.models import utc_now

HORIZONS = [1, 7, 30, 90, 180, 360]
MIN_RETURN_SAMPLES = 20
EXCLUDED_TRAINING_SOURCES = {"SAMPLE", "ESTIMATE", "MANUAL", "JSON"}
DEFAULT_MODEL_VERSION = "predictor_v2_baseline"

# 宏观指标 ID
NOMINAL_RATE = "DGS10"
REAL_RATE = "DFII10"
VIX = "VIXCLS"
DOLLAR = "DTWEXBGS"


DEFAULT_MODEL_PARAMS: dict[str, float] = {
    "short_momentum_weight": 0.6,
    "short_score_weight": 0.4,
    "score_similarity_sigma": 15.0,
    "short_return_clip_pct": 30.0,
    "long_annual_base_return_pct": 8.0,
    "long_macro_adjustment_scale": 3.0,
    "long_return_clip_low_pct": -20.0,
    "long_return_clip_high_pct": 50.0,
    "interval_std_multiplier": 1.5,
}

PREDICTION_PARAM_SPACE: dict[str, list[float]] = {
    "short_momentum_weight": [0.35, 0.45, 0.55, 0.6, 0.7, 0.8],
    "short_score_weight": [0.2, 0.3, 0.4, 0.45, 0.55, 0.65],
    "score_similarity_sigma": [8.0, 10.0, 12.0, 15.0, 18.0, 22.0, 28.0],
    "short_return_clip_pct": [8.0, 12.0, 18.0, 24.0, 30.0],
    "long_annual_base_return_pct": [0.0, 3.0, 5.0, 8.0, 10.0, 12.0],
    "long_macro_adjustment_scale": [0.0, 1.0, 2.0, 3.0, 4.0],
    "long_return_clip_low_pct": [-30.0, -20.0, -15.0, -10.0],
    "long_return_clip_high_pct": [20.0, 30.0, 40.0, 50.0],
    "interval_std_multiplier": [1.0, 1.25, 1.5, 1.75, 2.0],
}

OPTIMIZATION_HORIZON_WEIGHTS: dict[str, float] = {
    "1": 0.18,
    "7": 0.22,
    "30": 0.24,
    "90": 0.18,
    "180": 0.10,
    "360": 0.08,
}


@dataclass
class PricePrediction:
    horizon_days: int
    current_price: float
    predicted_price: float
    expected_return_pct: float
    confidence_low: float
    confidence_high: float
    sample_count: int
    reliability: float  # 0-1 置信度
    method: str
    note: str


def ensure_default_prediction_model(db: Session) -> PredictionModelVersion:
    """确保默认预测模型版本存在并处于激活状态。"""
    has_active = db.scalar(
        select(PredictionModelVersion).where(PredictionModelVersion.is_active == True)  # noqa: E712
    )
    row = db.scalar(
        select(PredictionModelVersion).where(PredictionModelVersion.version == DEFAULT_MODEL_VERSION)
    )
    if row is None:
        row = PredictionModelVersion(
            version=DEFAULT_MODEL_VERSION,
            method="multi_signal_ensemble_v2",
            params_json=json_dumps(DEFAULT_MODEL_PARAMS),
            is_active=has_active is None,
            notes="默认基线预测模型：短期动量+评分回归，长期宏观环境+持仓结构。",
        )
        db.add(row)
        db.commit()
        db.refresh(row)
    elif has_active is None:
        row.is_active = True
        db.commit()
        db.refresh(row)
    return row


def json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def _active_prediction_model(db: Session) -> PredictionModelVersion:
    default_model = ensure_default_prediction_model(db)
    active = db.scalar(
        select(PredictionModelVersion)
        .where(PredictionModelVersion.is_active == True)  # noqa: E712
        .order_by(PredictionModelVersion.created_at.desc())
    )
    return active or default_model


def _model_params(model: PredictionModelVersion) -> dict[str, float]:
    try:
        raw = json.loads(model.params_json or "{}")
    except json.JSONDecodeError:
        raw = {}
    params = dict(DEFAULT_MODEL_PARAMS)
    for key, default_value in DEFAULT_MODEL_PARAMS.items():
        try:
            params[key] = float(raw.get(key, default_value))
        except (TypeError, ValueError):
            params[key] = default_value
    return params


def _param(params: dict[str, float], key: str) -> float:
    return float(params.get(key, DEFAULT_MODEL_PARAMS[key]))


def _load_macro_snapshot(db: Session) -> dict[str, float]:
    """加载最新宏观指标值。"""
    macro = {}
    for sid in [NOMINAL_RATE, REAL_RATE, VIX, DOLLAR]:
        row = db.scalar(
            select(MacroObservation)
            .where(MacroObservation.series_id == sid)
            .order_by(MacroObservation.timestamp.desc())
        )
        if row:
            macro[sid] = row.value
    return macro


def _load_cftc_position(db: Session) -> float | None:
    """加载 CFTC 净多占比。"""
    row = db.scalar(select(CftcPosition).order_by(CftcPosition.timestamp.desc()))
    if row and row.open_interest > 0:
        return row.noncommercial_net / row.open_interest
    return None


def _prediction_training_context(db: Session) -> dict[str, Any]:
    score_rows = db.scalars(
        select(GoldScoreSnapshot).order_by(GoldScoreSnapshot.timestamp.asc())
    ).all()
    if not score_rows:
        return {"ok": False, "reason": "no score history"}

    all_scores_df = pd.DataFrame([
        {"timestamp": pd.Timestamp(r.timestamp), "total_score": r.total_score, "source": r.source}
        for r in score_rows
    ])
    allowed_sources = configured_prediction_sources()
    sources_upper = all_scores_df["source"].astype(str).str.upper()
    scores_df = all_scores_df[
        all_scores_df["source"].astype(str).isin(allowed_sources)
        & ~sources_upper.isin(EXCLUDED_TRAINING_SOURCES)
    ].copy()

    gold_rows = db.scalars(select(GoldPrice).order_by(GoldPrice.date.asc())).all()
    if not gold_rows:
        return {"ok": False, "reason": "no gold price data"}

    gold_df = pd.DataFrame([
        {"timestamp": pd.Timestamp(r.date), "gold_price": r.close}
        for r in gold_rows
    ]).sort_values("timestamp")

    if scores_df.empty or gold_df.empty:
        return {
            "ok": False,
            "reason": "insufficient trusted v2 score data",
            "required_score_sources": sorted(allowed_sources),
            "available_score_sources": sorted(all_scores_df["source"].astype(str).unique().tolist()),
        }

    merged = pd.merge_asof(
        scores_df.sort_values("timestamp"),
        gold_df.sort_values("timestamp"),
        on="timestamp", direction="backward"
    ).dropna(subset=["gold_price"])

    return {
        "ok": True,
        "all_scores_df": all_scores_df,
        "scores_df": scores_df,
        "gold_df": gold_df,
        "merged": merged,
        "gold_idx": gold_df.set_index("timestamp")["gold_price"],
        "allowed_sources": allowed_sources,
    }


def predict_gold_prices(db: Session, persist: bool = False) -> dict[str, Any]:
    """v2 多信号集成预测。"""
    model = _active_prediction_model(db)
    params = _model_params(model)

    context = _prediction_training_context(db)
    if not context.get("ok"):
        return context

    all_scores_df = context["all_scores_df"]
    scores_df = context["scores_df"]
    gold_df = context["gold_df"]
    merged = context["merged"]
    gold_idx = context["gold_idx"]
    allowed_sources = context["allowed_sources"]

    current_price = float(gold_df.iloc[-1]["gold_price"])
    current_score_row = scores_df.iloc[-1]
    current_score = float(current_score_row["total_score"])
    current_score_source = str(current_score_row.get("source") or "")

    # ── 加载宏观快照 ──
    macro = _load_macro_snapshot(db)
    cftc_net = _load_cftc_position(db)

    # ── 构建收益数据 ──
    ret_data = _build_return_data(merged, gold_idx)
    evaluation = _walk_forward_evaluation(merged, gold_idx, params)

    # ── 预测各期限 ──
    predictions = []
    for horizon in HORIZONS:
        if horizon <= 30:
            pred = _short_term_predict(horizon, current_price, current_score, merged, gold_idx, ret_data, params)
        else:
            pred = _long_term_predict(horizon, current_price, macro, cftc_net, merged, gold_idx, ret_data, params)
        predictions.append(pred)

    run_id = None
    if persist:
        run_id = persist_prediction_snapshots(
            db=db,
            predictions=predictions,
            model_version=model.version,
            score_value=current_score,
            score_source=current_score_source,
            input_summary={
                "training_sources": sorted(allowed_sources),
                "current_score": current_score,
                "current_score_source": current_score_source,
                "macro": macro,
                "cftc_net_ratio": cftc_net,
                "method": "multi_signal_ensemble_v2",
                "model_version": model.version,
                "model_params": params,
            },
        )

    # ── 统计 ──
    allowed_count = int(all_scores_df["source"].astype(str).isin(allowed_sources).sum())
    total_count = len(all_scores_df)
    excluded_count = total_count - allowed_count

    return {
        "ok": True,
        "current_price": round(current_price, 2),
        "current_score": current_score,
        "method": "multi_signal_ensemble_v2",
        "model_version": model.version,
        "model_params": params,
        "persisted_run_id": run_id,
        "method_note": "短期：动量+评分回归；长期：宏观环境+持仓结构。训练只使用配置的同版本评分源；可靠性<0.5的长期预测应弱化参考。",
        "training_sources": sorted(allowed_sources),
        "data_quality": {
            "score_rows": len(scores_df),
            "gold_price_rows": len(gold_df),
            "allowed_score_rows": allowed_count,
            "excluded_score_rows": excluded_count,
            "total_score_rows": total_count,
        },
        "evaluation": evaluation,
        "predictions": [
            {
                "horizon": f"{p.horizon_days}天",
                "days": p.horizon_days,
                "predicted": round(p.predicted_price, 2),
                "return_pct": round(p.expected_return_pct, 2),
                "low": round(p.confidence_low, 2),
                "high": round(p.confidence_high, 2),
                "samples": p.sample_count,
                "reliability": round(p.reliability, 2),
                "reliability_label": _reliability_label(p.reliability),
                "method": p.method,
                "note": p.note,
                "error_metrics": evaluation.get(str(p.horizon_days), {}),
            }
            for p in predictions
        ],
    }


def persist_prediction_snapshots(
    db: Session,
    predictions: list[PricePrediction],
    model_version: str,
    score_value: float | None,
    score_source: str | None,
    input_summary: dict[str, Any],
    min_interval_hours: int = 20,
) -> str:
    """保存一次预测运行的所有 horizon 快照。"""
    now = utc_now()
    now_naive = now.replace(tzinfo=None)
    cutoff = now_naive - timedelta(hours=max(0, min_interval_hours))
    latest_existing = db.scalar(
        select(GoldPredictionSnapshot)
        .where(
            GoldPredictionSnapshot.model_version == model_version,
            GoldPredictionSnapshot.timestamp >= cutoff,
        )
        .order_by(GoldPredictionSnapshot.timestamp.desc())
    )
    if latest_existing is not None:
        evaluate_due_predictions(db)
        return latest_existing.run_id

    run_id = now.strftime("%Y%m%d%H%M%S") + "_" + uuid4().hex[:8]
    summary_json = json_dumps(input_summary)
    for pred in predictions:
        row = GoldPredictionSnapshot(
            run_id=run_id,
            timestamp=now_naive,
            target_timestamp=(now + timedelta(days=pred.horizon_days)).replace(tzinfo=None),
            horizon_days=pred.horizon_days,
            current_price=pred.current_price,
            predicted_price=pred.predicted_price,
            expected_return_pct=pred.expected_return_pct,
            confidence_low=pred.confidence_low,
            confidence_high=pred.confidence_high,
            reliability=pred.reliability,
            method=pred.method,
            model_version=model_version,
            score_value=score_value,
            score_source=score_source,
            input_summary_json=summary_json,
            note=pred.note,
            source="PREDICTOR",
        )
        db.add(row)
    db.commit()
    # 自动评估已到期预测
    evaluate_due_predictions(db)
    return run_id


def evaluate_due_predictions(db: Session, limit: int = 500) -> dict[str, Any]:
    """评估所有已经到期但尚未比对真实价格的预测快照。"""
    now = utc_now().replace(tzinfo=None)
    existing_eval_ids = {
        int(row[0])
        for row in db.execute(select(GoldPredictionEvaluation.prediction_id)).all()
    }
    due_rows = db.scalars(
        select(GoldPredictionSnapshot)
        .where(GoldPredictionSnapshot.target_timestamp <= now)
        .order_by(GoldPredictionSnapshot.target_timestamp.asc())
        .limit(limit)
    ).all()
    due_rows = [row for row in due_rows if row.id not in existing_eval_ids]

    evaluated = 0
    skipped = 0
    for pred in due_rows:
        actual = db.scalar(
            select(GoldPrice)
            .where(GoldPrice.date >= pred.target_timestamp)
            .order_by(GoldPrice.date.asc())
        )
        if actual is None:
            skipped += 1
            continue

        actual_price = float(actual.close)
        predicted_price = float(pred.predicted_price)
        current_price = float(pred.current_price)
        error_price = predicted_price - actual_price
        abs_error = abs(error_price)
        abs_pct_error = (abs_error / actual_price * 100) if actual_price else None
        actual_return = (actual_price / current_price - 1) * 100 if current_price else 0.0
        predicted_return = float(pred.expected_return_pct)
        direction_hit = _direction_with_deadband(predicted_return) == _direction_with_deadband(actual_return)

        db.add(
            GoldPredictionEvaluation(
                prediction_id=int(pred.id),
                actual_timestamp=actual.date,
                actual_price=actual_price,
                predicted_price=predicted_price,
                error_price=error_price,
                abs_error_price=abs_error,
                abs_pct_error=abs_pct_error,
                predicted_return_pct=predicted_return,
                actual_return_pct=actual_return,
                direction_hit=direction_hit,
                horizon_days=int(pred.horizon_days),
                model_version=pred.model_version,
            )
        )
        evaluated += 1

    if evaluated:
        db.commit()
        refresh_prediction_model_metrics(db)

    return {
        "ok": True,
        "evaluated": evaluated,
        "skipped_no_actual_price": skipped,
        "remaining_due": max(0, len(due_rows) - evaluated - skipped),
    }


def _direction_with_deadband(return_pct: float, deadband_pct: float = 0.25) -> int:
    if abs(return_pct) < deadband_pct:
        return 0
    return 1 if return_pct > 0 else -1


def _aware_utc(value: Any) -> Any:
    if value is None or not hasattr(value, "tzinfo"):
        return value
    if value.tzinfo is None:
        return value.replace(tzinfo=utc_now().tzinfo)
    return value.astimezone(utc_now().tzinfo)


def refresh_prediction_model_metrics(db: Session) -> None:
    """按模型版本回写最新误差指标。"""
    versions = db.scalars(select(PredictionModelVersion)).all()
    for version in versions:
        rows = db.scalars(
            select(GoldPredictionEvaluation).where(
                GoldPredictionEvaluation.model_version == version.version
            )
        ).all()
        if not rows:
            continue
        version.evaluated_count = len(rows)
        version.mae_price = round(float(np.mean([r.abs_error_price for r in rows])), 4)
        pct_values = [r.abs_pct_error for r in rows if r.abs_pct_error is not None]
        version.mape_price_pct = round(float(np.mean(pct_values)), 4) if pct_values else None
        version.direction_accuracy = round(float(np.mean([1.0 if r.direction_hit else 0.0 for r in rows])), 4)
    db.commit()


def prediction_evaluation_summary(db: Session) -> dict[str, Any]:
    """返回分 horizon / 分模型版本的预测验证汇总。"""
    rows = db.scalars(select(GoldPredictionEvaluation)).all()
    pending = db.scalars(select(GoldPredictionSnapshot)).all()
    evaluated_ids = {int(r.prediction_id) for r in rows}
    now = utc_now()
    due_pending = [
        p for p in pending
        if _aware_utc(p.target_timestamp) <= now and int(p.id) not in evaluated_ids
    ]
    future_pending = [
        p for p in pending
        if _aware_utc(p.target_timestamp) > now and int(p.id) not in evaluated_ids
    ]

    if not rows:
        return {
            "ok": True,
            "summary": {
                "evaluated_count": 0,
                "due_pending_count": len(due_pending),
                "future_pending_count": len(future_pending),
            },
            "by_horizon": [],
            "by_model": [],
        }

    df = pd.DataFrame([
        {
            "horizon_days": r.horizon_days,
            "model_version": r.model_version,
            "abs_error_price": r.abs_error_price,
            "abs_pct_error": r.abs_pct_error,
            "direction_hit": bool(r.direction_hit),
        }
        for r in rows
    ])

    def _group_summary(group_cols: list[str]) -> list[dict[str, Any]]:
        grouped = df.groupby(group_cols, dropna=False)
        out = []
        for key, g in grouped:
            if not isinstance(key, tuple):
                key = (key,)
            item = {col: key[i] for i, col in enumerate(group_cols)}
            item.update({
                "count": int(len(g)),
                "mae_price": round(float(g["abs_error_price"].mean()), 4),
                "mape_price_pct": round(float(g["abs_pct_error"].dropna().mean()), 4)
                if not g["abs_pct_error"].dropna().empty else None,
                "direction_accuracy": round(float(g["direction_hit"].mean()), 4),
            })
            out.append(item)
        return out

    return {
        "ok": True,
        "summary": {
            "evaluated_count": int(len(df)),
            "due_pending_count": len(due_pending),
            "future_pending_count": len(future_pending),
            "mae_price": round(float(df["abs_error_price"].mean()), 4),
            "mape_price_pct": round(float(df["abs_pct_error"].dropna().mean()), 4)
            if not df["abs_pct_error"].dropna().empty else None,
            "direction_accuracy": round(float(df["direction_hit"].mean()), 4),
        },
        "by_horizon": sorted(_group_summary(["horizon_days"]), key=lambda x: x["horizon_days"]),
        "by_model": sorted(_group_summary(["model_version"]), key=lambda x: x["model_version"]),
    }


def _sample_prediction_params(n_iter: int, random_seed: int = 42) -> list[dict[str, float]]:
    rng = random.Random(random_seed)
    samples = [dict(DEFAULT_MODEL_PARAMS)]
    for _ in range(max(0, n_iter)):
        candidate = dict(DEFAULT_MODEL_PARAMS)
        for key, values in PREDICTION_PARAM_SPACE.items():
            candidate[key] = float(rng.choice(values))
        samples.append(candidate)
    return samples


def _score_prediction_metrics(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    weighted_mape = 0.0
    weighted_mae = 0.0
    weighted_direction = 0.0
    used_weight = 0.0
    valid_horizons = 0
    total_samples = 0

    for horizon, weight in OPTIMIZATION_HORIZON_WEIGHTS.items():
        row = metrics.get(horizon, {})
        if not row.get("ok"):
            continue
        mape = row.get("mape_price_pct")
        mae = row.get("mae_price")
        direction = row.get("direction_accuracy")
        sample_count = int(row.get("sample_count") or 0)
        if mape is None or mae is None or direction is None or sample_count <= 0:
            continue
        # 小样本 horizon 降权，避免 180/360 天虚高。
        sample_weight = min(1.0, sample_count / 30)
        w = weight * sample_weight
        weighted_mape += float(mape) * w
        weighted_mae += float(mae) * w
        weighted_direction += float(direction) * w
        used_weight += w
        valid_horizons += 1
        total_samples += sample_count

    if used_weight <= 0:
        return {
            "ok": False,
            "optimization_score": None,
            "reason": "no valid horizon metrics",
            "valid_horizons": 0,
            "sample_count": 0,
        }

    avg_mape = weighted_mape / used_weight
    avg_mae = weighted_mae / used_weight
    avg_direction = weighted_direction / used_weight
    # 分数越高越好：方向准确率加分，MAPE 扣分。MAE 用于展示和次级排序。
    optimization_score = avg_direction * 100 - avg_mape * 2
    return {
        "ok": True,
        "optimization_score": round(float(optimization_score), 4),
        "weighted_mape_price_pct": round(float(avg_mape), 4),
        "weighted_mae_price": round(float(avg_mae), 4),
        "weighted_direction_accuracy": round(float(avg_direction), 4),
        "valid_horizons": valid_horizons,
        "sample_count": total_samples,
    }


def optimize_prediction_model_params(
    db: Session,
    n_iter: int = 80,
    top_k: int = 5,
    random_seed: int = 42,
    save_best: bool = True,
    auto_activate: bool = False,
    activation_thresholds: dict[str, float | int] | None = None,
) -> dict[str, Any]:
    """生成候选预测模型参数，并用 walk-forward 多 horizon 指标排序。

    只保存候选版本，不自动激活。后续需调用 /predict/models/{version}/activate。
    """
    context = _prediction_training_context(db)
    if not context.get("ok"):
        return {"ok": False, "reason": context.get("reason", "insufficient data"), "results": []}

    merged = context["merged"]
    gold_idx = context["gold_idx"]
    candidates = _sample_prediction_params(n_iter=n_iter, random_seed=random_seed)
    results: list[dict[str, Any]] = []
    for index, params in enumerate(candidates):
        metrics = _walk_forward_evaluation(merged, gold_idx, params)
        scored = _score_prediction_metrics(metrics)
        label = "baseline" if index == 0 else f"candidate_{index}"
        results.append({
            "label": label,
            "params": params,
            "metrics": metrics,
            **scored,
        })

    def sort_key(item: dict[str, Any]) -> tuple[float, float, float]:
        if not item.get("ok"):
            return (-1e9, -1e9, 1e9)
        return (
            float(item.get("optimization_score") or -1e9),
            float(item.get("weighted_direction_accuracy") or 0),
            -float(item.get("weighted_mape_price_pct") or 1e9),
        )

    results.sort(key=sort_key, reverse=True)
    baseline = next((item for item in results if item["label"] == "baseline"), None)
    best = results[0] if results else None
    saved_version = None
    activation = {
        "auto_activate_requested": bool(auto_activate),
        "activated": False,
        "eligible": False,
        "reasons": ["no saved candidate"],
        "thresholds": activation_thresholds or {},
    }
    if save_best and best and best.get("ok"):
        saved = save_prediction_model_candidate(db, best)
        saved_version = saved.version
        activation = prediction_model_activation_decision(best, activation_thresholds)
        activation["auto_activate_requested"] = bool(auto_activate)
        if auto_activate and activation["eligible"]:
            activate_prediction_model_version(db, saved.version)
            activation["activated"] = True
            activation["activated_version"] = saved.version

    return {
        "ok": bool(best and best.get("ok")),
        "saved_version": saved_version,
        "activation": activation,
        "best": _compact_optimization_result(best) if best else None,
        "baseline": _compact_optimization_result(baseline) if baseline else None,
        "top": [_compact_optimization_result(item) for item in results[:max(1, top_k)]],
        "message": (
            f"Saved candidate model {saved_version}. Activate manually after review."
            if saved_version else "Optimization completed without saving a candidate."
        ),
    }


def prediction_model_activation_decision(
    result: dict[str, Any],
    thresholds: dict[str, float | int] | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or {}
    min_score = float(thresholds.get("min_score", 40.0))
    max_mape = float(thresholds.get("max_mape_price_pct", 8.0))
    min_direction = float(thresholds.get("min_direction_accuracy", 0.52))
    min_samples = int(thresholds.get("min_samples", 120))
    min_valid_horizons = int(thresholds.get("min_valid_horizons", 4))

    checks = [
        ("optimization_score", result.get("optimization_score"), ">=", min_score),
        ("weighted_mape_price_pct", result.get("weighted_mape_price_pct"), "<=", max_mape),
        ("weighted_direction_accuracy", result.get("weighted_direction_accuracy"), ">=", min_direction),
        ("sample_count", result.get("sample_count"), ">=", min_samples),
        ("valid_horizons", result.get("valid_horizons"), ">=", min_valid_horizons),
    ]
    reasons: list[str] = []
    eligible = True
    for name, value, op, threshold in checks:
        if value is None:
            eligible = False
            reasons.append(f"{name} missing")
            continue
        passed = float(value) >= float(threshold) if op == ">=" else float(value) <= float(threshold)
        if not passed:
            eligible = False
            reasons.append(f"{name} {value} not {op} {threshold}")
    if eligible:
        reasons.append("all thresholds passed")
    return {
        "eligible": eligible,
        "activated": False,
        "reasons": reasons,
        "thresholds": {
            "min_score": min_score,
            "max_mape_price_pct": max_mape,
            "min_direction_accuracy": min_direction,
            "min_samples": min_samples,
            "min_valid_horizons": min_valid_horizons,
        },
    }


def activate_prediction_model_version(db: Session, version: str) -> PredictionModelVersion | None:
    target = db.scalar(
        select(PredictionModelVersion).where(PredictionModelVersion.version == version)
    )
    if target is None:
        return None
    rows = db.scalars(select(PredictionModelVersion)).all()
    for row in rows:
        row.is_active = row.version == version
    db.commit()
    db.refresh(target)
    return target


def _compact_optimization_result(item: dict[str, Any] | None) -> dict[str, Any] | None:
    if item is None:
        return None
    return {
        "label": item.get("label"),
        "ok": item.get("ok"),
        "optimization_score": item.get("optimization_score"),
        "weighted_mape_price_pct": item.get("weighted_mape_price_pct"),
        "weighted_mae_price": item.get("weighted_mae_price"),
        "weighted_direction_accuracy": item.get("weighted_direction_accuracy"),
        "valid_horizons": item.get("valid_horizons"),
        "sample_count": item.get("sample_count"),
        "params": item.get("params"),
    }


def save_prediction_model_candidate(db: Session, result: dict[str, Any]) -> PredictionModelVersion:
    from datetime import datetime, timezone

    version = f"predictor_candidate_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    params = result.get("params") or DEFAULT_MODEL_PARAMS
    record = PredictionModelVersion(
        version=version,
        method="multi_signal_ensemble_v2",
        params_json=json_dumps(params),
        is_active=False,
        mae_price=result.get("weighted_mae_price"),
        mape_price_pct=result.get("weighted_mape_price_pct"),
        direction_accuracy=result.get("weighted_direction_accuracy"),
        evaluated_count=int(result.get("sample_count") or 0),
        notes=(
            "候选预测模型，基于多 horizon walk-forward 自动搜索生成；"
            f"score={result.get('optimization_score')}, "
            f"valid_horizons={result.get('valid_horizons')}。默认不自动激活。"
        ),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def _build_return_data(merged: pd.DataFrame, gold_idx: pd.Series) -> dict[int, pd.DataFrame]:
    """为每个期限构建历史收益数据。"""
    ret_data = {}
    for h in HORIZONS:
        returns = []
        for _, row in merged.iterrows():
            ts = row["timestamp"]
            future_ts = ts + timedelta(days=h)
            future = gold_idx[gold_idx.index >= future_ts]
            if future.empty:
                continue
            fp = future.iloc[0]
            entry = row["gold_price"]
            if entry <= 0:
                continue
            returns.append({
                "score": row["total_score"],
                "entry_price": float(entry),
                "future_price": float(fp),
                "return_pct": (fp / entry - 1) * 100,
            })
        ret_data[h] = pd.DataFrame(returns)
    return ret_data


def _reliability_label(value: float) -> str:
    if value >= 0.75:
        return "高"
    if value >= 0.5:
        return "中"
    return "低"


def _walk_forward_evaluation(
    merged: pd.DataFrame,
    gold_idx: pd.Series,
    params: dict[str, float],
) -> dict[str, dict[str, Any]]:
    """用真实历史未来价做分 horizon walk-forward 误差评估。"""
    history = _build_return_data(merged, gold_idx)
    out: dict[str, dict[str, Any]] = {}
    for horizon, df in history.items():
        if len(df) < 80:
            out[str(horizon)] = {
                "ok": False,
                "reason": "样本不足，无法稳定评估",
                "sample_count": int(len(df)),
            }
            continue

        # 用过去窗口内“评分相似样本”的收益均值预测下一条，避免使用未来信息。
        # 评估点按 horizon 拉开，最多 21 天取一个点，降低长期 horizon 的重叠样本虚高。
        actual_returns: list[float] = []
        predicted_returns: list[float] = []
        actual_prices: list[float] = []
        predicted_prices: list[float] = []
        min_train = max(60, int(len(df) * 0.5))
        eval_step = max(1, min(horizon, 21))
        for idx in range(min_train, len(df), eval_step):
            train = df.iloc[:idx]
            row = df.iloc[idx]
            sigma = max(1.0, _param(params, "score_similarity_sigma"))
            weights = np.exp(-((train["score"] - row["score"]) ** 2) / (2 * sigma**2))
            if float(weights.sum()) <= 0:
                pred_ret = float(train["return_pct"].mean())
            else:
                pred_ret = float(np.average(train["return_pct"], weights=weights))
            predicted_returns.append(pred_ret)
            actual_returns.append(float(row["return_pct"]))
            entry_price = float(row["entry_price"])
            predicted_prices.append(entry_price * (1 + pred_ret / 100))
            actual_prices.append(float(row["future_price"]))

        pred = np.array(predicted_returns)
        actual = np.array(actual_returns)
        pred_price = np.array(predicted_prices)
        actual_price = np.array(actual_prices)
        errors = pred - actual
        price_errors = pred_price - actual_price
        nonzero_mask = np.abs(actual) > 1e-9
        price_nonzero_mask = np.abs(actual_price) > 1e-9
        actual_direction = np.where(np.abs(actual) < 0.25, 0, np.sign(actual))
        predicted_direction = np.where(np.abs(pred) < 0.25, 0, np.sign(pred))
        direction_accuracy = float((predicted_direction == actual_direction).mean()) if len(actual) else 0.0
        out[str(horizon)] = {
            "ok": True,
            "evaluation_method": f"walk_forward_step_{eval_step}d_with_0.25pct_deadband",
            "sample_count": int(len(actual)),
            "mae_price": round(float(np.mean(np.abs(price_errors))), 4),
            "mape_price_pct": round(float(np.mean(np.abs(price_errors[price_nonzero_mask] / actual_price[price_nonzero_mask])) * 100), 4)
            if price_nonzero_mask.any()
            else None,
            "mae_return_pct": round(float(np.mean(np.abs(errors))), 4),
            "mape_return_pct": round(float(np.mean(np.abs(errors[nonzero_mask] / actual[nonzero_mask])) * 100), 4)
            if nonzero_mask.any()
            else None,
            "direction_accuracy": round(direction_accuracy, 4),
        }
    return out


def _short_term_predict(
    horizon: int,
    current_price: float,
    current_score: float,
    merged: pd.DataFrame,
    gold_idx: pd.Series,
    ret_data: dict[int, pd.DataFrame],
    params: dict[str, float],
) -> PricePrediction:
    """短期预测：动量 + 评分方向 → 加权组合。"""
    df = ret_data.get(horizon, pd.DataFrame())
    n = len(df)

    # 1) 动量信号：最近 horizon 天的价格变化
    recent_prices = merged["gold_price"].tail(max(horizon, 5))
    if len(recent_prices) >= 2:
        momentum = (recent_prices.iloc[-1] / recent_prices.iloc[0] - 1) * 100
        price_start = float(recent_prices.iloc[0])
        price_end = float(recent_prices.iloc[-1])
    else:
        momentum = 0
        price_start = price_end = current_price

    # 2) 评分信号：score_similarity 加权
    if n >= 10:
        sigma = max(1.0, _param(params, "score_similarity_sigma"))
        weights = np.exp(-((df["score"] - current_score) ** 2) / (2 * sigma**2))
        expected_ret = float(np.average(df["return_pct"], weights=weights))
        std_ret = float(df["return_pct"].std())
    else:
        expected_ret = 0.0
        std_ret = 5.0

    # 3) 信号组合：短期动量 + 评分回归
    momentum_weight = _param(params, "short_momentum_weight")
    score_weight = _param(params, "short_score_weight")
    weight_sum = max(1e-9, abs(momentum_weight) + abs(score_weight))
    momentum_weight = momentum_weight / weight_sum
    score_weight = score_weight / weight_sum
    momentum_contrib = momentum * momentum_weight
    score_contrib = expected_ret * score_weight
    combined_ret = momentum_contrib + score_contrib
    clip_pct = abs(_param(params, "short_return_clip_pct"))
    combined_ret = float(np.clip(combined_ret, -clip_pct, clip_pct))

    std_ret = max(std_ret, 3.0)
    predicted = current_price * (1 + combined_ret / 100)
    interval_mult = max(0.1, _param(params, "interval_std_multiplier"))
    low = current_price * (1 + (combined_ret - interval_mult * std_ret) / 100)
    high = current_price * (1 + (combined_ret + interval_mult * std_ret) / 100)

    reliability = min(1.0, n / MIN_RETURN_SAMPLES)

    if horizon <= 7:
        theory = (
            "● 预测理论: 短期主要看价格动量、资金惯性和最新多空评分；"
            "宏观变量通常通过情绪和仓位间接影响。"
        )
    else:
        theory = (
            "● 预测理论: 30天属于中短期过渡，价格动量仍重要，"
            "但会加入历史相似评分环境的均值回归。"
        )

    detail = (
        f"{theory}\n"
        f"● 价格动量({horizon}日): {price_start:,.0f}→{price_end:,.0f} = {momentum:+.1f}% × {momentum_weight:.0%} = {momentum_contrib:+.1f}%\n"
        f"● 评分回归: 当前评分{current_score:+.0f}，历史相似样本均值{expected_ret:+.1f}% × {score_weight:.0%} = {score_contrib:+.1f}%\n"
        f"● 模型参数: score_sigma={max(1.0, _param(params, 'score_similarity_sigma')):.1f}，收益截断±{clip_pct:.0f}%\n"
        f"● 历史波动率(std): {std_ret:.1f}%\n"
        f"● 历史样本: {n} 条\n"
    )

    return PricePrediction(
        horizon_days=horizon,
        current_price=current_price,
        predicted_price=predicted,
        expected_return_pct=combined_ret,
        confidence_low=low,
        confidence_high=high,
        sample_count=n,
        reliability=reliability,
        method="momentum+score",
        note=detail,
    )


def _long_term_predict(
    horizon: int,
    current_price: float,
    macro: dict[str, float],
    cftc_net: float | None,
    merged: pd.DataFrame,
    gold_idx: pd.Series,
    ret_data: dict[int, pd.DataFrame],
    params: dict[str, float],
) -> PricePrediction:
    """长期预测：宏观环境 + 持仓结构 → 分组均值 + 均值回归。"""
    df = ret_data.get(horizon, pd.DataFrame())
    n = len(df)

    if horizon <= 90:
        theory = (
            "● 预测理论: 中期主要看实际利率、名义利率、美元、VIX和CFTC仓位；"
            "这些变量决定黄金的机会成本、避险需求和投机资金方向。"
        )
    else:
        theory = (
            "● 预测理论: 长期主要看实际利率中枢、通胀环境、央行/机构配置和持仓结构；"
            "短期噪声权重降低，预测区间会明显变宽。"
        )

    lines = [theory]
    macro_score = 0.0
    nominal = macro.get(NOMINAL_RATE, 4.5)
    real = macro.get(REAL_RATE, 2.0)
    vix = macro.get(VIX, 20)

    if nominal < 4.0:
        macro_score += 0.3
        lines.append(f"● 10Y名义利率 {nominal:.1f}% < 4.0% → 低利率利多 +0.3")
    elif nominal > 5.0:
        macro_score -= 0.3
        lines.append(f"● 10Y名义利率 {nominal:.1f}% > 5.0% → 高利率利空 -0.3")
    else:
        lines.append(f"● 10Y名义利率 {nominal:.1f}%（中性）")

    if real > 2.0:
        macro_score -= 0.2
        lines.append(f"● 实际利率 {real:.1f}% > 2.0% → 高实际利率利空 -0.2")
    else:
        lines.append(f"● 实际利率 {real:.1f}%（中性）")

    if vix > 25:
        macro_score += 0.2
        lines.append(f"● VIX {vix:.0f} > 25 → 恐慌利多 +0.2")
    else:
        lines.append(f"● VIX {vix:.0f}（中性）")

    if cftc_net is not None:
        if cftc_net > 0.5:
            macro_score += 0.1
            lines.append(f"● CFTC 净多占比 {cftc_net:.0%} > 50% → 投机偏多 +0.1")
        elif cftc_net < 0.2:
            macro_score -= 0.1
            lines.append(f"● CFTC 净多占比 {cftc_net:.0%} < 20% → 投机偏空 -0.1")
        else:
            lines.append(f"● CFTC 净多占比 {cftc_net:.0%}（中性）")

    # 计算预期收益
    annual_base = _param(params, "long_annual_base_return_pct")
    base_ret = annual_base * (horizon / 365)
    macro_scale = _param(params, "long_macro_adjustment_scale")
    macro_adj = macro_score * macro_scale * (horizon / 90)
    expected_ret = base_ret + macro_adj
    expected_ret = float(np.clip(
        expected_ret,
        _param(params, "long_return_clip_low_pct"),
        _param(params, "long_return_clip_high_pct"),
    ))

    if n >= 10:
        std_ret = max(float(df["return_pct"].std()), 5.0)
    else:
        std_ret = horizon * 0.15

    predicted = current_price * (1 + expected_ret / 100)
    interval_mult = max(0.1, _param(params, "interval_std_multiplier"))
    low = current_price * (1 + (expected_ret - interval_mult * std_ret) / 100)
    high = current_price * (1 + (expected_ret + interval_mult * std_ret) / 100)

    reliability = min(0.9, n / MIN_RETURN_SAMPLES) * max(0.3, 1 - horizon / 720)

    lines.insert(1, f"● 年化基准收益率: {annual_base:.0f}% × {horizon}/365 → {base_ret:+.1f}%")
    lines.append(f"● 宏观调整合计: macro_score {macro_score:+.1f} × scale {macro_scale:.1f} → {macro_adj:+.1f}%")
    lines.append(
        f"● 模型参数: 长期截断 {params['long_return_clip_low_pct']:.0f}% 到 {params['long_return_clip_high_pct']:.0f}%，"
        f"区间倍数 {interval_mult:.1f}σ"
    )
    lines.append(f"● 历史波动率(std): {std_ret:.1f}%")
    lines.append(f"● 历史样本: {n} 条")
    detail = "\n".join(lines)

    return PricePrediction(
        horizon_days=horizon,
        current_price=current_price,
        predicted_price=predicted,
        expected_return_pct=expected_ret,
        confidence_low=low,
        confidence_high=high,
        sample_count=n,
        reliability=reliability,
        method="macro_regime",
        note=detail,
    )
