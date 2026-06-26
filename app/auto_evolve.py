"""自动进化引擎：闭环监控预测表现 → 自动搜索更优参数 → 达标自动激活。

每个采集周期结束前自动运行：
1. 检查激活模型的近期表现（方向准确率、MAPE）
2. 若退化超过阈值，自动搜索更优评分参数
3. 若候选显著优于当前，自动激活
4. 同时检查预测模型参数是否需要进化
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select

from app.auto_settings import resolved_auto_settings
from app.models import (
    GoldPredictionEvaluation,
    ModelActivationAudit,
    PredictionModelVersion,
    ScoreParamsVersion,
)
from app.scoring.score_optimizer import (
    activate_version,
    optimize_score_params,
    save_best_params,
)
from app.scoring.gold_predictor import (
    _active_prediction_model,
    activate_prediction_model_version,
    optimize_prediction_model_params,
)

logger = logging.getLogger(__name__)

# ── 进化阈值 ─────────────────────────────────────────────────────
MIN_DIRECTION_ACCURACY = 0.48          # 低于此触发进化
MIN_ACTIVATION_HIT_RATE = 0.50         # 候选命中率高于此才考虑激活
MIN_SAMPLE_COUNT = 100                  # 至少这么多评估样本
ACTIVATION_LIFT = 0.03                  # 候选需比当前高 3% 才自动激活
MAX_MAPE_DEGRADATION = 0.5             # 激活候选时 MAPE 恶化不超过 0.5%
COOLDOWN_HOURS = 12                    # 同一类型进化冷却期
RECENT_WINDOW_DAYS = 7                 # 近期表现评估窗口
MAX_RECENT_DEGRADATION = 0.05          # 近期退化超过5%，触发进化
SCORE_N_ITER = 15                       # 评分参数搜索迭代（快速版）
SCORE_HORIZON_DAYS = 20                # 评分回测窗口
PRED_N_ITER = 30                       # 预测模型搜索迭代


def _utc_aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def auto_evolve_if_needed(db, *, force: bool = False, settings: dict[str, Any] | None = None) -> dict | None:
    """自动进化主入口：评分参数 + 预测模型参数。

    在每个 collect_and_score_job 周期结束时调用。
    返回进化结果摘要（如有），否则 None。
    """
    results = {}

    settings = settings or resolved_auto_settings(db)
    if not settings.get("AUTO_SELF_HEALING_ENABLED", True) and not force:
        return {"skipped": True, "reason": "self healing disabled"}

    # 1) 评分参数进化
    score_result = _evolve_score_params(db, force=force, settings=settings)
    if score_result:
        results["score"] = score_result

    # 2) 预测模型参数进化
    pred_result = _evolve_prediction_model(db, force=force, settings=settings)
    if pred_result:
        results["prediction"] = pred_result

    return results if results else None


def _check_cooldown(db, prefix: str, hours: int) -> bool:
    """检查是否在冷却期内。"""
    recent = db.scalar(
        select(ScoreParamsVersion)
        .where(ScoreParamsVersion.version.like(f"{prefix}%"))
        .order_by(ScoreParamsVersion.created_at.desc())
    )
    if recent and recent.created_at:
        age = (datetime.now(timezone.utc) - _utc_aware(recent.created_at)).total_seconds() / 3600
        if age < hours:
            return True
    return False


def _check_audit_cooldown(db, model_type: str, hours: int) -> bool:
    recent = db.scalar(
        select(ModelActivationAudit)
        .where(ModelActivationAudit.model_type == model_type)
        .order_by(ModelActivationAudit.created_at.desc())
    )
    if recent and recent.created_at:
        age = (datetime.now(timezone.utc) - _utc_aware(recent.created_at)).total_seconds() / 3600
        return age < hours
    return False


def _evolve_score_params(db, *, force: bool = False, settings: dict[str, Any] | None = None) -> dict | None:
    """评估当前评分参数表现，必要时搜索并可能激活更优参数。"""
    # 冷却检查
    if not force and _check_cooldown(db, "auto_evolve_score_", COOLDOWN_HOURS):
        return None

    # 从激活模型的评估数据中获取近期表现
    active_model = _active_prediction_model(db)
    if not active_model or not active_model.evaluated_count:
        return None

    samples = active_model.evaluated_count
    accuracy = active_model.direction_accuracy or 0
    mape = active_model.mape_price_pct or 99

    min_samples = int((settings or {}).get("AUTO_PREDICTION_MIN_SAMPLES") or MIN_SAMPLE_COUNT)
    if samples < min_samples:
        logger.debug("自动进化: 样本不足 %d < %d", samples, MIN_SAMPLE_COUNT)
        return None

    # 判断是否需要进化
    need_evolve = False
    reason = ""

    if accuracy < MIN_DIRECTION_ACCURACY:
        need_evolve = True
        reason = f"方向准确率 {accuracy:.1%} < {MIN_DIRECTION_ACCURACY:.1%}"

    # 也检查近期退化（评估表中有 model_version 匹配的数据）
    cutoff = datetime.now(timezone.utc) - timedelta(days=RECENT_WINDOW_DAYS)
    recent_evals = db.scalars(
        select(GoldPredictionEvaluation)
        .where(
            GoldPredictionEvaluation.evaluated_at >= cutoff,
            GoldPredictionEvaluation.model_version == active_model.version,
        )
    ).all()
    if len(recent_evals) >= 10:
        recent_acc = sum(1 for e in recent_evals if e.direction_hit) / len(recent_evals)
        if accuracy - recent_acc > MAX_RECENT_DEGRADATION:
            need_evolve = True
            reason = f"近期退化 {accuracy - recent_acc:.1%} > {MAX_RECENT_DEGRADATION:.1%} (整体{accuracy:.1%} vs 近期{recent_acc:.1%})"

    if force:
        need_evolve = True
        reason = reason or "强制无人值守自检触发"

    if not need_evolve:
        return None

    logger.info("自动进化(评分): %s，启动搜索", reason)

    # 运行参数搜索
    n_iter = int((settings or {}).get("AUTO_OPTIMIZE_N_ITER") or SCORE_N_ITER)
    horizon_days = int((settings or {}).get("AUTO_OPTIMIZE_HORIZON_DAYS") or SCORE_HORIZON_DAYS)
    results = optimize_score_params(db, n_iter=max(1, min(n_iter, 300)), horizon_days=horizon_days)
    if not results or not results[0].get("ok"):
        logger.warning("自动进化(评分): 搜索未找到有效结果")
        return None

    best = results[0]
    hit_rate = best.get("hit_rate")
    sample_count = best.get("sample_count")

    if hit_rate is None or sample_count is None:
        return None

    # 质量门控
    min_hit_rate = float((settings or {}).get("AUTO_OPTIMIZE_MIN_HIT_RATE") or MIN_ACTIVATION_HIT_RATE)
    if float(hit_rate) < min_hit_rate:
        logger.info("自动进化(评分): 最优命中率 %.1f%% < %.1f%%，不激活",
                    float(hit_rate) * 100, min_hit_rate * 100)
        return {"trigger": reason, "hit_rate": float(hit_rate), "activated": False,
                "reason": "命中率不达标"}

    if int(sample_count) < min_samples:
        logger.info("自动进化(评分): 样本 %d < %d", int(sample_count), MIN_SAMPLE_COUNT)
        return {"trigger": reason, "hit_rate": float(hit_rate), "activated": False,
                "reason": "样本不足"}

    # 判断是否值得激活
    activation_check = best.get("activation_check") or {}
    should_activate = (
        bool((settings or {}).get("AUTO_SELF_HEALING_AUTOFIX", True))
        and activation_check.get("eligible")
        and float(hit_rate) > accuracy + ACTIVATION_LIFT
    )

    version = f"auto_evolve_score_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}"
    saved = save_best_params(
        db, best,
        version=version,
        horizon_days=horizon_days,
        notes=(
            f"自动进化: {reason}。搜索{SCORE_N_ITER}组 → "
            f"命中率{float(hit_rate):.1%}（当前{accuracy:.1%}）。"
            f"{'已自动激活' if should_activate else '已保存为候选'}。"
        ),
    )

    if should_activate:
        activate_version(db, version)
        logger.info("自动进化(评分): 激活 %s (%.1f%% → %.1f%%)", version, accuracy * 100, float(hit_rate) * 100)
    else:
        logger.info("自动进化(评分): 候选 %s 已保存 (lift=%.1f%% < %.1f%%)",
                    version, (float(hit_rate) - accuracy) * 100, ACTIVATION_LIFT * 100)

    return {
        "trigger": reason,
        "version": version,
        "hit_rate": float(hit_rate),
        "activated": should_activate,
        "previous": accuracy,
        "lift": float(hit_rate) - accuracy,
    }


def _evolve_prediction_model(db, *, force: bool = False, settings: dict[str, Any] | None = None) -> dict | None:
    """评估预测模型表现，必要时搜索更优预测参数。"""
    if not force and _check_audit_cooldown(db, "prediction", COOLDOWN_HOURS * 2):
        return None

    active_model = _active_prediction_model(db)
    if not active_model or not active_model.evaluated_count:
        return None

    samples = active_model.evaluated_count
    accuracy = active_model.direction_accuracy or 0

    min_samples = int((settings or {}).get("AUTO_PREDICTION_MIN_SAMPLES") or MIN_SAMPLE_COUNT)
    if samples < min_samples:
        return None

    if accuracy >= MIN_DIRECTION_ACCURACY + 0.02 and not force:
        return None  # 表现尚可

    logger.info("自动进化(预测): 方向准确率 %.1f%%，启动搜索", accuracy * 100)

    try:
        result = optimize_prediction_model_params(
            db,
            n_iter=int((settings or {}).get("AUTO_PREDICTION_N_ITER") or PRED_N_ITER),
            top_k=3,
            random_seed=None,
            save_best=True,
            auto_activate=bool((settings or {}).get("AUTO_SELF_HEALING_AUTOFIX", True)),
            activation_thresholds={
                "min_score": float((settings or {}).get("AUTO_PREDICTION_MIN_SCORE") or 40.0),
                "max_mape_price_pct": float((settings or {}).get("AUTO_PREDICTION_MAX_MAPE_PCT") or 8.0),
                "min_direction_accuracy": max(float((settings or {}).get("AUTO_PREDICTION_MIN_DIRECTION_ACCURACY") or 0.52), accuracy + 0.03),
                "min_samples": min_samples,
                "min_valid_horizons": int((settings or {}).get("AUTO_PREDICTION_MIN_VALID_HORIZONS") or 3),
                "min_baseline_lift": 0.02,
                "max_mape_worse_ratio": 1.15,
                "max_recent_degradation": 0.05,
            },
        )
        return {"trigger": f"accuracy {accuracy:.1%}", "searched": True, "result": result}
    except Exception as e:
        logger.warning("自动进化(预测): 搜索失败: %s", e)
        return None
