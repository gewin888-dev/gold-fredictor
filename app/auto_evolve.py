"""自动进化桥接：评估预测表现 → 触发参数搜索 → 保存候选。

用法：
    from app.auto_evolve import auto_evolve_if_needed
    auto_evolve_if_needed(db)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from app.auto_settings import resolved_auto_settings
from app.models import GoldPredictionEvaluation, ScoreParamsVersion
from app.scoring.score_optimizer import (
    optimize_score_params,
    save_best_params,
)

logger = logging.getLogger(__name__)

# 触发阈值：近期方向准确率低于此值，自动搜索更优参数
MIN_DIRECTION_ACCURACY = 0.52
# 激活门槛：搜索结果的方向命中率高于此值才自动激活
MIN_ACTIVATION_HIT_RATE = 0.58
# 激活门槛：搜索结果至少需要这么多样本
MIN_SAMPLE_COUNT = 80
# 冷却期：两次自动进化至少间隔 N 小时
COOLDOWN_HOURS = 24


def auto_evolve_if_needed(db) -> dict | None:
    """评估近期预测表现，必要时生成评分参数候选；不会自动激活。"""
    settings = resolved_auto_settings(db)
    if not settings["AUTO_OPTIMIZE_SCORE_PARAMS"]:
        return None

    # 冷却检查：24 小时内已经进化过则跳过
    recent = db.scalar(
        select(ScoreParamsVersion)
        .where(ScoreParamsVersion.notes.like("%自动进化触发%"))
        .order_by(ScoreParamsVersion.created_at.desc())
    )
    if recent and recent.created_at:
        age = (datetime.now(timezone.utc) - recent.created_at).total_seconds() / 3600
        if age < COOLDOWN_HOURS:
            return None

    # 评估近期预测方向准确率
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    evals = db.scalars(
        select(GoldPredictionEvaluation)
        .where(GoldPredictionEvaluation.evaluated_at >= cutoff)
        .order_by(GoldPredictionEvaluation.evaluated_at.desc())
    ).all()

    if len(evals) < 30:
        return None  # 样本不足

    hits = sum(1 for e in evals if e.direction_hit)
    accuracy = hits / len(evals)

    if accuracy >= MIN_DIRECTION_ACCURACY:
        return None  # 表现尚可，不需要进化

    logger.info("自动进化触发：近期方向准确率 %.1f%% < %.1f%%，启动参数搜索",
                accuracy * 100, MIN_DIRECTION_ACCURACY * 100)

    # 运行参数搜索
    results = optimize_score_params(
        db,
        n_iter=int(settings["AUTO_OPTIMIZE_N_ITER"]),
        horizon_days=int(settings["AUTO_OPTIMIZE_HORIZON_DAYS"]),
    )
    if not results or not results[0].get("ok"):
        logger.warning("自动进化：参数搜索未找到有效结果")
        return None

    best = results[0]
    hit_rate = best.get("hit_rate")
    sample_count = best.get("sample_count")

    if hit_rate is None or sample_count is None:
        return None

    # 质量门控
    if float(hit_rate) < MIN_ACTIVATION_HIT_RATE:
        logger.info("自动进化：最优命中率 %.1f%% 未达激活门槛 %.1f%%",
                    float(hit_rate) * 100, MIN_ACTIVATION_HIT_RATE * 100)
        return None
    if int(sample_count) < MIN_SAMPLE_COUNT:
        logger.info("自动进化：样本数 %d 不足，需要 ≥ %d", int(sample_count), MIN_SAMPLE_COUNT)
        return None

    # 保存并激活
    version = f"auto_evolve_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}"
    saved = save_best_params(
        db,
        best,
        version=version,
        horizon_days=int(settings["AUTO_OPTIMIZE_HORIZON_DAYS"]),
        notes=(
            f"自动进化触发（近期预测准确率{accuracy:.1%}）→ "
            f"搜索{int(settings['AUTO_OPTIMIZE_N_ITER'])}组 → 命中率{float(hit_rate):.1%}。"
            "已保存为候选，需人工审核后激活。"
        ),
    )

    logger.info("自动进化完成：版本 %s 已保存为候选，命中率 %.1f%%", version, float(hit_rate) * 100)
    return {
        "trigger": f"accuracy {accuracy:.1%} < {MIN_DIRECTION_ACCURACY:.1%}",
        "version": version,
        "hit_rate": float(hit_rate),
        "sample_count": int(sample_count),
        "activated": False,
        "reason": "候选已生成；正式交付默认不自动激活，需人工审核。",
    }
