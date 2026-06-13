"""评分模型自我优化：参数搜索 + 滚动回测，自动找到最优权重组合。

用法：
    from app.scoring.score_optimizer import optimize_score_params, ScoreParams

    params, hit_rate = optimize_score_params(db, n_iter=200)
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.models import (
    CftcPosition,
    GoldPrice,
    GoldScoreSnapshot,
    MacroObservation,
    ScoreParamsVersion,
)
from app.scoring.gold_score import (
    DOLLAR,
    INFLATION_EXPECTATION,
    NOMINAL_RATE,
    FED_RATE,
    REAL_RATE,
    VIX,
    _cftc_position_score,
    _clamp,
    _series_frame,
)


# ── 可调参数定义 ────────────────────────────────────────────────────


@dataclass
class ScoreParams:
    """评分模型所有可调参数及其默认值。"""

    # 因子灵敏度系数（越大该因子越敏感）
    real_rate_coef: float = 30.0
    nominal_rate_coef: float = 40.0
    dollar_coef: float = 4.0
    vix_coef: float = 1.2
    inflation_coef: float = 25.0
    cftc_coef: float = 30.0
    fed_rate_coef: float = 20.0

    # 各因子截断范围
    real_rate_clamp_low: float = -25.0
    real_rate_clamp_high: float = 25.0
    dollar_clamp_low: float = -20.0
    dollar_clamp_high: float = 20.0
    vix_clamp_low: float = -15.0
    vix_clamp_high: float = 15.0
    inflation_clamp_low: float = -15.0
    inflation_clamp_high: float = 15.0
    cftc_clamp_low: float = -15.0
    cftc_clamp_high: float = 15.0

    # 趋势均线窗口
    trend_ma_short: int = 20
    trend_ma_long: int = 60

    # 趋势得分权重
    trend_ma_short_weight: float = 8.0
    trend_ma_long_weight: float = 12.0

    # 方向判定阈值
    bullish_threshold: float = 30.0
    bearish_threshold: float = -30.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ScoreParams":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    @classmethod
    def defaults(cls) -> "ScoreParams":
        """返回默认参数（即 'rule_v1' 版本）。"""
        return cls()


# ── 参数搜索空间 ────────────────────────────────────────────────────


PARAM_SPACE: dict[str, list[float | int]] = {
    "real_rate_coef":        [10, 15, 20, 25, 30, 35, 40, 50],
    "nominal_rate_coef":     [20, 30, 40, 50, 60],
    "dollar_coef":           [2, 3, 4, 5, 6, 8],
    "vix_coef":              [0.5, 1.0, 1.5, 2.0, 2.5],
    "inflation_coef":        [10, 15, 20, 25, 30, 35],
    "cftc_coef":             [15, 20, 25, 30, 35, 40],
    "nominal_rate_coef":     [20, 30, 40, 50, 60],
    "fed_rate_coef":         [10, 15, 20, 25, 30],
    "trend_ma_short_weight": [4, 6, 8, 10, 12, 14],
    "trend_ma_long_weight":  [6, 8, 10, 12, 14, 16, 18],
    "bullish_threshold":     [15, 20, 25, 30, 35, 40],
    "bearish_threshold":     [-40, -35, -30, -25, -20, -15],
}

N_ITER_DEFAULT = 150


# ── 参数化评分 ───────────────────────────────────────────────────────


def compute_score_at_date(
    db: Session,
    target_date: pd.Timestamp,
    merged: pd.DataFrame,
    params: ScoreParams,
    cftc_cache: dict[pd.Timestamp, tuple[float | None, str | None]] | None = None,
) -> float:
    """用指定参数计算某个日期的多空评分（不写库）。

    merged 必须是按 timestamp 排序的历史对齐 DataFrame。
    返回 total_score float。
    """
    mask = merged["timestamp"] <= target_date
    if mask.sum() < params.trend_ma_long + 5:
        return 0.0  # 数据不足

    hist = merged[mask]
    latest = hist.iloc[-1]

    factor_sum = 0.0

    # 实际利率
    if REAL_RATE in hist.columns and len(hist) > params.trend_ma_short:
        rr_clean = hist[[REAL_RATE]].dropna()
        if len(rr_clean) > params.trend_ma_short:
            real_rate_change = float(
                rr_clean[REAL_RATE].iloc[-1] - rr_clean[REAL_RATE].iloc[-1 - params.trend_ma_short]
            )
            factor_sum += _clamp(
                -real_rate_change * params.real_rate_coef,
                params.real_rate_clamp_low,
                params.real_rate_clamp_high,
            )

    # 美元指数
    if DOLLAR in hist.columns and len(hist) > params.trend_ma_short:
        d_clean = hist[[DOLLAR]].dropna()
        if len(d_clean) > params.trend_ma_short:
            dollar_change_pct = (
                d_clean[DOLLAR].iloc[-1] / d_clean[DOLLAR].iloc[-1 - params.trend_ma_short] - 1
            ) * 100
            factor_sum += _clamp(
                -dollar_change_pct * params.dollar_coef,
                params.dollar_clamp_low,
                params.dollar_clamp_high,
            )

    # 避险情绪
    if VIX in hist.columns and len(hist) > params.trend_ma_short:
        v_clean = hist[[VIX]].dropna()
        if len(v_clean) > params.trend_ma_short:
            vix_change = float(
                v_clean[VIX].iloc[-1] - v_clean[VIX].iloc[-1 - params.trend_ma_short]
            )
            factor_sum += _clamp(
                vix_change * params.vix_coef,
                params.vix_clamp_low,
                params.vix_clamp_high,
            )

    # 通胀预期
    if INFLATION_EXPECTATION in hist.columns and len(hist) > params.trend_ma_short:
        ie_clean = hist[[INFLATION_EXPECTATION]].dropna()
        if len(ie_clean) > params.trend_ma_short:
            inflation_change = float(
                ie_clean[INFLATION_EXPECTATION].iloc[-1]
                - ie_clean[INFLATION_EXPECTATION].iloc[-1 - params.trend_ma_short]
            )
            factor_sum += _clamp(
                inflation_change * params.inflation_coef,
                params.inflation_clamp_low,
                params.inflation_clamp_high,
            )

    # 名义利率
    if NOMINAL_RATE in hist.columns and len(hist) > params.trend_ma_short:
        nr_clean = hist[[NOMINAL_RATE]].dropna()
        if len(nr_clean) > params.trend_ma_short:
            nominal_change = float(
                nr_clean[NOMINAL_RATE].iloc[-1] - nr_clean[NOMINAL_RATE].iloc[-1 - params.trend_ma_short]
            )
            factor_sum += _clamp(
                -nominal_change * params.nominal_rate_coef,
                -20, 20,
            )

    # 联邦基金利率
    if FED_RATE in hist.columns and len(hist) > 3:
        fed_clean = hist[[FED_RATE]].dropna()
        if len(fed_clean) > 3:
            fed_change = float(
                fed_clean[FED_RATE].iloc[-1] - fed_clean[FED_RATE].iloc[-1 - 3]
            )
            factor_sum += _clamp(
                -fed_change * params.fed_rate_coef,
                -15, 15,
            )

    # 黄金趋势
    if "gold_price" in hist.columns and len(hist) >= params.trend_ma_long:
        gold_col = hist["gold_price"].dropna()
        if len(gold_col) >= params.trend_ma_long:
            gold_ma_short = gold_col.tail(params.trend_ma_short).mean()
            gold_ma_long = gold_col.tail(params.trend_ma_long).mean()
            latest_gold = gold_col.iloc[-1]
            trend = 0.0
            if latest_gold > gold_ma_short:
                trend += params.trend_ma_short_weight
            else:
                trend -= params.trend_ma_short_weight
            if gold_ma_short > gold_ma_long:
                trend += params.trend_ma_long_weight
            else:
                trend -= params.trend_ma_long_weight
            factor_sum += trend

    # CFTC 持仓
    if cftc_cache is not None:
        key = target_date.normalize()
        cftc_score, _ = cftc_cache.get(key, (None, None))
        if cftc_score is not None:
            factor_sum += cftc_score

    return round(float(_clamp(factor_sum, -100, 100)), 2)


def _direction_from_params(score: float, params: ScoreParams) -> int:
    """返回方向信号：1=看多，-1=看空，0=中性。"""
    if score >= params.bullish_threshold:
        return 1
    if score <= params.bearish_threshold:
        return -1
    return 0


# ── 参数回测评估 ───────────────────────────────────────────────────


def evaluate_params(
    db: Session,
    params: ScoreParams,
    horizon_days: int = 20,
    min_samples: int = 10,
) -> dict[str, Any]:
    """评估一组参数在历史数据上的方向命中率。

    返回 {"ok": bool, "hit_rate": float|None, "sample_count": int, "signal_ratio": float}
    """
    # 加载金价和宏观数据
    gold_rows = db.scalars(select(GoldPrice).order_by(GoldPrice.date.asc())).all()
    if not gold_rows:
        return {"ok": False, "hit_rate": None, "sample_count": 0, "reason": "no gold price data"}
    gold_df = pd.DataFrame(
        [{"timestamp": pd.Timestamp(r.date).normalize(), "gold_price": r.close} for r in gold_rows]
    )
    gold_df["timestamp"] = pd.to_datetime(gold_df["timestamp"])
    gold_df = gold_df.sort_values("timestamp").set_index("timestamp")

    # 加载所有评分所需序列并 merge
    series_ids = [REAL_RATE, INFLATION_EXPECTATION, VIX, DOLLAR, NOMINAL_RATE, FED_RATE]
    frames: dict[str, pd.DataFrame] = {}
    for sid in series_ids:
        df = _series_frame(db, sid)
        if df.empty:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        frames[sid] = df.sort_values("timestamp")

    # 用第一个 FRED 序列作为基准 merge
    first_sid = series_ids[0]
    if first_sid not in frames:
        return {"ok": False, "hit_rate": None, "sample_count": 0, "reason": "no FRED data"}

    merged = frames[first_sid]
    for sid in series_ids[1:]:
        if sid in frames:
            merged = pd.merge_asof(
                merged.sort_values("timestamp"),
                frames[sid].sort_values("timestamp"),
                on="timestamp",
                direction="backward",
            )

    # 合并金价（标准化到午夜）
    gold_for_merge = gold_df.reset_index().copy()
    gold_for_merge["timestamp"] = pd.to_datetime(gold_for_merge["timestamp"])
    merged = pd.merge_asof(
        merged.sort_values("timestamp"),
        gold_for_merge.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )

    # CFTC 缓存
    cftc_cache: dict[pd.Timestamp, tuple[float | None, str | None]] = {}
    cftc_rows = db.scalars(select(CftcPosition).order_by(CftcPosition.timestamp.asc())).all()
    for row in cftc_rows:
        if row.open_interest <= 0:
            continue
        net_ratio = row.noncommercial_net / row.open_interest
        score = round(_clamp(net_ratio * params.cftc_coef, params.cftc_clamp_low, params.cftc_clamp_high), 2)
        cftc_cache[pd.Timestamp(row.timestamp).normalize()] = (score, None)

    # 可用评分日期
    min_required = params.trend_ma_long + 20
    all_dates = merged["timestamp"].dt.tz_localize(None).sort_values().tolist()
    eligible_dates = all_dates[min_required:]

    if not eligible_dates:
        return {"ok": False, "hit_rate": None, "sample_count": 0, "reason": "not enough history"}

    # 逐日评分
    gold_prices = gold_df["gold_price"].to_dict()
    records = []
    for dt in eligible_dates:
        score = compute_score_at_date(db, dt, merged.copy(), params, cftc_cache)
        gold_at_dt = gold_prices.get(dt)
        if gold_at_dt is None:
            continue
        # 找 horizon_days 后的金价
        future_dt = dt + timedelta(days=horizon_days)
        gold_future = None
        for t in sorted(gold_prices.keys()):
            if t >= future_dt:
                gold_future = gold_prices[t]
                break
        if gold_future is None:
            continue
        signal = _direction_from_params(score, params)
        records.append(
            {
                "date": dt,
                "score": score,
                "signal": signal,
                "entry_price": gold_at_dt,
                "exit_price": gold_future,
                "return_pct": (gold_future / gold_at_dt - 1) * 100,
            }
        )

    if len(records) < min_samples:
        return {"ok": False, "hit_rate": None, "sample_count": len(records), "reason": "too few samples"}

    df = pd.DataFrame(records)
    directional = df[df["signal"] != 0]

    if directional.empty:
        bull_baseline = float((df["return_pct"] > 0).mean())
        return {
            "ok": True,
            "hit_rate": None,
            "sample_count": len(df),
            "signal_count": 0,
            "avg_return": round(float(df["return_pct"].mean()), 4),
            "bull_baseline_hit_rate": round(bull_baseline, 4),
        }

    hit_rate = float(
        directional.apply(
            lambda r: (r["signal"] > 0 and r["return_pct"] > 0)
            or (r["signal"] < 0 and r["return_pct"] < 0),
            axis=1,
        ).mean()
    )
    bull_baseline = float((df["return_pct"] > 0).mean())

    return {
        "ok": True,
        "hit_rate": round(hit_rate, 4),
        "sample_count": len(df),
        "signal_count": len(directional),
        "signal_ratio": round(len(directional) / len(df), 4),
        "avg_return": round(float(directional["return_pct"].mean()), 4),
        "bull_baseline_hit_rate": round(bull_baseline, 4),
    }


# ── 参数搜索 ─────────────────────────────────────────────────────────


def _sample_params(n: int) -> list[ScoreParams]:
    """从参数空间中随机采样 n 组参数。"""
    samples = []
    for _ in range(n):
        d = {}
        for key, values in PARAM_SPACE.items():
            d[key] = random.choice(values)
        samples.append(ScoreParams.from_dict(d))
    return samples


def optimize_score_params(
    db: Session,
    n_iter: int = N_ITER_DEFAULT,
    horizon_days: int = 20,
    top_k: int = 5,
    random_seed: int = 42,
) -> list[dict[str, Any]]:
    """随机搜索最优评分参数。

    返回评分排名靠前的参数列表（含性能指标），按 hit_rate 降序。
    """
    random.seed(random_seed)
    candidates = _sample_params(n_iter)

    # 始终包含默认参数作为基线
    default_params = ScoreParams.defaults()
    all_candidates = [default_params] + candidates

    results = []
    for i, params in enumerate(all_candidates):
        label = "baseline" if i == 0 else f"candidate_{i}"
        eval_result = evaluate_params(db, params, horizon_days=horizon_days)
        results.append(
            {
                "label": label,
                "params": params.to_dict(),
                **eval_result,
            }
        )

    # 按 hit_rate 排序（None 排后面）
    def sort_key(r: dict) -> float:
        hr = r.get("hit_rate")
        return hr if hr is not None else -999

    results.sort(key=sort_key, reverse=True)
    return results[:top_k + 2]  # 返回 top_k + baseline（如排到外面）


def save_best_params(
    db: Session,
    result: dict[str, Any],
    version: str,
    horizon_days: int = 20,
    notes: str = "",
) -> ScoreParamsVersion:
    """将最优参数保存到数据库。"""
    existing = db.scalar(
        select(ScoreParamsVersion).where(ScoreParamsVersion.version == version)
    )
    if existing is not None:
        existing.params_json = json.dumps(result["params"], ensure_ascii=False)
        existing.hit_rate = result.get("hit_rate")
        existing.sample_count = result.get("sample_count")
        existing.backtest_horizon_days = horizon_days
        existing.notes = notes
        db.commit()
        db.refresh(existing)
        return existing

    record = ScoreParamsVersion(
        version=version,
        params_json=json.dumps(result["params"], ensure_ascii=False),
        hit_rate=result.get("hit_rate"),
        sample_count=result.get("sample_count"),
        backtest_horizon_days=horizon_days,
        is_active=False,
        notes=notes,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def activate_version(db: Session, version: str) -> ScoreParamsVersion | None:
    """激活一个参数版本（其他版本设为非活跃）。"""
    # 先全部关闭
    all_active = db.scalars(
        select(ScoreParamsVersion).where(ScoreParamsVersion.is_active == True)  # noqa: E712
    ).all()
    for v in all_active:
        v.is_active = False

    target = db.scalar(
        select(ScoreParamsVersion).where(ScoreParamsVersion.version == version)
    )
    if target is None:
        return None
    target.is_active = True
    db.commit()
    db.refresh(target)
    return target


def get_active_params(db: Session) -> ScoreParams | None:
    """获取当前激活的参数版本。"""
    active = db.scalar(
        select(ScoreParamsVersion).where(ScoreParamsVersion.is_active == True)  # noqa: E712
    )
    if active is None:
        return None
    return ScoreParams.from_dict(json.loads(active.params_json))
