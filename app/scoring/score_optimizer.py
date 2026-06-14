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
    ExternalMarketIndicator,
    ScoreParamsVersion,
)
from app.scoring.gold_score import (
    BANK_RESERVES,
    DOLLAR,
    DEBT_TO_GDP,
    FED_BALANCE_SHEET,
    FEDERAL_DEFICIT,
    INFLATION_EXPECTATION,
    NOMINAL_RATE,
    FED_RATE,
    GOLD_VOLATILITY,
    REAL_RATE,
    REAL_RATE_5Y,
    REAL_RATE_30Y,
    TERM_PREMIUM_10Y,
    TREASURY_GENERAL_ACCOUNT,
    OVERNIGHT_RRP,
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
    real_curve_coef: float = 28.0
    term_premium_coef: float = 18.0
    liquidity_coef: float = 1.0
    fiscal_coef: float = 1.0
    etf_flow_coef: float = 0.35
    option_vol_coef: float = 1.0

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
    real_curve_clamp_low: float = -20.0
    real_curve_clamp_high: float = 20.0
    term_premium_clamp_low: float = -12.0
    term_premium_clamp_high: float = 12.0
    liquidity_clamp_low: float = -12.0
    liquidity_clamp_high: float = 12.0
    fiscal_clamp_low: float = -10.0
    fiscal_clamp_high: float = 10.0
    etf_flow_clamp_low: float = -12.0
    etf_flow_clamp_high: float = 12.0
    option_vol_clamp_low: float = -8.0
    option_vol_clamp_high: float = 8.0

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
    "fed_rate_coef":         [10, 15, 20, 25, 30],
    "real_curve_coef":       [18, 22, 28, 34, 40],
    "term_premium_coef":     [10, 14, 18, 22, 26],
    "liquidity_coef":        [0.5, 0.8, 1.0, 1.2, 1.5],
    "fiscal_coef":           [0.5, 0.8, 1.0, 1.2, 1.5],
    "etf_flow_coef":         [0.15, 0.25, 0.35, 0.45, 0.6],
    "option_vol_coef":       [0.6, 0.8, 1.0, 1.2, 1.5],
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
    etf_flow_cache: dict[pd.Timestamp, float] | None = None,
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

    # 实际收益率曲线
    real_curve_cols = [col for col in (REAL_RATE_5Y, REAL_RATE, REAL_RATE_30Y) if col in hist.columns]
    if len(real_curve_cols) >= 2:
        curve_changes = []
        for col in real_curve_cols:
            clean = hist[[col]].dropna()
            if len(clean) > params.trend_ma_short:
                curve_changes.append(float(clean[col].iloc[-1] - clean[col].iloc[-1 - params.trend_ma_short]))
        if curve_changes:
            avg_change = sum(curve_changes) / len(curve_changes)
            slope_score = 0.0
            if REAL_RATE_5Y in hist.columns and REAL_RATE_30Y in hist.columns:
                clean = hist[[REAL_RATE_5Y, REAL_RATE_30Y]].dropna()
                if len(clean) > params.trend_ma_short:
                    latest_slope = float(clean[REAL_RATE_30Y].iloc[-1] - clean[REAL_RATE_5Y].iloc[-1])
                    old_slope = float(
                        clean[REAL_RATE_30Y].iloc[-1 - params.trend_ma_short]
                        - clean[REAL_RATE_5Y].iloc[-1 - params.trend_ma_short]
                    )
                    slope_score = (latest_slope - old_slope) * 8
            factor_sum += _clamp(
                -avg_change * params.real_curve_coef + slope_score,
                params.real_curve_clamp_low,
                params.real_curve_clamp_high,
            )

    # 美债期限溢价
    if TERM_PREMIUM_10Y in hist.columns:
        clean = hist[[TERM_PREMIUM_10Y]].dropna()
        if len(clean) > params.trend_ma_short:
            premium_change = float(clean[TERM_PREMIUM_10Y].iloc[-1] - clean[TERM_PREMIUM_10Y].iloc[-1 - params.trend_ma_short])
            factor_sum += _clamp(
                premium_change * params.term_premium_coef,
                params.term_premium_clamp_low,
                params.term_premium_clamp_high,
            )

    # 美元流动性
    liquidity_parts: list[float] = []
    for col, weight in [
        (FED_BALANCE_SHEET, 0.7),
        (BANK_RESERVES, 0.5),
        (TREASURY_GENERAL_ACCOUNT, -0.35),
        (OVERNIGHT_RRP, -0.25),
    ]:
        if col not in hist.columns:
            continue
        clean = hist[[col]].dropna()
        if len(clean) > params.trend_ma_short:
            base = float(clean[col].iloc[-1 - params.trend_ma_short])
            if base:
                pct = (float(clean[col].iloc[-1]) / base - 1) * 100
                liquidity_parts.append(pct * weight)
    if liquidity_parts:
        factor_sum += _clamp(
            sum(liquidity_parts) * params.liquidity_coef,
            params.liquidity_clamp_low,
            params.liquidity_clamp_high,
        )

    # 财政压力
    fiscal_parts: list[float] = []
    if DEBT_TO_GDP in hist.columns:
        clean = hist[[DEBT_TO_GDP]].dropna()
        if len(clean) > 1:
            idx = min(params.trend_ma_long, len(clean) - 1)
            fiscal_parts.append((float(clean[DEBT_TO_GDP].iloc[-1]) - float(clean[DEBT_TO_GDP].iloc[-1 - idx])) * 1.2)
    if FEDERAL_DEFICIT in hist.columns:
        clean = hist[[FEDERAL_DEFICIT]].dropna()
        if len(clean) > 1:
            idx = min(252, len(clean) - 1)
            fiscal_parts.append(-(float(clean[FEDERAL_DEFICIT].iloc[-1]) - float(clean[FEDERAL_DEFICIT].iloc[-1 - idx])) / 200000)
    if fiscal_parts:
        factor_sum += _clamp(
            sum(fiscal_parts) * params.fiscal_coef,
            params.fiscal_clamp_low,
            params.fiscal_clamp_high,
        )

    # 期权隐含波动率（GVZ 代理）
    if GOLD_VOLATILITY in hist.columns:
        clean = hist[[GOLD_VOLATILITY]].dropna()
        if len(clean) > params.trend_ma_short:
            gvz_now = float(clean[GOLD_VOLATILITY].iloc[-1])
            gvz_ma = float(clean[GOLD_VOLATILITY].tail(params.trend_ma_short).mean())
            gvz_change = gvz_now - float(clean[GOLD_VOLATILITY].iloc[-1 - params.trend_ma_short])
            option_score = ((gvz_now - gvz_ma) * 0.25 + gvz_change * 0.35) * params.option_vol_coef
            factor_sum += _clamp(option_score, params.option_vol_clamp_low, params.option_vol_clamp_high)

    # ETF 资金流（若存在历史授权/官方入库值）
    if etf_flow_cache is not None:
        key = target_date.normalize()
        value = etf_flow_cache.get(key)
        if value is not None:
            factor_sum += _clamp(
                value * params.etf_flow_coef,
                params.etf_flow_clamp_low,
                params.etf_flow_clamp_high,
            )

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
    series_ids = [
        REAL_RATE,
        INFLATION_EXPECTATION,
        VIX,
        DOLLAR,
        NOMINAL_RATE,
        FED_RATE,
        REAL_RATE_5Y,
        REAL_RATE_30Y,
        TERM_PREMIUM_10Y,
        FED_BALANCE_SHEET,
        TREASURY_GENERAL_ACCOUNT,
        OVERNIGHT_RRP,
        BANK_RESERVES,
        FEDERAL_DEFICIT,
        DEBT_TO_GDP,
        GOLD_VOLATILITY,
    ]
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

    etf_flow_cache: dict[pd.Timestamp, float] = {}
    etf_rows = db.scalars(
        select(ExternalMarketIndicator)
        .where(ExternalMarketIndicator.indicator_id == "GLD_FLOW_TONNES")
        .order_by(ExternalMarketIndicator.timestamp.asc())
    ).all()
    for row in etf_rows:
        etf_flow_cache[pd.Timestamp(row.timestamp).normalize()] = float(row.value)

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
        score = compute_score_at_date(db, dt, merged.copy(), params, cftc_cache, etf_flow_cache)
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
    long_signals = int((directional["signal"] > 0).sum()) if not directional.empty else 0
    short_signals = int((directional["signal"] < 0).sum()) if not directional.empty else 0

    if directional.empty:
        bull_baseline = float((df["return_pct"] > 0).mean())
        return {
            "ok": True,
            "hit_rate": None,
            "sample_count": len(df),
            "signal_count": 0,
            "long_signal_count": 0,
            "short_signal_count": 0,
            "signal_ratio": 0.0,
            "avg_return": round(float(df["return_pct"].mean()), 4),
            "worst_decile_return": round(float(df["return_pct"].quantile(0.10)), 4),
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
    signed_returns = directional["return_pct"] * directional["signal"]
    sorted_dates = df["date"].sort_values().tolist()
    first_cut = sorted_dates[len(sorted_dates) // 2]
    recent_directional = directional[directional["date"] >= first_cut]
    recent_hit_rate = None
    if not recent_directional.empty:
        recent_hit_rate = float(
            recent_directional.apply(
                lambda r: (r["signal"] > 0 and r["return_pct"] > 0)
                or (r["signal"] < 0 and r["return_pct"] < 0),
                axis=1,
            ).mean()
        )

    return {
        "ok": True,
        "hit_rate": round(hit_rate, 4),
        "sample_count": len(df),
        "signal_count": len(directional),
        "long_signal_count": long_signals,
        "short_signal_count": short_signals,
        "signal_ratio": round(len(directional) / len(df), 4),
        "avg_return": round(float(signed_returns.mean()), 4),
        "worst_decile_return": round(float(signed_returns.quantile(0.10)), 4),
        "bull_baseline_hit_rate": round(bull_baseline, 4),
        "baseline_lift": round(hit_rate - bull_baseline, 4),
        "recent_hit_rate": round(recent_hit_rate, 4) if recent_hit_rate is not None else None,
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

    baseline = next((r for r in results if r["label"] == "baseline"), None)
    baseline_hit_rate = baseline.get("hit_rate") if baseline else None
    for row in results:
        if row.get("hit_rate") is not None and baseline_hit_rate is not None:
            row["baseline_lift"] = round(float(row["hit_rate"]) - float(baseline_hit_rate), 4)
        row["activation_check"] = score_params_activation_decision(row, baseline_hit_rate)

    def sort_key(r: dict) -> tuple[float, float, float, float]:
        hr = r.get("hit_rate")
        lift = r.get("baseline_lift")
        signal_ratio = r.get("signal_ratio")
        worst = r.get("worst_decile_return")
        return (
            float(hr) if hr is not None else -999.0,
            float(lift) if lift is not None else -999.0,
            float(signal_ratio) if signal_ratio is not None else -999.0,
            float(worst) if worst is not None else -999.0,
        )

    results.sort(key=sort_key, reverse=True)
    top = results[:top_k]
    if baseline is not None and baseline not in top:
        top.append(baseline)
    return top


def score_params_activation_decision(
    result: dict[str, Any],
    baseline_hit_rate: float | None = None,
    *,
    min_samples: int = 120,
    min_signal_ratio: float = 0.20,
    min_baseline_lift: float = 0.03,
    min_long_signals: int = 10,
    min_short_signals: int = 10,
) -> dict[str, Any]:
    """评分参数候选激活门控。默认只供报告和显式授权自动激活使用。"""
    reasons: list[str] = []
    eligible = True

    sample_count = int(result.get("sample_count") or 0)
    signal_ratio = float(result.get("signal_ratio") or 0.0)
    long_count = int(result.get("long_signal_count") or 0)
    short_count = int(result.get("short_signal_count") or 0)
    hit_rate = result.get("hit_rate")
    if sample_count < min_samples:
        eligible = False
        reasons.append(f"sample_count {sample_count} < {min_samples}")
    if signal_ratio < min_signal_ratio:
        eligible = False
        reasons.append(f"signal_ratio {signal_ratio:.2f} < {min_signal_ratio:.2f}")
    if long_count < min_long_signals:
        eligible = False
        reasons.append(f"long_signal_count {long_count} < {min_long_signals}")
    if short_count < min_short_signals:
        eligible = False
        reasons.append(f"short_signal_count {short_count} < {min_short_signals}")
    if hit_rate is None:
        eligible = False
        reasons.append("hit_rate missing")
    elif baseline_hit_rate is not None and float(hit_rate) - float(baseline_hit_rate) < min_baseline_lift:
        eligible = False
        reasons.append(
            f"baseline_lift {float(hit_rate) - float(baseline_hit_rate):.3f} < {min_baseline_lift:.3f}"
        )
    recent_hit_rate = result.get("recent_hit_rate")
    if recent_hit_rate is not None and hit_rate is not None and float(recent_hit_rate) + 0.05 < float(hit_rate):
        eligible = False
        reasons.append("recent window degraded materially")
    if eligible:
        reasons.append("all thresholds passed")
    return {
        "eligible": eligible,
        "reasons": reasons,
        "thresholds": {
            "min_samples": min_samples,
            "min_signal_ratio": min_signal_ratio,
            "min_baseline_lift": min_baseline_lift,
            "min_long_signals": min_long_signals,
            "min_short_signals": min_short_signals,
        },
    }


def overfit_risk_assessment(result: dict[str, Any], baseline: dict[str, Any] | None = None) -> dict[str, Any]:
    """Heuristic overfit risk flags for human review."""
    warnings: list[str] = []
    hit_rate = result.get("hit_rate")
    baseline_hit = baseline.get("hit_rate") if baseline else result.get("bull_baseline_hit_rate")
    sample_count = int(result.get("sample_count") or 0)
    signal_count = int(result.get("signal_count") or 0)
    long_count = int(result.get("long_signal_count") or 0)
    short_count = int(result.get("short_signal_count") or 0)
    recent_hit_rate = result.get("recent_hit_rate")
    worst_decile = result.get("worst_decile_return")
    stored_hit_rate = result.get("stored_hit_rate")

    if hit_rate is not None and float(hit_rate) >= 0.90:
        warnings.append("命中率高于 90%，需要重点检查是否过拟合或样本区间过于单一。")
    if stored_hit_rate is not None and float(stored_hit_rate) >= 0.90:
        warnings.append("保存时命中率高于 90%，需要复核搜索过程是否过拟合。")
    if stored_hit_rate is not None and hit_rate is not None and float(stored_hit_rate) - float(hit_rate) >= 0.10:
        warnings.append("保存指标与当前统一口径复核差距超过 10 个百分点，存在口径漂移或过拟合风险。")
    if hit_rate is not None and baseline_hit is not None and float(hit_rate) - float(baseline_hit) >= 0.25:
        warnings.append("相对 baseline 提升超过 25 个百分点，建议做分段/滚动窗口复核。")
    if sample_count < 120:
        warnings.append("总样本数不足 120，候选稳定性较弱。")
    if signal_count and min(long_count, short_count) / max(1, signal_count) < 0.15:
        warnings.append("多空信号分布不均衡，可能只适合单边行情。")
    if recent_hit_rate is not None and hit_rate is not None and float(recent_hit_rate) + 0.05 < float(hit_rate):
        warnings.append("最近窗口表现弱于总体表现，存在退化迹象。")
    if worst_decile is not None and float(worst_decile) < -5:
        warnings.append("最差分位方向收益低于 -5%，尾部风险偏高。")

    level = "low"
    if len(warnings) >= 2:
        level = "high"
    elif warnings:
        level = "medium"
    return {
        "level": level,
        "warnings": warnings or ["未触发明显过拟合警告，但仍需人工审核。"],
        "not_recommended_for_direct_activation": level != "low",
    }


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


def deactivate_all_versions(db: Session) -> int:
    """停用所有评分参数版本，恢复默认 rule_v2 评分。"""
    rows = db.scalars(
        select(ScoreParamsVersion).where(ScoreParamsVersion.is_active == True)  # noqa: E712
    ).all()
    for row in rows:
        row.is_active = False
    db.commit()
    return len(rows)


def get_active_params(db: Session) -> ScoreParams | None:
    """获取当前激活的参数版本。"""
    active = db.scalar(
        select(ScoreParamsVersion).where(ScoreParamsVersion.is_active == True)  # noqa: E712
    )
    if active is None:
        return None
    return ScoreParams.from_dict(json.loads(active.params_json))
