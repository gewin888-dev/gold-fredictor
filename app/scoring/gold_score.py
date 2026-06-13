"""黄金多空评分引擎（v2 改进版）。

改进：
- P0: 多时间窗口（5/10/20日加权）+ MA交叉→连续百分比
- P1: 因子权重要素（配合optimizer）
- P2: CFTC过期衰减 + 短期动量因子（3日）

11因子 → 加权求和 → 方向判定（±30 阈值）
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.data_quality import source_quality
from app.models import (
    CentralBankGold,
    CftcPosition,
    ChinaGoldPremium,
    GoldPrice,
    GoldScoreSnapshot,
    MacroObservation,
    NewsSentiment,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.scoring.score_optimizer import ScoreParams


# ── FRED 序列 ID ──────────────────────────────────────────────────

REAL_RATE = "DFII10"
NOMINAL_RATE = "DGS10"
INFLATION_EXPECTATION = "T10YIE"
FED_RATE = "FEDFUNDS"
VIX = "VIXCLS"
DOLLAR = "DTWEXBGS"                 # 美元广义贸易加权指数
SP500 = "SP500"                      # 标普 500（美股分流效应）
SILVER = "SILVER"                    # 白银价格（新浪 hf_SI）
GLD_ETF = "GLD_ETF"                  # SPDR Gold Trust ETF 价格
GOOGLE_TREND = "GOOGLE_TREND"        # Google Trends "gold price" 搜索量
GDX = "GDX"                          # 黄金矿业股 ETF
WTI = "WTI"                          # WTI 原油价格
COPPER = "COPPER"                    # 铜价

REQUIRED_FRED_SERIES = [REAL_RATE, NOMINAL_RATE, INFLATION_EXPECTATION, VIX, DOLLAR]
# 可选指标（不阻塞评分）
BONUS_FRED_SERIES = [SP500, SILVER, GLD_ETF, GOOGLE_TREND, GDX, WTI, COPPER]
TRUSTED_SCORING_SOURCES = {"FRED", "YAHOO", "CFTC", "GDELT", "WGC", "IMF", "SGE", "SINA", "LBMA", "TEST", "NEWSAPI", "GOOGLE_TRENDS"}
# 中国溢价必须来自官方/授权口径；SINA 只能作为展示估算，不纳入评分。
PREMIUM_TRUSTED_SOURCES = {"SGE", "LBMA", "TEST"}


@dataclass(frozen=True)
class ScoreResult:
    timestamp: datetime
    total_score: float
    direction: str
    factor_scores: dict[str, float]
    risk_flags: list[str]
    summary: str
    factor_details: dict[str, dict[str, float]]


# ── 工具函数 ──────────────────────────────────────────────────────

def _series_frame(db: Session, series_id: str) -> pd.DataFrame:
    rows = db.scalars(
        select(MacroObservation)
        .where(MacroObservation.series_id == series_id)
        .order_by(MacroObservation.timestamp.asc())
    ).all()
    if not rows:
        return pd.DataFrame(columns=["timestamp", series_id])
    return pd.DataFrame(
        [{"timestamp": row.timestamp, series_id: row.value} for row in rows]
    ).sort_values("timestamp")


def _gold_price_frame(db: Session) -> pd.DataFrame:
    rows = db.scalars(
        select(GoldPrice).order_by(GoldPrice.date.asc())
    ).all()
    if not rows:
        return pd.DataFrame(columns=["timestamp", "gold_price"])
    return pd.DataFrame(
        [{"timestamp": row.date, "gold_price": row.close} for row in rows]
    ).sort_values("timestamp")


def _latest_change(df: pd.DataFrame, column: str, periods: int = 20) -> float | None:
    clean = df.dropna(subset=[column])
    if len(clean) <= periods:
        return None
    return float(clean[column].iloc[-1] - clean[column].iloc[-1 - periods])


def _multi_window_change(df: pd.DataFrame, column: str,
                          windows: list[int] = (5, 10, 20),
                          weights: tuple[float, ...] = (0.5, 0.3, 0.2)) -> float | None:
    """多时间窗口加权变化量。窗口越短权重越大（更敏感）。
    
    windows=(5,10,20), weights=(0.5,0.3,0.2) → 5日变化占50%，10日占30%，20日占20%。
    """
    clean = df.dropna(subset=[column])
    if len(clean) <= max(windows):
        return None
    total = 0.0
    for w, weight in zip(windows, weights):
        if len(clean) > w:
            change = float(clean[column].iloc[-1] - clean[column].iloc[-1 - w])
            total += change * weight
    return total


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _direction(total_score: float) -> str:
    if total_score >= 30:
        return "偏多"
    if total_score <= -30:
        return "偏空"
    return "中性"


def _age_days(timestamp: datetime | None) -> float | None:
    if timestamp is None:
        return None
    now = datetime.now(timezone.utc)
    normalized = timestamp
    if normalized.tzinfo is None:
        normalized = normalized.replace(tzinfo=timezone.utc)
    return (now - normalized).total_seconds() / 86400


def _source_key(source: str | None) -> str:
    return (source or "UNKNOWN").upper()


def _is_trusted_scoring_source(source: str | None) -> bool:
    return source_quality(source).can_score


def _skip_factor_note(name: str, source: str | None, reason: str) -> str:
    source_label = source or "UNKNOWN"
    return f"{name}因子未纳入评分：来源 {source_label} {reason}。"


HORIZON_GROUPS: dict[str, tuple[float, tuple[str, ...]]] = {
    "短线动量": (
        0.40,
        (
            "黄金趋势",
            "短期动量",
            "避险情绪",
            "GLD ETF",
            "矿业股GDX",
            "白银/黄金比",
            "搜索热度",
            "新闻情绪",
        ),
    ),
    "中期宏观": (
        0.40,
        (
            "实际利率",
            "名义利率",
            "联邦基金",
            "美元指数",
            "通胀预期",
            "CFTC投机仓位",
            "美股分流",
            "铜/金比",
            "原油WTI",
            "美元人民币",
            "新闻情绪",
            "中国溢价",
        ),
    ),
    "长期结构": (
        0.20,
        (
            "央行购金",
            "实际利率",
            "通胀预期",
            "CFTC投机仓位",
            "美元人民币",
        ),
    ),
}


def _aggregate_multi_horizon(raw_scores: dict[str, float]) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """把原始因子分转成短/中/长期聚合后的贡献分。

    原始因子代表局部信号，最终总分代表多周期组合观点。这样短线动量、
    中期宏观和长期结构不会被简单相加成一个失真的总分。
    """
    contributions: dict[str, float] = {}
    horizon_details: dict[str, dict[str, float]] = {}

    for horizon_name, (horizon_weight, factor_names) in HORIZON_GROUPS.items():
        present = {name: raw_scores[name] for name in factor_names if name in raw_scores}
        raw_sum = sum(present.values())
        horizon_score = _clamp(raw_sum, -100, 100)
        weighted_score = horizon_score * horizon_weight
        horizon_details[horizon_name] = {
            "权重": float(round(horizon_weight, 2)),
            "原始分": float(round(raw_sum, 2)),
            "贡献分": float(round(weighted_score, 2)),
            "因子数": float(len(present)),
        }
        if not present or abs(raw_sum) < 1e-9:
            continue

        for name, raw_value in present.items():
            contributions[name] = contributions.get(name, 0.0) + raw_value / raw_sum * weighted_score

    rounded = {name: float(round(value, 2)) for name, value in contributions.items()}
    return rounded, horizon_details


def _core_input_quality_notes(db: Session) -> list[str]:
    notes: list[str] = []
    for series_id in REQUIRED_FRED_SERIES:
        row = db.scalar(
            select(MacroObservation)
            .where(MacroObservation.series_id == series_id)
            .order_by(MacroObservation.timestamp.desc())
        )
        quality = source_quality(row.source if row else None)
        if row and not quality.can_score:
            notes.append(f"核心宏观序列 {series_id} 来源为 {row.source}（{quality.label}），当前评分更适合演示校验。")
    gold = db.scalar(select(GoldPrice).order_by(GoldPrice.date.desc()))
    gold_quality = source_quality(gold.source if gold else None)
    if gold and not gold_quality.can_score:
        notes.append(f"核心金价序列来源为 {gold.source}（{gold_quality.label}），当前评分更适合演示校验。")
    return notes


def _aligned_gold_macro_frame(db: Session, gold_prices: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    frames = [_series_frame(db, sid) for sid in REQUIRED_FRED_SERIES]
    missing = [sid for sid, frame in zip(REQUIRED_FRED_SERIES, frames) if frame.empty]
    if missing:
        return pd.DataFrame(), missing

    merged = gold_prices.sort_values("timestamp")
    for frame in frames:
        merged = pd.merge_asof(
            merged.sort_values("timestamp"),
            frame.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )

    fed_frame = _series_frame(db, FED_RATE)
    if not fed_frame.empty:
        merged = pd.merge_asof(
            merged.sort_values("timestamp"),
            fed_frame.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )

    merged = merged.dropna(subset=["gold_price", *REQUIRED_FRED_SERIES])

    # 可选指标（不阻塞）
    for sid in BONUS_FRED_SERIES:
        bonus = _series_frame(db, sid)
        if not bonus.empty:
            merged = pd.merge_asof(
                merged.sort_values("timestamp"),
                bonus.sort_values("timestamp"),
                on="timestamp", direction="backward",
            )

    return merged, []


# ── 各因子评分函数 ────────────────────────────────────────────────

def _cftc_position_score(db: Session, coef: float = 30.0, clamp_lo: float = -15, clamp_hi: float = 15
                         ) -> tuple[float | None, str | None]:
    latest = db.scalar(select(CftcPosition).order_by(CftcPosition.timestamp.desc()))
    if not latest or latest.open_interest <= 0:
        return None, None
    if not _is_trusted_scoring_source(latest.source):
        return None, _skip_factor_note("CFTC投机仓位", latest.source, "不是可信评分源")
    age = _age_days(latest.timestamp)
    # P2: 过期衰减 — 14天内满分，超过14天线性衰减，35天后归零
    if _source_key(latest.source) != "TEST" and age is not None:
        decay = _clamp((35 - age) / (35 - 14), 0.0, 1.0) if age > 14 else 1.0
        if age > 35:
            return None, _skip_factor_note("CFTC投机仓位", latest.source, f"已过期 {age:.0f} 天")
    else:
        decay = 1.0
    net_ratio = latest.noncommercial_net / latest.open_interest
    score = round(_clamp(net_ratio * coef * decay, clamp_lo, clamp_hi), 2)
    note = f"CFTC 非商业净持仓占总持仓约 {net_ratio:.1%}"
    if decay < 1.0:
        note += f"（数据已 {age:.0f} 天，衰减至 {decay:.0%} 权重）"
    note += "。"
    return score, note


def _china_premium_score(db: Session) -> tuple[float | None, str | None]:
    row = db.scalar(
        select(ChinaGoldPremium).order_by(ChinaGoldPremium.timestamp.desc())
    )
    if not row or row.premium_pct is None:
        return None, None
    if _source_key(row.source) not in PREMIUM_TRUSTED_SOURCES:
        return None, _skip_factor_note("中国溢价", row.source, "不是官方 SGE/授权溢价源")
    age = _age_days(row.timestamp)
    if _source_key(row.source) != "TEST" and age is not None and age > 10:
        return None, _skip_factor_note("中国溢价", row.source, f"已过期 {age:.0f} 天")
    score = round(_clamp((row.premium_pct - 1.0) * 3, -10, 10), 2)
    note = f"中国黄金溢价约 {row.premium_pct:.1f}%。"
    return score, note


def _cb_gold_score(db: Session) -> tuple[float | None, str | None]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=600)
    records = db.scalars(
        select(CentralBankGold)
        .where(
            CentralBankGold.country == "Global",
            CentralBankGold.timestamp >= cutoff,
        )
        .order_by(CentralBankGold.timestamp.desc())
    ).all()
    trusted_records = [
        record for record in records
        if record.net_change_tonnes is not None and _is_trusted_scoring_source(record.source)
    ]
    if not trusted_records and records:
        newest = records[0]
        return None, _skip_factor_note("央行购金", newest.source, "不是可信评分源")
    rows = [record.net_change_tonnes for record in trusted_records]
    if not rows or len(rows) < 2:
        return None, None
    net = sum(r for r in rows if r is not None)
    avg_quarterly = net / len(rows)
    score = round(_clamp(avg_quarterly / 20, -10, 10), 2)
    note = f"近 {len(rows)} 个季度全球央行净购金约 {net:.0f} 吨。"
    return score, note


def _sentiment_score(db: Session) -> tuple[float | None, str | None]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    result = db.scalar(
        select(func.avg(NewsSentiment.sentiment_score)).where(
            NewsSentiment.timestamp >= cutoff,
            NewsSentiment.source.in_(TRUSTED_SCORING_SOURCES),
        )
    )
    if result is None:
        latest = db.scalar(select(NewsSentiment).order_by(NewsSentiment.timestamp.desc()))
        if latest and not _is_trusted_scoring_source(latest.source):
            return None, _skip_factor_note("新闻情绪", latest.source, "不是可信评分源")
        return None, None
    score = round(_clamp(float(result) * 1.5, -10, 10), 2)
    note = f"近 7 日黄金新闻情绪均值为 {float(result):.2f}。"
    return score, note


# ── 核心评分函数 ──────────────────────────────────────────────────

def compute_gold_score(db: Session) -> ScoreResult:
    """v2 改进版评分：多时间窗口 + MA连续化 + 动量因子 + CFTC衰减。"""
    gold_prices = _gold_price_frame(db)
    if gold_prices.empty:
        raise ValueError("No gold price data in gold_prices table. Run POST /collect/gold_history first.")

    merged, missing = _aligned_gold_macro_frame(db, gold_prices)
    if missing:
        raise ValueError(f"Missing FRED data for: {', '.join(missing)}")

    if len(merged) < 60:
        raise ValueError("Not enough observations to compute score. Need at least 60 aligned rows.")

    latest = merged.iloc[-1]
    factor_scores: dict[str, float] = {}
    risk_flags: list[str] = _core_input_quality_notes(db)
    current_price = float(latest["gold_price"])
    details: dict[str, dict[str, float]] = {}  # 因子原始变化值

    # P0-1: 多时间窗口加权变化（5/10/20日）
    # 1. 实际利率
    rr_change = _multi_window_change(merged, REAL_RATE)
    if rr_change is not None:
        factor_scores["实际利率"] = round(_clamp(-rr_change * 30, -25, 25), 2)
        details["实际利率"] = {"5日变化": _latest_change(merged, REAL_RATE, 5) or 0,
                               "10日变化": _latest_change(merged, REAL_RATE, 10) or 0,
                               "20日变化": _latest_change(merged, REAL_RATE, 20) or 0}

    # 2. 名义利率
    nr_change = _multi_window_change(merged, NOMINAL_RATE)
    if nr_change is not None:
        factor_scores["名义利率"] = round(_clamp(-nr_change * 40, -20, 20), 2)
        details["名义利率"] = {"5日变化": _latest_change(merged, NOMINAL_RATE, 5) or 0,
                               "10日变化": _latest_change(merged, NOMINAL_RATE, 10) or 0,
                               "20日变化": _latest_change(merged, NOMINAL_RATE, 20) or 0}
        if latest[NOMINAL_RATE] >= 4.5:
            risk_flags.append("10年美债收益率处于高位，黄金持有机会成本较高。")

    # 3. 联邦基金利率（月频，keep 3-period）
    if FED_RATE in merged.columns:
        fed_change = _latest_change(merged, FED_RATE, 3)
        if fed_change is not None:
            factor_scores["联邦基金"] = round(_clamp(-fed_change * 20, -15, 15), 2)
            if latest[FED_RATE] >= 5.0:
                risk_flags.append("联邦基金利率仍处于高位，降息周期启动将利好黄金。")

    # 4. 美元指数
    # P0-1: multi-window on percentage change
    d_clean = merged[[DOLLAR]].dropna()
    if len(d_clean) > 20:
        pct_5 = (latest[DOLLAR] / d_clean[DOLLAR].iloc[-6] - 1) * 100 if len(d_clean) > 5 else 0
        pct_10 = (latest[DOLLAR] / d_clean[DOLLAR].iloc[-11] - 1) * 100 if len(d_clean) > 10 else 0
        pct_20 = (latest[DOLLAR] / d_clean[DOLLAR].iloc[-21] - 1) * 100 if len(d_clean) > 20 else 0
        dollar_multi = pct_5 * 0.5 + pct_10 * 0.3 + pct_20 * 0.2
        factor_scores["美元指数"] = round(_clamp(-dollar_multi * 4, -20, 20), 2)
    else:
        base = merged[DOLLAR].iloc[-21] if len(merged) > 20 else merged[DOLLAR].iloc[0]
        dollar_change_pct = (latest[DOLLAR] / base - 1) * 100 if base and base != 0 else 0.0
        factor_scores["美元指数"] = round(_clamp(-dollar_change_pct * 4, -20, 20), 2)

    # 5. 避险情绪 VIX
    vix_change = _multi_window_change(merged, VIX)
    if vix_change is not None:
        factor_scores["避险情绪"] = round(_clamp(vix_change * 1.2, -15, 15), 2)
        if "避险情绪" not in details:
            details["避险情绪"] = {"5日变化": _latest_change(merged, VIX, 5) or 0,
                                   "10日变化": _latest_change(merged, VIX, 10) or 0,
                                   "20日变化": _latest_change(merged, VIX, 20) or 0}
        if latest[VIX] >= 25:
            risk_flags.append("VIX 处于较高水平，市场避险波动上升。")

    # 6. 通胀预期
    ie_change = _multi_window_change(merged, INFLATION_EXPECTATION)
    if ie_change is not None:
        factor_scores["通胀预期"] = round(_clamp(ie_change * 25, -15, 15), 2)
        if "通胀预期" not in details:
            details["通胀预期"] = {"5日变化": _latest_change(merged, INFLATION_EXPECTATION, 5) or 0,
                                   "10日变化": _latest_change(merged, INFLATION_EXPECTATION, 10) or 0,
                                   "20日变化": _latest_change(merged, INFLATION_EXPECTATION, 20) or 0}

    # 7. P0-2: 黄金趋势 — 连续百分比值
    gold_ma20 = merged["gold_price"].tail(20).mean()
    gold_ma60 = merged["gold_price"].tail(60).mean()
    current_price = float(latest["gold_price"])
    dev_short = (current_price - gold_ma20) / gold_ma20 * 100  # MA20 偏离%
    dev_long = (gold_ma20 - gold_ma60) / gold_ma60 * 100       # MA20 vs MA60 偏离%
    trend_score = _clamp(dev_short * 2 + dev_long * 3, -20, 20)
    factor_scores["黄金趋势"] = round(trend_score, 2)
    details["黄金趋势"] = {"MA20偏离%": round(dev_short, 2), "MA60偏离%": round(dev_long, 2)}

    # P2-2: 新增短期动量因子（3日金价涨跌）
    if len(merged) > 3:
        mom_3d = (current_price / merged["gold_price"].iloc[-4] - 1) * 100
        factor_scores["短期动量"] = round(_clamp(mom_3d * 2, -10, 10), 2)

    # 12. 美股分流效应（SP500 涨 → 资金流股市 → 黄金利空）
    if SP500 in merged.columns:
        sp_clean = merged[[SP500]].dropna()
        if len(sp_clean) >= 2:
            # 用最大可用天数算变化
            n_days = min(len(sp_clean) - 1, 20)
            sp_change = float(sp_clean[SP500].iloc[-1] - sp_clean[SP500].iloc[-1 - n_days])
            # 折算为 20 天等效变化
            sp_change = sp_change * (20 / max(n_days, 1))
            factor_scores["美股分流"] = round(_clamp(-sp_change * 0.01, -10, 10), 2)
            details["美股分流"] = {"SP500现值": round(float(sp_clean[SP500].iloc[-1]), 0),
                                   f"{n_days}日变化": round(sp_change, 2)}

    # 13. 白银/黄金比：作为贵金属风险偏好确认信号，不再机械视为黄金利空。
    if SILVER in merged.columns:
        gold_s = merged["gold_price"]
        silver_s = merged[SILVER].dropna()
        common_idx = gold_s.index.intersection(silver_s.index)
        if len(common_idx) >= 2:
            latest_ratio = float(silver_s.loc[common_idx[-1]]) / float(gold_s.loc[common_idx[-1]])
            n = min(len(common_idx) - 1, 20)
            old_ratio = float(silver_s.loc[common_idx[-1 - n]]) / float(gold_s.loc[common_idx[-1 - n]])
            ratio_change = (latest_ratio / old_ratio - 1) * 100
            ratio_score = ratio_change * 1.5 if trend_score >= 0 else -ratio_change * 1.0
            factor_scores["白银/黄金比"] = round(_clamp(ratio_score, -6, 6), 2)
            details["白银/黄金比"] = {"比值": round(latest_ratio, 4), f"{n}日变化%": round(ratio_change, 2)}

    # 14. GLD ETF 动量（ETF价涨=资金流入=利多）
    if GLD_ETF in merged.columns:
        gld_clean = merged[[GLD_ETF]].dropna()
        if len(gld_clean) >= 2:
            n_days = min(len(gld_clean) - 1, 20)
            gld_change = float(gld_clean[GLD_ETF].iloc[-1] - gld_clean[GLD_ETF].iloc[-1 - n_days])
            gld_change = gld_change * (20 / max(n_days, 1))
            factor_scores["GLD ETF"] = round(_clamp(gld_change * 0.05, -10, 10), 2)
            details["GLD ETF"] = {"价格": round(float(gld_clean[GLD_ETF].iloc[-1]), 2),
                                  f"{n_days}日变化": round(gld_change, 2)}

    # 15. 美元人民币汇率（人民贬=沪金涨=利多）
    usdcny_row = db.scalar(select(ChinaGoldPremium).order_by(ChinaGoldPremium.timestamp.desc()))
    if usdcny_row and usdcny_row.usdcny:
        # 用最老可用数据算变化，不强制 20 天
        oldest = db.scalar(
            select(ChinaGoldPremium)
            .where(ChinaGoldPremium.usdcny.isnot(None))
            .order_by(ChinaGoldPremium.timestamp.asc())
        )
        if oldest and oldest.usdcny and oldest.timestamp < usdcny_row.timestamp:
            cny_change = float(usdcny_row.usdcny) - float(oldest.usdcny)
            factor_scores["美元人民币"] = round(_clamp(cny_change * 2, -10, 10), 2)
            details["美元人民币"] = {"当前": round(float(usdcny_row.usdcny), 4),
                                    "变化": round(cny_change, 4)}

    # 16. Google Trends 搜索热度：温和升温可确认趋势，极端过热才按拥挤交易降分。
    if GOOGLE_TREND in merged.columns:
        gt_clean = merged[[GOOGLE_TREND]].dropna()
        if len(gt_clean) >= 5:
            gt_now = float(gt_clean[GOOGLE_TREND].iloc[-1])
            gt_ma20 = float(gt_clean[GOOGLE_TREND].tail(20).mean())
            deviation = (gt_now / max(gt_ma20, 1) - 1) * 100
            if deviation > 50:
                heat_score = -(deviation - 50) * 0.15
            elif deviation > 15:
                heat_score = (deviation - 15) * 0.08
            elif deviation < -30:
                heat_score = -2.0
            else:
                heat_score = 0.0
            factor_scores["搜索热度"] = round(_clamp(heat_score, -8, 3), 2)
            details["搜索热度"] = {"当前": round(gt_now, 1), "20日均值": round(gt_ma20, 1), "偏离%": round(deviation, 2)}

    # 17. GDX 矿业股动量（矿企领先金价）
    if GDX in merged.columns:
        gdx_clean = merged[[GDX]].dropna()
        if len(gdx_clean) >= 2:
            n = min(len(gdx_clean) - 1, 20)
            gdx_chg = float(gdx_clean[GDX].iloc[-1] - gdx_clean[GDX].iloc[-1 - n])
            gdx_chg = gdx_chg * (20 / max(n, 1))
            factor_scores["矿业股GDX"] = round(_clamp(gdx_chg * 0.3, -10, 10), 2)
            details["矿业股GDX"] = {"价格": round(float(gdx_clean[GDX].iloc[-1]), 2),
                                    f"{n}日变化": round(gdx_chg, 2)}

    # 18. 原油（油价涨→通胀预期↑→利多）
    if WTI in merged.columns:
        wti_clean = merged[[WTI]].dropna()
        if len(wti_clean) >= 2:
            n = min(len(wti_clean) - 1, 20)
            wti_chg = float(wti_clean[WTI].iloc[-1] - wti_clean[WTI].iloc[-1 - n])
            wti_chg = wti_chg * (20 / max(n, 1))
            # 油价和黄金同向（通胀驱动），>100 剧烈变化时反向（需求崩溃）
            factor_scores["原油WTI"] = round(_clamp(wti_chg * 0.03, -10, 10), 2)
            details["原油WTI"] = {"价格": round(float(wti_clean[WTI].iloc[-1]), 2),
                                  f"{n}日变化": round(wti_chg, 2)}

    # 19. 铜/金比：只作为弱风险偏好信号，避免把商品共振上涨误判为黄金强利空。
    if COPPER in merged.columns:
        gold_s = merged["gold_price"]
        copper_s = merged[COPPER].dropna()
        common_idx = gold_s.index.intersection(copper_s.index)
        if len(common_idx) >= 2:
            n = min(len(common_idx) - 1, 20)
            latest_ratio = float(copper_s.loc[common_idx[-1]]) / float(gold_s.loc[common_idx[-1]])
            old_ratio = float(copper_s.loc[common_idx[-1 - n]]) / float(gold_s.loc[common_idx[-1 - n]])
            ratio_change = (latest_ratio / old_ratio - 1) * 100
            factor_scores["铜/金比"] = round(_clamp(-ratio_change * 1.5, -5, 5), 2)
            details["铜/金比"] = {"比值": round(latest_ratio, 4), f"{n}日变化%": round(ratio_change, 2)}

    # 8. CFTC 投机仓位（P2-1: 衰减）
    cftc_score, cftc_note = _cftc_position_score(db)
    if cftc_score is not None:
        factor_scores["CFTC投机仓位"] = cftc_score
    if cftc_note:
        risk_flags.append(cftc_note)

    # 9. 中国溢价（仅官方/授权口径参与评分）
    premium_score, premium_note = _china_premium_score(db)
    if premium_score is not None:
        factor_scores["中国溢价"] = premium_score
    if premium_note:
        risk_flags.append(premium_note)

    # 10. 央行购金
    cb_score, cb_note = _cb_gold_score(db)
    if cb_score is not None:
        factor_scores["央行购金"] = cb_score
    if cb_note:
        risk_flags.append(cb_note)

    # 11. 新闻情绪
    sent_score, sent_note = _sentiment_score(db)
    if sent_score is not None:
        factor_scores["新闻情绪"] = sent_score
    if sent_note:
        risk_flags.append(sent_note)

    # 风险提示汇总
    if latest[REAL_RATE] >= 2.0:
        risk_flags.append("实际利率处于较高水平，可能压制无息资产估值。")
    dollar_20d = (latest[DOLLAR] / merged[DOLLAR].iloc[-21] - 1) * 100 if len(merged) > 20 else 0
    if abs(dollar_20d) >= 2:
        risk_flags.append("美元指数近 20 个交易日波动较大，需关注汇率因子扰动。")
    if not risk_flags:
        risk_flags.append("当前未触发显著宏观风险阈值。")

    raw_factor_scores = dict(factor_scores)
    factor_scores, horizon_details = _aggregate_multi_horizon(raw_factor_scores)
    details["多周期评分"] = {
        f"{name}_贡献分": values["贡献分"]
        for name, values in horizon_details.items()
    } | {
        f"{name}_原始分": values["原始分"]
        for name, values in horizon_details.items()
    }
    details["原始因子分"] = {name: float(round(value, 2)) for name, value in raw_factor_scores.items()}

    total = round(_clamp(sum(factor_scores.values()), -100, 100), 2)
    direction = _direction(total)
    short_score = horizon_details.get("短线动量", {}).get("贡献分", 0)
    medium_score = horizon_details.get("中期宏观", {}).get("贡献分", 0)
    long_score = horizon_details.get("长期结构", {}).get("贡献分", 0)
    summary = (
        f"黄金多空评分为 {total}，方向为{direction}。"
        f"短线{short_score:+.1f}，中期{medium_score:+.1f}，长期{long_score:+.1f}。"
        "该结果仅用于数据分析和风险提示。"
    )
    if any("更适合演示校验" in flag for flag in risk_flags):
        summary += " 当前存在样本或占位输入，不能视为生产级评分。"

    return ScoreResult(
        timestamp=datetime.now(timezone.utc),
        total_score=total,
        direction=direction,
        factor_scores=factor_scores,
        risk_flags=risk_flags,
        summary=summary,
        factor_details=details,
    )


def compute_and_store_gold_score(db: Session) -> GoldScoreSnapshot:
    result = compute_gold_score(db)
    snapshot = GoldScoreSnapshot(
        timestamp=result.timestamp,
        total_score=result.total_score,
        direction=result.direction,
        factor_scores=json.dumps({"scores": result.factor_scores, "details": result.factor_details}, ensure_ascii=False),
        risk_flags=json.dumps(result.risk_flags, ensure_ascii=False),
        summary=result.summary,
        source="rule_v2",
    )
    db.add(snapshot)
    db.commit()
    db.refresh(snapshot)
    return snapshot


def compute_gold_score_with_params(db: Session, params: "ScoreParams") -> ScoreResult:
    """使用优化后的参数计算评分（代替硬编码规则）。"""
    gold_prices = _gold_price_frame(db)
    if gold_prices.empty:
        raise ValueError("No gold price data in gold_prices table.")

    merged, missing = _aligned_gold_macro_frame(db, gold_prices)
    if missing:
        raise ValueError(f"Missing FRED data for: {', '.join(missing)}")

    min_data = max(getattr(params, "trend_ma_long", 60), 60)
    if len(merged) < min_data:
        raise ValueError(f"Need at least {min_data} aligned rows, got {len(merged)}.")

    latest = merged.iloc[-1]
    hist_len = min(len(merged), getattr(params, "trend_ma_short", 20))

    factor_scores: dict[str, float] = {}
    risk_flags: list[str] = _core_input_quality_notes(db)
    details: dict[str, dict[str, float]] = {}
    current_price = float(latest["gold_price"])

    # 实际利率 — multi-window
    rr_change = _multi_window_change(merged, REAL_RATE)
    if rr_change is not None:
        factor_scores["实际利率"] = round(
            _clamp(-rr_change * params.real_rate_coef, params.real_rate_clamp_low, params.real_rate_clamp_high), 2
        )

    # 名义利率
    if NOMINAL_RATE in merged.columns:
        nr_change = _multi_window_change(merged, NOMINAL_RATE)
        if nr_change is not None:
            nom_coef = getattr(params, "nominal_rate_coef", 40.0)
            factor_scores["名义利率"] = round(_clamp(-nr_change * nom_coef, -20, 20), 2)

    # 联邦基金利率
    if FED_RATE in merged.columns:
        fed_change = _latest_change(merged, FED_RATE, 3)
        if fed_change is not None:
            fed_coef = getattr(params, "fed_rate_coef", 20.0)
            factor_scores["联邦基金"] = round(_clamp(-fed_change * fed_coef, -15, 15), 2)

    # 美元指数 — multi-window pct
    d_clean = merged[[DOLLAR]].dropna()
    if len(d_clean) > 20:
        pct_5 = (latest[DOLLAR] / d_clean[DOLLAR].iloc[-6] - 1) * 100 if len(d_clean) > 5 else 0
        pct_10 = (latest[DOLLAR] / d_clean[DOLLAR].iloc[-11] - 1) * 100 if len(d_clean) > 10 else 0
        pct_20 = (latest[DOLLAR] / d_clean[DOLLAR].iloc[-21] - 1) * 100 if len(d_clean) > 20 else 0
        dollar_multi = pct_5 * 0.5 + pct_10 * 0.3 + pct_20 * 0.2
    else:
        dollar_multi = (latest[DOLLAR] / d_clean[DOLLAR].iloc[-1 * min(len(d_clean)-1,20)] - 1) * 100 if len(d_clean) > 1 else 0
    factor_scores["美元指数"] = round(
        _clamp(-dollar_multi * params.dollar_coef, params.dollar_clamp_low, params.dollar_clamp_high), 2
    )

    # VIX — multi-window
    vix_change = _multi_window_change(merged, VIX)
    if vix_change is not None:
        factor_scores["避险情绪"] = round(
            _clamp(vix_change * params.vix_coef, params.vix_clamp_low, params.vix_clamp_high), 2
        )
        if latest[VIX] >= 25:
            risk_flags.append("VIX 处于较高水平，市场避险波动上升。")

    # 通胀预期 — multi-window
    ie_change = _multi_window_change(merged, INFLATION_EXPECTATION)
    if ie_change is not None:
        factor_scores["通胀预期"] = round(
            _clamp(ie_change * params.inflation_coef, params.inflation_clamp_low, params.inflation_clamp_high), 2
        )

    # 黄金趋势 — 连续百分比
    gold_col = merged["gold_price"].dropna()
    if len(gold_col) >= params.trend_ma_long:
        gold_ma_short = gold_col.tail(params.trend_ma_short).mean()
        gold_ma_long = gold_col.tail(params.trend_ma_long).mean()
        dev_short = (current_price - gold_ma_short) / gold_ma_short * 100
        dev_long = (gold_ma_short - gold_ma_long) / gold_ma_long * 100
        trend_score = _clamp(dev_short * 2 + dev_long * 3, -20, 20)
        factor_scores["黄金趋势"] = round(trend_score, 2)
        details["黄金趋势"] = {"MA20偏离%": round(dev_short, 2), "MA60偏离%": round(dev_long, 2)}

    # 短期动量
    if len(merged) > 3:
        mom_3d = (current_price / gold_col.iloc[-4] - 1) * 100
        factor_scores["短期动量"] = round(_clamp(mom_3d * 2, -10, 10), 2)

    # CFTC — with decay
    cftc_coef = getattr(params, "cftc_coef", 30.0)
    cftc_score, cftc_note = _cftc_position_score(db, coef=cftc_coef,
                                                  clamp_lo=params.cftc_clamp_low,
                                                  clamp_hi=params.cftc_clamp_high)
    if cftc_score is not None:
        factor_scores["CFTC投机仓位"] = cftc_score
    if cftc_note:
        risk_flags.append(cftc_note)

    # 中国溢价
    premium_score, premium_note = _china_premium_score(db)
    if premium_score is not None:
        factor_scores["中国溢价"] = premium_score
    if premium_note:
        risk_flags.append(premium_note)

    # 央行购金
    cb_score, cb_note = _cb_gold_score(db)
    if cb_score is not None:
        factor_scores["央行购金"] = cb_score
    if cb_note:
        risk_flags.append(cb_note)

    # 新闻情绪
    sent_score, sent_note = _sentiment_score(db)
    if sent_score is not None:
        factor_scores["新闻情绪"] = sent_score
    if sent_note:
        risk_flags.append(sent_note)

    # 风险汇总
    if latest[REAL_RATE] >= 2.0:
        risk_flags.append("实际利率处于较高水平，可能压制无息资产估值。")
    if not risk_flags:
        risk_flags.append("当前未触发显著宏观风险阈值。")

    raw_factor_scores = dict(factor_scores)
    factor_scores, horizon_details = _aggregate_multi_horizon(raw_factor_scores)
    details["多周期评分"] = {
        f"{name}_贡献分": values["贡献分"]
        for name, values in horizon_details.items()
    } | {
        f"{name}_原始分": values["原始分"]
        for name, values in horizon_details.items()
    }
    details["原始因子分"] = {name: float(round(value, 2)) for name, value in raw_factor_scores.items()}

    total = round(_clamp(sum(factor_scores.values()), -100, 100), 2)
    direction = _direction(total)
    short_score = horizon_details.get("短线动量", {}).get("贡献分", 0)
    medium_score = horizon_details.get("中期宏观", {}).get("贡献分", 0)
    long_score = horizon_details.get("长期结构", {}).get("贡献分", 0)
    summary = (
        f"黄金多空评分为 {total}，方向为{direction}。"
        f"短线{short_score:+.1f}，中期{medium_score:+.1f}，长期{long_score:+.1f}。"
        "该结果仅用于数据分析和风险提示。"
    )

    return ScoreResult(
        timestamp=datetime.now(timezone.utc),
        total_score=total,
        direction=direction,
        factor_scores=factor_scores,
        risk_flags=risk_flags,
        summary=summary,
        factor_details=details,
    )


def compute_and_store_gold_score_with_params(db: Session, params: "ScoreParams",
                                              source: str = "rule_v2_optimized") -> GoldScoreSnapshot:
    result = compute_gold_score_with_params(db, params)
    snapshot = GoldScoreSnapshot(
        timestamp=result.timestamp,
        total_score=result.total_score,
        direction=result.direction,
        factor_scores=json.dumps({"scores": result.factor_scores, "details": result.factor_details}, ensure_ascii=False),
        risk_flags=json.dumps(result.risk_flags, ensure_ascii=False),
        summary=result.summary,
        source=source,
    )
    db.add(snapshot)
    db.commit()
    db.refresh(snapshot)
    return snapshot
