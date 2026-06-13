"""Comprehensive audit of all scoring factors in gold_score.py.

Covers:
- All 17 factors: formula logic, clamp ranges, edge cases
- Missing data / zero values / NaN handling
- Sum of factor_scores == total_score (pre-clamp check)
- compute_gold_score_with_params / optimizer compatibility
- GoldScoreSnapshot JSON round-trip
"""
import json
import math
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.models import (
    CentralBankGold,
    ChinaGoldPremium,
    CftcPosition,
    GoldPrice,
    GoldScoreSnapshot,
    MacroObservation,
    NewsSentiment,
)
from app.scoring.gold_score import (
    COPPER,
    DOLLAR,
    FED_RATE,
    GDX,
    GLD_ETF,
    GOOGLE_TREND,
    INFLATION_EXPECTATION,
    NOMINAL_RATE,
    REAL_RATE,
    SILVER,
    SP500,
    VIX,
    WTI,
    _clamp,
    _multi_window_change,
    compute_and_store_gold_score,
    compute_gold_score,
    compute_gold_score_with_params,
)
from app.scoring.score_optimizer import ScoreParams


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def db():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


NOW = datetime.now(timezone.utc)
DAYS = 80


def _add_macro(db, series_id, value_fn, days=DAYS, source="TEST"):
    start = NOW - timedelta(days=days - 1)
    for i in range(days):
        db.add(MacroObservation(
            series_id=series_id,
            timestamp=start + timedelta(days=i),
            value=float(value_fn(i)),
            source=source,
        ))


def _add_gold(db, price_fn=None, days=DAYS, source="TEST"):
    start = NOW - timedelta(days=days - 1)
    for i in range(days):
        p = float(price_fn(i)) if price_fn else 2400.0 + i * 2.0
        db.add(GoldPrice(date=start + timedelta(days=i), open=p - 3, high=p + 8,
                         low=p - 8, close=p, source=source))


def _add_required(db, days=DAYS):
    """Insert the 5 required FRED series + FedFunds + gold prices."""
    _add_macro(db, REAL_RATE,            lambda i: 2.2 - i * 0.005)
    _add_macro(db, NOMINAL_RATE,         lambda i: 4.2 - i * 0.002)
    _add_macro(db, INFLATION_EXPECTATION,lambda i: 2.1 + i * 0.002)
    _add_macro(db, VIX,                  lambda i: 14 + i * 0.04)
    _add_macro(db, DOLLAR,               lambda i: 120 - i * 0.05)
    _add_macro(db, FED_RATE,             lambda i: 5.3 - i * 0.001)
    _add_gold(db, days=days)
    db.commit()


def _add_bonus(db):
    """Insert all optional bonus series."""
    _add_macro(db, SP500,        lambda i: 5000 + i * 5)
    _add_macro(db, SILVER,       lambda i: 30 + i * 0.01)
    _add_macro(db, GLD_ETF,      lambda i: 220 + i * 0.2)
    _add_macro(db, GOOGLE_TREND, lambda i: 50 + (i % 10))
    _add_macro(db, GDX,          lambda i: 35 + i * 0.05)
    _add_macro(db, WTI,          lambda i: 80 - i * 0.02)
    _add_macro(db, COPPER,       lambda i: 4.5 + i * 0.001)
    db.commit()


def _add_all(db):
    _add_required(db)
    _add_bonus(db)


# ── Unit tests: utility functions ─────────────────────────────────────────────

def test_clamp_basic():
    assert _clamp(50, -20, 20) == 20
    assert _clamp(-50, -20, 20) == -20
    assert _clamp(10, -20, 20) == 10


def test_clamp_equal_bounds():
    assert _clamp(5, 0, 0) == 0


def test_multi_window_change_insufficient_data():
    import pandas as pd
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    # max window=20, only 3 rows — must return None
    assert _multi_window_change(df, "x") is None


def test_multi_window_change_with_nan():
    import pandas as pd
    vals = [float("nan")] * 10 + [1.0] * 30
    df = pd.DataFrame({"x": vals})
    # After dropna, 30 clean rows > max window 20 → should succeed
    result = _multi_window_change(df, "x")
    assert result is not None
    # Flat series → all window changes = 0
    assert result == pytest.approx(0.0)


# ── Factor formula correctness ────────────────────────────────────────────────

def test_real_rate_factor_sign(db):
    """Falling real rate → positive gold score."""
    _add_required(db)
    result = compute_gold_score(db)
    # REAL_RATE decreases over time (value_fn = 2.2 - i*0.005)
    # multi-window change should be negative → score = -change*30 → positive
    assert result.factor_scores["实际利率"] > 0


def test_nominal_rate_factor_sign(db):
    """Falling nominal rate → positive score."""
    _add_required(db)
    result = compute_gold_score(db)
    assert result.factor_scores["名义利率"] > 0


def test_dollar_factor_sign(db):
    """Falling dollar → positive gold score."""
    _add_required(db)
    result = compute_gold_score(db)
    # DOLLAR = 120 - i*0.05 → falling → score = -negative_pct*4 → positive
    assert result.factor_scores["美元指数"] > 0


def test_vix_factor_sign(db):
    """Rising VIX → positive gold score (safe-haven demand)."""
    _add_required(db)
    result = compute_gold_score(db)
    # VIX = 14 + i*0.04 → rising → score = change*1.2 → positive
    assert result.factor_scores["避险情绪"] > 0


def test_inflation_factor_sign(db):
    """Rising inflation expectations → positive gold score."""
    _add_required(db)
    result = compute_gold_score(db)
    # T10YIE = 2.1 + i*0.002 → rising → score = change*25 → positive
    assert result.factor_scores["通胀预期"] > 0


def test_gold_trend_factor_present(db):
    """黄金趋势 always computed when ≥60 rows."""
    _add_required(db)
    result = compute_gold_score(db)
    assert "黄金趋势" in result.factor_scores


def test_short_term_momentum_factor(db):
    """短期动量 present and within clamp."""
    _add_required(db)
    result = compute_gold_score(db)
    assert "短期动量" in result.factor_scores
    assert -10 <= result.factor_scores["短期动量"] <= 10


# ── Bonus factors present when data available ──────────────────────────────────

def test_bonus_factors_present_when_data_added(db):
    _add_all(db)
    result = compute_gold_score(db)
    for key in ["美股分流", "GLD ETF", "搜索热度", "矿业股GDX", "原油WTI"]:
        assert key in result.factor_scores, f"Missing factor: {key}"


def test_silver_gold_ratio_factor(db):
    _add_all(db)
    result = compute_gold_score(db)
    assert "白银/黄金比" in result.factor_scores
    assert -10 <= result.factor_scores["白银/黄金比"] <= 10


def test_copper_gold_ratio_factor(db):
    _add_all(db)
    result = compute_gold_score(db)
    assert "铜/金比" in result.factor_scores
    assert -10 <= result.factor_scores["铜/金比"] <= 10


# ── Clamp ranges for each factor ──────────────────────────────────────────────

FACTOR_CLAMPS = {
    "实际利率":  (-25, 25),
    "名义利率":  (-20, 20),
    "联邦基金":  (-15, 15),
    "美元指数":  (-20, 20),
    "避险情绪":  (-15, 15),
    "通胀预期":  (-15, 15),
    "黄金趋势":  (-20, 20),
    "短期动量":  (-10, 10),
    "美股分流":  (-10, 10),
    "白银/黄金比": (-10, 10),
    "GLD ETF":   (-10, 10),
    "搜索热度":  (-10, 10),
    "矿业股GDX": (-10, 10),
    "原油WTI":   (-10, 10),
    "铜/金比":   (-10, 10),
}


def test_all_factors_within_clamp_ranges(db):
    _add_all(db)
    # Also add CFTC and China premium for full coverage
    db.add(CftcPosition(
        market_name="GOLD", contract_market_code="088691", exchange_code="CMX",
        timestamp=NOW, open_interest=300000,
        noncommercial_long=180000, noncommercial_short=50000, noncommercial_spreading=0,
        commercial_long=60000, commercial_short=210000, noncommercial_net=130000,
        source="TEST",
    ))
    db.add(ChinaGoldPremium(timestamp=NOW, premium_pct=2.5, usdcny=7.25, source="SGE"))
    db.commit()

    result = compute_gold_score(db)

    for factor, (lo, hi) in FACTOR_CLAMPS.items():
        if factor in result.factor_scores:
            s = result.factor_scores[factor]
            assert lo <= s <= hi, f"{factor} score {s} outside [{lo}, {hi}]"

    # CFTC and China premium clamps
    if "CFTC投机仓位" in result.factor_scores:
        assert -15 <= result.factor_scores["CFTC投机仓位"] <= 15
    if "中国溢价" in result.factor_scores:
        assert -10 <= result.factor_scores["中国溢价"] <= 10
    if "央行购金" in result.factor_scores:
        assert -10 <= result.factor_scores["央行购金"] <= 10
    if "新闻情绪" in result.factor_scores:
        assert -10 <= result.factor_scores["新闻情绪"] <= 10


# ── Total = sum of factors (pre-100 clamp) ───────────────────────────────────

def test_total_score_equals_sum_of_factors(db):
    """total_score must equal clamp(sum(weighted factor contributions), -100, 100)."""
    _add_all(db)
    result = compute_gold_score(db)
    raw_sum = sum(result.factor_scores.values())
    expected_total = round(_clamp(raw_sum, -100, 100), 2)
    assert result.total_score == pytest.approx(expected_total, abs=0.01)


# ── GoldScoreSnapshot JSON round-trip ────────────────────────────────────────

def test_snapshot_factor_scores_json_roundtrip(db):
    """compute_and_store persists factor_scores as JSON; total_score must match."""
    _add_required(db)
    snapshot = compute_and_store_gold_score(db)

    stored = json.loads(snapshot.factor_scores)
    # compute_and_store wraps with {"scores":..., "details":...}
    assert "scores" in stored
    reconstructed_total = round(_clamp(sum(stored["scores"].values()), -100, 100), 2)
    assert snapshot.total_score == pytest.approx(reconstructed_total, abs=0.01)


def test_snapshot_with_params_stores_scores_and_details(db):
    """compute_and_store_with_params uses the same JSON schema as default scoring."""
    from app.scoring.gold_score import compute_and_store_gold_score_with_params
    _add_required(db)
    params = ScoreParams()
    snapshot = compute_and_store_gold_score_with_params(db, params)

    stored = json.loads(snapshot.factor_scores)
    assert isinstance(stored, dict)
    assert "scores" in stored
    assert "details" in stored


# ── Edge cases: missing / sparse data ────────────────────────────────────────

def test_no_bonus_data_does_not_raise(db):
    """Score computes fine with only required data."""
    _add_required(db)
    result = compute_gold_score(db)
    for key in ["美股分流", "白银/黄金比", "GLD ETF", "搜索热度", "矿业股GDX", "原油WTI", "铜/金比"]:
        assert key not in result.factor_scores


def test_cftc_missing_open_interest_skipped(db):
    """CFTC with open_interest=0 must be skipped (division-by-zero guard)."""
    _add_required(db)
    db.add(CftcPosition(
        market_name="GOLD", contract_market_code="088691", exchange_code="CMX",
        timestamp=NOW, open_interest=0,
        noncommercial_long=100000, noncommercial_short=50000, noncommercial_spreading=0,
        commercial_long=60000, commercial_short=210000, noncommercial_net=50000,
        source="TEST",
    ))
    db.commit()
    result = compute_gold_score(db)
    assert "CFTC投机仓位" not in result.factor_scores


def test_cftc_expired_data_skipped(db):
    """CFTC data older than 35 days (non-TEST source) must be skipped."""
    _add_required(db)
    old_time = datetime.now(timezone.utc) - timedelta(days=40)
    db.add(CftcPosition(
        market_name="GOLD", contract_market_code="088691", exchange_code="CMX",
        timestamp=old_time, open_interest=300000,
        noncommercial_long=180000, noncommercial_short=50000, noncommercial_spreading=0,
        commercial_long=60000, commercial_short=210000, noncommercial_net=130000,
        source="CFTC",
    ))
    db.commit()
    result = compute_gold_score(db)
    assert "CFTC投机仓位" not in result.factor_scores


def test_cftc_decay_between_14_and_35_days(db):
    """CFTC data aged 20 days should apply decay (score < full-strength)."""
    _add_required(db)
    fresh_db = db
    # Fresh CFTC (TEST source → no decay)
    fresh_db.add(CftcPosition(
        market_name="GOLD", contract_market_code="088691", exchange_code="CMX",
        timestamp=NOW, open_interest=300000,
        noncommercial_long=180000, noncommercial_short=50000, noncommercial_spreading=0,
        commercial_long=60000, commercial_short=210000, noncommercial_net=130000,
        source="TEST",
    ))
    fresh_db.commit()
    result_fresh = compute_gold_score(fresh_db)
    full_score = result_fresh.factor_scores.get("CFTC投机仓位")
    assert full_score is not None

    # Now test decay: use real source, 20-day-old timestamp
    engine2 = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine2)
    db2 = sessionmaker(bind=engine2)()
    _add_required(db2)
    db2.add(CftcPosition(
        market_name="GOLD", contract_market_code="088691", exchange_code="CMX",
        timestamp=datetime.now(timezone.utc) - timedelta(days=20),
        open_interest=300000,
        noncommercial_long=180000, noncommercial_short=50000, noncommercial_spreading=0,
        commercial_long=60000, commercial_short=210000, noncommercial_net=130000,
        source="CFTC",
    ))
    db2.commit()
    result_decayed = compute_gold_score(db2)
    decayed_score = result_decayed.factor_scores.get("CFTC投机仓位")
    assert decayed_score is not None
    # Decayed score must be strictly less than full score (same ratio, but decayed)
    assert abs(decayed_score) < abs(full_score)
    db2.close()


def test_china_premium_untrusted_source_skipped(db):
    """SINA-sourced China premium must not score."""
    _add_required(db)
    db.add(ChinaGoldPremium(timestamp=NOW, premium_pct=2.5, source="SINA"))
    db.commit()
    result = compute_gold_score(db)
    assert "中国溢价" not in result.factor_scores
    assert any("中国溢价因子未纳入评分" in f for f in result.risk_flags)


def test_china_premium_expired_skipped(db):
    """China premium older than 10 days (non-TEST) skipped."""
    _add_required(db)
    old = datetime.now(timezone.utc) - timedelta(days=15)
    db.add(ChinaGoldPremium(timestamp=old, premium_pct=2.5, source="SGE"))
    db.commit()
    result = compute_gold_score(db)
    assert "中国溢价" not in result.factor_scores


def test_china_premium_formula(db):
    """premium_pct=1.0 → raw score = (1.0-1.0)*3=0; premium_pct=4.33 → score clamped at 10."""
    _add_required(db)
    db.add(ChinaGoldPremium(timestamp=NOW, premium_pct=4.5, source="SGE"))
    db.commit()
    result = compute_gold_score(db)
    assert "中国溢价" in result.factor_scores
    assert result.factor_details["原始因子分"]["中国溢价"] == pytest.approx(10.0, abs=0.01)


def test_cb_gold_score_insufficient_records(db):
    """央行购金 with only 1 record → skipped (needs ≥2)."""
    _add_required(db)
    db.add(CentralBankGold(
        country="Global", period="2024Q4",
        timestamp=NOW - timedelta(days=30),
        net_change_tonnes=150.0, source="WGC",
    ))
    db.commit()
    result = compute_gold_score(db)
    assert "央行购金" not in result.factor_scores


def test_cb_gold_score_with_records(db):
    """央行购金 with ≥2 trusted records → contributes."""
    _add_required(db)
    for i, period in enumerate(["2024Q2", "2024Q3", "2024Q4"]):
        db.add(CentralBankGold(
            country="Global", period=period,
            timestamp=NOW - timedelta(days=200 - i * 90),
            net_change_tonnes=100.0 + i * 20, source="WGC",
        ))
    db.commit()
    result = compute_gold_score(db)
    assert "央行购金" in result.factor_scores
    assert -10 <= result.factor_scores["央行购金"] <= 10


def test_sentiment_score_no_trusted_data(db):
    """News sentiment from untrusted source → skipped."""
    _add_required(db)
    db.add(NewsSentiment(
        timestamp=NOW, sentiment_score=0.8, source="UNKNOWN",
    ))
    db.commit()
    result = compute_gold_score(db)
    assert "新闻情绪" not in result.factor_scores


def test_sentiment_score_trusted(db):
    """News sentiment from trusted source within 7 days → included."""
    _add_required(db)
    for i in range(5):
        db.add(NewsSentiment(
            timestamp=NOW - timedelta(days=i),
            sentiment_score=0.5, source="GDELT",
        ))
    db.commit()
    result = compute_gold_score(db)
    assert "新闻情绪" in result.factor_scores
    assert -10 <= result.factor_scores["新闻情绪"] <= 10


def test_usdcny_factor_requires_old_row(db):
    """美元人民币 only added when a historical row ≥20 days old exists."""
    _add_required(db)
    # Only one recent row → no old_row → factor absent
    db.add(ChinaGoldPremium(timestamp=NOW, usdcny=7.25, premium_pct=None, source="SGE"))
    db.commit()
    result = compute_gold_score(db)
    assert "美元人民币" not in result.factor_scores


def test_usdcny_factor_with_history(db):
    _add_required(db)
    db.add(ChinaGoldPremium(timestamp=NOW, usdcny=7.30, premium_pct=None, source="SGE"))
    db.add(ChinaGoldPremium(timestamp=NOW - timedelta(days=25), usdcny=7.10, premium_pct=None, source="SGE"))
    db.commit()
    result = compute_gold_score(db)
    assert "美元人民币" in result.factor_scores
    # cny_change = 7.30-7.10 = 0.20 → score = 0.20*2 = 0.40 (positive, CNY weakened)
    assert result.factor_details["原始因子分"]["美元人民币"] == pytest.approx(0.4, abs=0.01)


def test_dollar_factor_zero_base_no_crash():
    """Dollar factor must not crash when dollar values are near-zero (edge guard)."""
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    db2 = sessionmaker(bind=engine)()
    # Insert required series, but with near-zero DOLLAR values
    _add_macro(db2, REAL_RATE,             lambda i: 2.2 - i * 0.005)
    _add_macro(db2, NOMINAL_RATE,          lambda i: 4.2 - i * 0.002)
    _add_macro(db2, INFLATION_EXPECTATION, lambda i: 2.1 + i * 0.002)
    _add_macro(db2, VIX,                   lambda i: 14 + i * 0.04)
    _add_macro(db2, DOLLAR,                lambda i: 0.001 + i * 0.0001)
    _add_macro(db2, FED_RATE,              lambda i: 5.3 - i * 0.001)
    _add_gold(db2)
    db2.commit()
    result = compute_gold_score(db2)
    assert not math.isnan(result.total_score)
    db2.close()


def test_sp500_normalization_to_20_days(db):
    """SP500 change is scaled to 20-day equivalent; result within clamp."""
    _add_all(db)
    result = compute_gold_score(db)
    assert "美股分流" in result.factor_scores
    # SP500 rising → gold bearish → negative score
    assert result.factor_scores["美股分流"] < 0


def test_insufficient_rows_raises(db):
    """<60 aligned rows → ValueError."""
    _add_required(db, days=40)
    with pytest.raises(ValueError, match="Not enough observations"):
        compute_gold_score(db)


def test_no_gold_prices_raises(db):
    with pytest.raises(ValueError, match="No gold price data"):
        compute_gold_score(db)


# ── compute_gold_score_with_params ───────────────────────────────────────────

def test_with_params_produces_same_core_factors(db):
    _add_required(db)
    params = ScoreParams()
    result = compute_gold_score_with_params(db, params)
    for key in ["实际利率", "名义利率", "美元指数", "避险情绪", "通胀预期", "黄金趋势"]:
        assert key in result.factor_scores


def test_with_params_respects_coef(db):
    """Doubling real_rate_coef should produce larger (or equal) absolute 实际利率 score."""
    _add_required(db)
    p1 = ScoreParams(real_rate_coef=30.0)
    p2 = ScoreParams(real_rate_coef=60.0)
    r1 = compute_gold_score_with_params(db, p1)
    r2 = compute_gold_score_with_params(db, p2)
    # Doubled coef → score should be larger in magnitude (unless already at clamp)
    assert abs(r2.factor_scores["实际利率"]) >= abs(r1.factor_scores["实际利率"])


def test_with_params_clamp_bounds_respected(db):
    """ScoreParams clamp fields are used and factor stays within bounds."""
    _add_required(db)
    params = ScoreParams(real_rate_clamp_low=-5.0, real_rate_clamp_high=5.0, real_rate_coef=1000.0)
    result = compute_gold_score_with_params(db, params)
    assert -5.0 <= result.factor_scores["实际利率"] <= 5.0


def test_with_params_total_equals_sum(db):
    _add_required(db)
    result = compute_gold_score_with_params(db, ScoreParams())
    raw_sum = sum(result.factor_scores.values())
    expected = round(_clamp(raw_sum, -100, 100), 2)
    assert result.total_score == pytest.approx(expected, abs=0.01)


def test_with_params_trend_ma_windows(db):
    """trend_ma_long param controls the minimum-data gate."""
    _add_required(db, days=80)
    # trend_ma_long=70 → needs 70 rows, we have 80 → OK
    p = ScoreParams(trend_ma_short=20, trend_ma_long=70)
    result = compute_gold_score_with_params(db, p)
    assert "黄金趋势" in result.factor_scores


def test_with_params_no_bonus_data_no_crash(db):
    _add_required(db)
    result = compute_gold_score_with_params(db, ScoreParams())
    assert not math.isnan(result.total_score)


# ── Direction thresholds ──────────────────────────────────────────────────────

def test_direction_bullish_threshold(db):
    """When total >= 30 → 偏多."""
    from app.scoring.gold_score import _direction
    assert _direction(30) == "偏多"
    assert _direction(31) == "偏多"
    assert _direction(29.9) == "中性"


def test_direction_bearish_threshold(db):
    from app.scoring.gold_score import _direction
    assert _direction(-30) == "偏空"
    assert _direction(-31) == "偏空"
    assert _direction(-29.9) == "中性"


# ── NaN propagation guard ─────────────────────────────────────────────────────

def test_no_nan_in_factor_scores(db):
    _add_all(db)
    result = compute_gold_score(db)
    for k, v in result.factor_scores.items():
        assert not math.isnan(v), f"NaN in factor {k}"
    assert not math.isnan(result.total_score)
