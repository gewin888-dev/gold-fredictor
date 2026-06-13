from datetime import datetime, timezone

from app.scoring.gold_score import compute_and_store_gold_score, compute_gold_score
from app.models import CftcPosition
from conftest import insert_cftc_position, insert_gold_prices, insert_required_macro_observations


def test_compute_gold_score_returns_expected_shape(db_session):
    insert_required_macro_observations(db_session)
    insert_gold_prices(db_session)

    result = compute_gold_score(db_session)

    assert -100 <= result.total_score <= 100
    assert result.direction in {"偏多", "中性", "偏空"}
    assert {"实际利率", "名义利率", "美元指数", "避险情绪", "通胀预期", "黄金趋势"}.issubset(result.factor_scores)
    assert result.risk_flags
    assert "不构成投资建议" not in result.summary
    assert "仅用于数据分析和风险提示" in result.summary


def test_compute_gold_score_includes_cftc_when_available(db_session):
    insert_required_macro_observations(db_session)
    insert_gold_prices(db_session)
    insert_cftc_position(db_session)

    result = compute_gold_score(db_session)

    assert "CFTC投机仓位" in result.factor_scores
    assert any("CFTC 非商业净持仓" in flag for flag in result.risk_flags)


def test_compute_gold_score_skips_sample_cftc_source(db_session):
    insert_required_macro_observations(db_session)
    insert_gold_prices(db_session)
    db_session.add(
        CftcPosition(
            market_name="GOLD - COMMODITY EXCHANGE INC.",
            contract_market_code="088691",
            exchange_code="CMX",
            timestamp=datetime(2025, 3, 21, tzinfo=timezone.utc),
            open_interest=300000,
            noncommercial_long=180000,
            noncommercial_short=50000,
            noncommercial_spreading=15000,
            commercial_long=60000,
            commercial_short=210000,
            noncommercial_net=130000,
            source="SAMPLE",
        )
    )
    db_session.commit()

    result = compute_gold_score(db_session)

    assert "CFTC投机仓位" not in result.factor_scores
    assert any("CFTC投机仓位因子未纳入评分" in flag for flag in result.risk_flags)


def test_compute_and_store_gold_score_persists_snapshot(db_session):
    insert_required_macro_observations(db_session)
    insert_gold_prices(db_session)

    snapshot = compute_and_store_gold_score(db_session)

    assert snapshot.id is not None
    assert snapshot.source == "rule_v2"
    assert snapshot.direction in {"偏多", "中性", "偏空"}


def test_compute_gold_score_requires_required_series(db_session):
    insert_gold_prices(db_session)
    try:
        compute_gold_score(db_session)
    except ValueError as exc:
        assert "Missing FRED data" in str(exc)
    else:
        raise AssertionError("Expected missing data error")
