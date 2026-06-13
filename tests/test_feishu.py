from app.notifications import feishu
from app.models import GoldScoreSnapshot
from datetime import datetime, timezone


class EmptySettings:
    feishu_webhook_url = ""
    feishu_secret = ""


def test_send_text_message_skips_when_webhook_missing(monkeypatch):
    monkeypatch.setattr(feishu, "get_settings", lambda: EmptySettings())

    result = feishu.send_text_message("hello")

    assert result["ok"] is True
    assert result["skipped"] is True
    assert "FEISHU_WEBHOOK_URL" in result["reason"]


def test_build_score_alert_text_contains_analysis_fields():
    snapshot = GoldScoreSnapshot(
        timestamp=datetime(2026, 6, 11, tzinfo=timezone.utc),
        total_score=43.2,
        direction="偏多",
        factor_scores='{"黄金趋势": 20, "CFTC投机仓位": 15}',
        risk_flags='["CFTC 非商业净持仓占总持仓约 54.0%。"]',
        summary="黄金多空评分为 43.2，方向为偏多。该结果仅用于数据分析和风险提示。",
        source="TEST",
    )

    text = feishu.build_score_alert_text(
        snapshot,
        {"status": "ok"},
        [{"timestamp": "2026-06-16", "name": "美国 CPI 数据", "importance": "high"}],
    )

    assert "黄金走势监控报告" in text
    assert "数据健康: ok" in text
    assert "黄金趋势" in text
    assert "CFTC 非商业净持仓" in text
    assert "美国 CPI 数据" in text
    assert "不构成投资建议" in text


def test_send_score_alert_with_health_skips_without_webhook(monkeypatch):
    monkeypatch.setattr(feishu, "get_settings", lambda: EmptySettings())
    snapshot = GoldScoreSnapshot(
        timestamp=datetime(2026, 6, 11, tzinfo=timezone.utc),
        total_score=0,
        direction="中性",
        factor_scores="{}",
        risk_flags="[]",
        summary="测试。",
        source="TEST",
    )

    result = feishu.send_score_alert_with_health(snapshot, {"status": "ok"})

    assert result["ok"] is True
    assert result["skipped"] is True
