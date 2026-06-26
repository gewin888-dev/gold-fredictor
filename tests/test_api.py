from fastapi.testclient import TestClient

from app.database import get_db
from app.main import app
from conftest import insert_cftc_position, insert_gold_prices, insert_required_macro_observations


def test_health_endpoint():
    client = TestClient(app)

    assert client.get("/health").json() == {"status": "ok"}


def test_data_health_endpoint(db_session):
    def override_db():
        yield db_session

    app.dependency_overrides[get_db] = override_db
    try:
        client = TestClient(app)
        response = client.get("/health/data")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["status"] == "error"


def test_score_endpoints_use_database_dependency(db_session):
    insert_required_macro_observations(db_session)
    insert_gold_prices(db_session)

    def override_db():
        yield db_session

    app.dependency_overrides[get_db] = override_db
    try:
        client = TestClient(app)

        computed = client.post("/score/compute")
        latest = client.get("/score/latest")
        macro = client.get("/macro/latest")
    finally:
        app.dependency_overrides.clear()

    assert computed.status_code == 200
    assert computed.json()["direction"] in {"偏多", "中性", "偏空"}
    assert latest.json()["ok"] is True
    assert macro.json()["ok"] is True
    assert "GOLDAMGBD228NLBM" in macro.json()["data"]


def test_feishu_test_endpoint_skips_without_webhook(db_session, monkeypatch):
    from app.notifications import feishu

    class _NoWebhook:
        feishu_webhook_url = ""
        feishu_secret = ""

    monkeypatch.setattr(feishu, "get_settings", lambda: _NoWebhook())

    def override_db():
        yield db_session

    app.dependency_overrides[get_db] = override_db
    try:
        client = TestClient(app)
        response = client.post("/notify/feishu/test")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["skipped"] is True


def test_cftc_latest_endpoint(db_session):
    insert_cftc_position(db_session)

    def override_db():
        yield db_session

    app.dependency_overrides[get_db] = override_db
    try:
        client = TestClient(app)
        response = client.get("/positions/cftc/latest")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["data"]["contract_market_code"] == "088691"
    assert payload["data"]["noncommercial_net"] == 130000


def test_score_backtest_endpoint_returns_reason_when_missing_data(db_session):
    def override_db():
        yield db_session

    app.dependency_overrides[get_db] = override_db
    try:
        client = TestClient(app)
        response = client.get("/backtest/score?horizon_days=10")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["ok"] is False
    assert response.json()["horizon_days"] == 10


def test_upcoming_events_endpoint(db_session):
    from app.events.calendar import load_sample_macro_events

    load_sample_macro_events(db_session)

    def override_db():
        yield db_session

    app.dependency_overrides[get_db] = override_db
    try:
        client = TestClient(app)
        response = client.get("/events/upcoming?days_ahead=40")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert len(payload["data"]) >= 1


def test_auto_optimize_settings_endpoint_uses_safe_db_settings(db_session):
    def override_db():
        yield db_session

    app.dependency_overrides[get_db] = override_db
    try:
        client = TestClient(app)
        initial = client.get("/settings/auto-optimize")
        updated = client.post(
            "/settings/auto-optimize",
            json={
                "AUTO_OPTIMIZE_SCORE_PARAMS": True,
                "AUTO_ACTIVATE_OPTIMIZED_PARAMS": False,
                "FRED_API_KEY": True,
            },
        )
        after = client.get("/settings/auto-optimize")
    finally:
        app.dependency_overrides.clear()

    assert initial.status_code == 200
    assert initial.json()["ok"] is True
    assert updated.status_code == 200
    assert updated.json()["ok"] is True
    assert "FRED_API_KEY" in updated.json()["rejected_keys"]
    assert updated.json()["settings"]["AUTO_OPTIMIZE_SCORE_PARAMS"] is True
    assert after.json()["settings"]["AUTO_OPTIMIZE_SCORE_PARAMS"] is True
    assert "health" in after.json()


def test_config_audit_masks_secrets_and_marks_db_overrides(db_session):
    def override_db():
        yield db_session

    app.dependency_overrides[get_db] = override_db
    try:
        client = TestClient(app)
        client.post(
            "/settings/auto-optimize",
            json={"AUTO_SELF_HEALING_ENABLED": False},
        )
        response = client.get("/config/audit")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    items = {row["key"]: row for row in payload["items"]}
    assert items["FRED_API_KEY"]["is_secret"] is True
    assert items["FRED_API_KEY"]["value"] in {"", "***"} or "***" in items["FRED_API_KEY"]["value"]
    assert items["AUTO_SELF_HEALING_ENABLED"]["source"] == "database"
    assert items["AUTO_SELF_HEALING_ENABLED"]["hot_reload"] is True
    assert "AUTO_SELF_HEALING_ENABLED" in payload["management"]["safe_hot_reload"]


def test_score_factor_registry_exposes_scoring_and_gray_status():
    client = TestClient(app)
    response = client.get("/score/factors/registry")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    factors = {row["name"]: row for row in payload["data"]}
    assert factors["ETF资金流"]["scored"] is True
    assert factors["ETF资金流"]["optimizable"] is True
    assert factors["COMEX库存"]["scored"] is False
    assert factors["COMEX库存"]["inactive_reason"]
    assert factors["地缘风险"]["scored"] is False


def test_external_indicator_rejects_unknown_indicator(db_session):
    def override_db():
        yield db_session

    app.dependency_overrides[get_db] = override_db
    try:
        client = TestClient(app)
        response = client.post(
            "/external/indicators",
            json={
                "indicator_id": "UNKNOWN_FACTOR",
                "timestamp": "2026-06-14T00:00:00Z",
                "value": 1.0,
                "source": "TEST",
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 400
    assert "Unknown indicator_id" in response.json()["detail"]


def test_external_indicator_rejects_invalid_timestamp(db_session):
    def override_db():
        yield db_session

    app.dependency_overrides[get_db] = override_db
    try:
        client = TestClient(app)
        response = client.post(
            "/external/indicators",
            json={
                "indicator_id": "GEO_RISK_INTENSITY",
                "timestamp": "not-a-date",
                "value": 1.0,
                "source": "TEST",
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 400
    assert "ISO-8601" in response.json()["detail"]


def test_deactivate_score_params_endpoint_restores_default(db_session):
    from app.models import ScoreParamsVersion

    db_session.add(
        ScoreParamsVersion(
            version="candidate_for_deactivate_test",
            params_json="{}",
            hit_rate=0.6,
            sample_count=130,
            backtest_horizon_days=20,
            is_active=True,
        )
    )
    db_session.commit()

    def override_db():
        yield db_session

    app.dependency_overrides[get_db] = override_db
    try:
        client = TestClient(app)
        response = client.post("/score/params/deactivate")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert response.json()["version"] == "default"
    assert db_session.query(ScoreParamsVersion).filter_by(is_active=True).count() == 0
