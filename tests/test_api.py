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
