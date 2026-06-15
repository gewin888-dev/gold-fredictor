from app.data import fred_client
from app.data.fred_client import FredClient


class _Response:
    def raise_for_status(self):
        return None

    def json(self):
        return {"observations": []}


def test_fred_client_verifies_ssl_by_default(monkeypatch):
    calls = []

    class _Settings:
        fred_api_key = "test-key"
        fred_verify_ssl = True

    monkeypatch.setattr(fred_client, "get_settings", lambda: _Settings())
    monkeypatch.setattr(
        fred_client.requests,
        "get",
        lambda *args, **kwargs: calls.append(kwargs) or _Response(),
    )

    FredClient()._get("series/observations", {"series_id": "DGS10"})

    assert calls[0]["verify"] is True


def test_fred_client_allows_explicit_ssl_compat_mode(monkeypatch):
    calls = []

    class _Settings:
        fred_api_key = "test-key"
        fred_verify_ssl = False

    monkeypatch.setattr(fred_client, "get_settings", lambda: _Settings())
    monkeypatch.setattr(
        fred_client.requests,
        "get",
        lambda *args, **kwargs: calls.append(kwargs) or _Response(),
    )

    FredClient()._get("series/observations", {"series_id": "DGS10"})

    assert calls[0]["verify"] is False
