from datetime import date

from quantlab.providers.http import HttpMarketDataProvider


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_http_provider_initializes():
    provider = HttpMarketDataProvider("https://example.com", "test-key")

    assert provider.base_url == "https://example.com"
    assert provider.api_key == "test-key"


def test_http_provider_parses_daily_bars(monkeypatch):
    payload = {
        "Time Series (Daily)": {
            "2026-01-03": {
                "1. open": "102.0",
                "2. high": "103.5",
                "3. low": "101.0",
                "4. close": "103.0",
                "5. volume": "1200000",
            },
            "2026-01-02": {
                "1. open": "101.0",
                "2. high": "102.5",
                "3. low": "100.0",
                "4. close": "102.0",
                "5. volume": "1100000",
            },
        }
    }

    def fake_get(*args, **kwargs):
        return DummyResponse(payload)

    monkeypatch.setattr("quantlab.providers.http.requests.get", fake_get)

    provider = HttpMarketDataProvider("https://example.com", "test-key")
    bars = provider.get_daily_bars(
        symbol="AAPL",
        start_date=date(2026, 1, 2),
        end_date=date(2026, 1, 3),
    )

    assert len(bars) == 2
    assert bars[0].as_of.isoformat() == "2026-01-02"
    assert bars[0].close == 102.0
    assert bars[1].as_of.isoformat() == "2026-01-03"
    assert bars[1].close == 103.0