from datetime import date

from quantlab.providers.mock import MockMarketDataProvider
from quantlab.services.market_data_service import MarketDataService


def test_mock_provider_returns_bars():
    provider = MockMarketDataProvider()

    bars = provider.get_daily_bars(
        symbol="AAPL",
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 3),
    )

    assert len(bars) == 3
    assert bars[0].symbol == "AAPL"
    assert bars[0].close == 100.0
    assert bars[-1].close == 102.0


def test_market_data_service_returns_rows():
    provider = MockMarketDataProvider()
    service = MarketDataService(provider)

    rows = service.get_daily_bars_as_rows(
        symbol="MSFT",
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 2),
    )

    assert len(rows) == 2
    assert rows[0]["symbol"] == "MSFT"
    assert rows[0]["as_of"] == "2026-01-01"