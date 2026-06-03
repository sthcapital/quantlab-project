from quantlab.providers.factory import create_market_data_provider
from quantlab.providers.http import HttpMarketDataProvider
from quantlab.providers.ibkr import IbkrMarketDataProvider


def test_factory_creates_alpha_vantage_provider():
    provider = create_market_data_provider(
        "alpha_vantage",
        base_url="https://example.com",
        api_key="test-key",
    )

    assert isinstance(provider, HttpMarketDataProvider)


def test_factory_creates_ibkr_provider():
    provider = create_market_data_provider(
        "ibkr",
        host="127.0.0.1",
        port=7497,
        client_id=7,
    )

    assert isinstance(provider, IbkrMarketDataProvider)
    assert provider.host == "127.0.0.1"
    assert provider.port == 7497
    assert provider.client_id == 7


def test_factory_rejects_unknown_provider():
    try:
        create_market_data_provider("unknown")
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "Unknown market data provider" in str(exc)
