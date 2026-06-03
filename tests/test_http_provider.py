
from quantlab.providers.http import HttpMarketDataProvider


def test_http_provider_initializes():
    provider = HttpMarketDataProvider("https://example.com", "test-key")

    assert provider.base_url == "https://example.com"
    assert provider.api_key == "test-key"