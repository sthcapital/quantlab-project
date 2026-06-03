from quantlab.providers.base import MarketDataProvider
from quantlab.providers.http import HttpMarketDataProvider
from quantlab.providers.ibkr import IbkrMarketDataProvider


def create_market_data_provider(name: str, **kwargs) -> MarketDataProvider:
    normalized = name.strip().lower()

    if normalized in {"alpha_vantage", "alphavantage", "http"}:
        return HttpMarketDataProvider(**kwargs)

    if normalized == "ibkr":
        return IbkrMarketDataProvider(**kwargs)

    raise ValueError(f"Unknown market data provider: {name}")
