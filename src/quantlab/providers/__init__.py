"""Provider factory — the single entry point for creating data providers."""

from __future__ import annotations

from quantlab.providers.base import Bar, MarketDataProvider
from quantlab.providers.providers import HttpMarketDataProvider, MockMarketDataProvider

try:
    from quantlab.providers.ibkr import IbkrProvider
    _IBKR_AVAILABLE = True
except ImportError:
    _IBKR_AVAILABLE = False
    IbkrProvider = None  # type: ignore


def create_market_data_provider(name: str, **kwargs) -> MarketDataProvider:
    """
    Create and return a market data provider by name.

    Supported names:
        "ibkr"                          — Interactive Brokers TWS
        "alpha_vantage" / "http"        — Alpha Vantage HTTP API
        "mock"                          — Deterministic mock for testing

    Example::

        provider = create_market_data_provider("ibkr", host="127.0.0.1", port=7497, client_id=1)
        bars = provider.get_daily_bars("AAPL", start_date, end_date)
    """
    normalized = name.strip().lower()

    if normalized == "ibkr":
        if not _IBKR_AVAILABLE:
            raise RuntimeError(
                "ib_insync is not installed. Run: pip install ib_insync"
            )
        return IbkrProvider(**kwargs)

    if normalized in {"alpha_vantage", "alphavantage", "http"}:
        return HttpMarketDataProvider(**kwargs)

    if normalized == "mock":
        return MockMarketDataProvider(**kwargs)

    if normalized == "polygon":
        from quantlab.providers.polygon import PolygonProvider
        return PolygonProvider(**kwargs)

    raise ValueError(
        f"Unknown market data provider: '{name}'. "
        f"Valid choices: ibkr, alpha_vantage, mock"
    )


__all__ = [
    "Bar",
    "MarketDataProvider",
    "IbkrProvider",
    "HttpMarketDataProvider",
    "MockMarketDataProvider",
    "create_market_data_provider",
]
