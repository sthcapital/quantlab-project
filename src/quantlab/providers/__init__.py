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
        "ibkr"                          — Interactive Brokers TWS (live data)
        "flatfile" / "flat_file"        — S3 flat files + local Parquet cache
                                          (bulk-loads the date range once;
                                           fastest for universe scans)
        "alpha_vantage" / "http"        — Alpha Vantage HTTP API
        "mock"                          — Deterministic mock for testing
        "polygon"                       — Polygon.io REST API

    Example::

        provider = create_market_data_provider("flatfile")
        bars = provider.get_daily_bars("AAPL", start_date, end_date)
    """
    normalized = name.strip().lower()

    if normalized == "ibkr":
        if not _IBKR_AVAILABLE:
            raise RuntimeError(
                "ib_insync is not installed. Run: pip install ib_insync"
            )
        return IbkrProvider(**kwargs)

    if normalized in {"flatfile", "flat_file", "flat-file"}:
        # No TWS connection is opened — not even if --with-options is passed.
        # Options scoring uses MassiveOptionsProvider (Polygon S3) exclusively.
        from quantlab.providers.flat_files import FlatFileMarketDataProvider
        return FlatFileMarketDataProvider(**kwargs)

    if normalized in {"alpha_vantage", "alphavantage", "http"}:
        return HttpMarketDataProvider(**kwargs)

    if normalized == "mock":
        return MockMarketDataProvider(**kwargs)

    if normalized == "polygon":
        from quantlab.providers.polygon import PolygonProvider
        return PolygonProvider(**kwargs)

    if normalized == "factset":
        from quantlab.providers.factset import FactSetProvider
        return FactSetProvider(**kwargs)

    raise ValueError(
        f"Unknown market data provider: '{name}'. "
        f"Valid choices: ibkr, flatfile, alpha_vantage, mock, polygon, factset"
    )


__all__ = [
    "Bar",
    "MarketDataProvider",
    "IbkrProvider",
    "HttpMarketDataProvider",
    "MockMarketDataProvider",
    "create_market_data_provider",
]
