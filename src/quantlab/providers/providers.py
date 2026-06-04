"""HTTP (Alpha Vantage) and mock market data providers."""

from __future__ import annotations

import json
import random
from datetime import date, timedelta
from typing import Sequence

import requests

from quantlab.providers.base import Bar, MarketDataProvider


class HttpMarketDataProvider(MarketDataProvider):
    """Alpha Vantage (or compatible) HTTP provider."""

    def __init__(
        self,
        base_url: str = "https://www.alphavantage.co",
        api_key: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def get_daily_bars(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> Sequence[Bar]:
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": "full",
            "datatype": "json",
        }
        if self.api_key:
            params["apikey"] = self.api_key

        resp = requests.get(f"{self.base_url}/query", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        series = data.get("Time Series (Daily)", {})
        bars = []
        for date_str, values in sorted(series.items()):
            bar_date = date.fromisoformat(date_str)
            if start_date <= bar_date <= end_date:
                bars.append(
                    Bar(
                        as_of=bar_date,
                        open=float(values["1. open"]),
                        high=float(values["2. high"]),
                        low=float(values["3. low"]),
                        close=float(values["4. close"]),
                        volume=float(values["6. volume"]),
                    )
                )
        return bars


class MockMarketDataProvider(MarketDataProvider):
    """Deterministic mock provider for offline testing."""

    def __init__(self, seed: int = 42, start_price: float = 100.0) -> None:
        self.seed = seed
        self.start_price = start_price

    def get_daily_bars(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> Sequence[Bar]:
        rng = random.Random(self.seed + hash(symbol))
        bars = []
        current = start_date
        price = self.start_price

        while current <= end_date:
            # Skip weekends
            if current.weekday() >= 5:
                current += timedelta(days=1)
                continue

            change = rng.gauss(0.0003, 0.015)
            open_p = price
            close_p = price * (1 + change)
            high_p = max(open_p, close_p) * (1 + abs(rng.gauss(0, 0.005)))
            low_p = min(open_p, close_p) * (1 - abs(rng.gauss(0, 0.005)))
            volume = rng.randint(1_000_000, 50_000_000)

            bars.append(
                Bar(
                    as_of=current,
                    open=round(open_p, 4),
                    high=round(high_p, 4),
                    low=round(low_p, 4),
                    close=round(close_p, 4),
                    volume=float(volume),
                )
            )
            price = close_p
            current += timedelta(days=1)

        return bars

    def get_spot_price(self, symbol: str) -> float | None:
        rng = random.Random(self.seed + hash(symbol))
        return round(self.start_price * (1 + rng.gauss(0.05, 0.1)), 2)
