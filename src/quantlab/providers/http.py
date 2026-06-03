import os
from datetime import date, datetime

import requests

from quantlab.providers.base import MarketDataProvider
from quantlab.types import PriceBar


class HttpMarketDataProvider(MarketDataProvider):
    def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
        self.base_url = (base_url or "https://www.alphavantage.co").rstrip("/")
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY", "")

    def get_daily_bars(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> list[PriceBar]:
        if not self.api_key:
            raise ValueError("Missing ALPHA_VANTAGE_API_KEY")

        response = requests.get(
            f"{self.base_url}/query",
            params={
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "outputsize": "compact",
                "apikey": self.api_key,
            },
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()

        if "Error Message" in payload:
            raise ValueError(f'Alpha Vantage error: {payload["Error Message"]}')

        if "Note" in payload:
            raise ValueError(f'Alpha Vantage note: {payload["Note"]}')

        if "Information" in payload:
            raise ValueError(f'Alpha Vantage information: {payload["Information"]}')

        series = payload.get("Time Series (Daily)")
        if not series:
            raise ValueError(f"Unexpected Alpha Vantage response keys: {list(payload.keys())}")

        bars: list[PriceBar] = []

        for as_of_str, values in series.items():
            as_of = datetime.strptime(as_of_str, "%Y-%m-%d").date()

            if as_of < start_date or as_of > end_date:
                continue

            bars.append(
                PriceBar(
                    symbol=symbol,
                    as_of=as_of,
                    open=float(values["1. open"]),
                    high=float(values["2. high"]),
                    low=float(values["3. low"]),
                    close=float(values["4. close"]),
                    volume=float(values["5. volume"]),
                )
            )

        bars.sort(key=lambda bar: bar.as_of)
        return bars