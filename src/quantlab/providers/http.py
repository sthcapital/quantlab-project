from datetime import date

import requests

from quantlab.providers.base import MarketDataProvider
from quantlab.types import PriceBar


class HttpMarketDataProvider(MarketDataProvider):
    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def get_daily_bars(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> list[PriceBar]:
        raise NotImplementedError(
            "HTTP provider scaffold only. Implement provider-specific request parsing here."
        )