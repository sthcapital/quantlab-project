from datetime import date
from typing import Any

from quantlab.providers.base import MarketDataProvider
from quantlab.types import PriceBar


class MarketDataService:
    def __init__(self, provider: MarketDataProvider) -> None:
        self.provider = provider

    def get_daily_bars(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> list[PriceBar]:
        return self.provider.get_daily_bars(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )

    def get_daily_bars_as_rows(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> list[dict[str, Any]]:
        bars = self.get_daily_bars(symbol, start_date, end_date)

        return [
            {
                "symbol": bar.symbol,
                "as_of": bar.as_of.isoformat(),
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            for bar in bars
        ]