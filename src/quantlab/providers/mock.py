from datetime import date, timedelta

from quantlab.providers.base import MarketDataProvider
from quantlab.types import PriceBar


class MockMarketDataProvider(MarketDataProvider):
    def get_daily_bars(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> list[PriceBar]:
        bars: list[PriceBar] = []
        current = start_date
        close_price = 100.0

        while current <= end_date:
            bars.append(
                PriceBar(
                    symbol=symbol,
                    as_of=current,
                    open=close_price - 1.0,
                    high=close_price + 1.5,
                    low=close_price - 2.0,
                    close=close_price,
                    volume=1_000_000.0,
                )
            )
            current += timedelta(days=1)
            close_price += 1.0

        return bars