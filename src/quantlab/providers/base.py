from abc import ABC, abstractmethod
from datetime import date

from quantlab.types import PriceBar


class MarketDataProvider(ABC):
    @abstractmethod
    def get_daily_bars(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> list[PriceBar]:
        raise NotImplementedError