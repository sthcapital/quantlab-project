"""Base protocol for all market data providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Sequence


@dataclass(frozen=True)
class Bar:
    """A single daily OHLCV bar."""

    as_of: date
    open: float
    high: float
    low: float
    close: float
    volume: float

    def pct_change(self, prev: "Bar") -> float:
        return (self.close / prev.close) - 1.0

    def true_range(self, prev: "Bar") -> float:
        return max(
            self.high - self.low,
            abs(self.high - prev.close),
            abs(self.low - prev.close),
        )


class MarketDataProvider(ABC):
    """Abstract base class every provider must implement."""

    @abstractmethod
    def get_daily_bars(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> Sequence[Bar]:
        """Return daily OHLCV bars in ascending date order."""
        ...

    def get_spot_price(self, symbol: str) -> float | None:
        """Return the most recent close price. Override for live providers."""
        return None
