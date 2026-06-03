from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass(frozen=True)
class PriceBar:
    symbol: str
    as_of: date
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None