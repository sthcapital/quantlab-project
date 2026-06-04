"""IBKR TWS market data provider via ib_insync."""

from __future__ import annotations

from datetime import date
from typing import Sequence

from ib_insync import IB, Stock

from quantlab.providers.base import Bar, MarketDataProvider


class IbkrProvider(MarketDataProvider):
    """Fetches daily bars from Interactive Brokers TWS."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        timeout: int = 10,
    ) -> None:
        self.host = host
        self.port = port
        self.client_id = client_id
        self.timeout = timeout

    def get_daily_bars(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> Sequence[Bar]:
        ib = IB()
        try:
            ib.connect(self.host, self.port, clientId=self.client_id, timeout=self.timeout)
            contract = Stock(symbol, "SMART", "USD")
            qualified = ib.qualifyContracts(contract)
            if not qualified:
                raise RuntimeError(f"Could not qualify contract for {symbol}")

            raw_bars = ib.reqHistoricalData(
                qualified[0],
                endDateTime="",
                durationStr="365 D",
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            )

            start_str = start_date.isoformat()
            end_str = end_date.isoformat()

            return [
                Bar(
                    as_of=date.fromisoformat(str(b.date)[:10]),
                    open=float(b.open),
                    high=float(b.high),
                    low=float(b.low),
                    close=float(b.close),
                    volume=float(b.volume),
                )
                for b in raw_bars
                if start_str <= str(b.date)[:10] <= end_str
            ]
        finally:
            if ib.isConnected():
                ib.disconnect()

    def get_spot_price(self, symbol: str) -> float | None:
        ib = IB()
        try:
            ib.connect(self.host, self.port, clientId=self.client_id + 50, timeout=self.timeout)
            ib.reqMarketDataType(3)  # delayed data fallback
            stock = Stock(symbol, "SMART", "USD")
            qualified = ib.qualifyContracts(stock)
            if not qualified:
                return None
            ticker = ib.reqTickers(qualified[0])[0]
            for candidate in [ticker.marketPrice(), ticker.last, ticker.close]:
                if candidate is not None and candidate == candidate and candidate > 0:
                    return float(candidate)
            # fall back to last historical close
            bars = ib.reqHistoricalData(
                qualified[0], endDateTime="", durationStr="5 D",
                barSizeSetting="1 day", whatToShow="TRADES", useRTH=True, formatDate=1,
            )
            return float(bars[-1].close) if bars else None
        finally:
            if ib.isConnected():
                ib.disconnect()
