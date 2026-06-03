from datetime import date, datetime

from ib_insync import IB, Stock

from quantlab.providers.base import MarketDataProvider
from quantlab.types import PriceBar


class IbkrMarketDataProvider(MarketDataProvider):
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
    ) -> None:
        self.host = host
        self.port = port
        self.client_id = client_id

    def get_daily_bars(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> list[PriceBar]:
        ib = IB()

        try:
            ib.connect(self.host, self.port, clientId=self.client_id, timeout=10)

            contract = Stock(symbol, "SMART", "USD")
            ib.qualifyContracts(contract)

            duration_days = max((end_date - start_date).days + 5, 10)

            bars = ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=f"{duration_days} D",
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            )

            result: list[PriceBar] = []

            for bar in bars:
                if isinstance(bar.date, datetime):
                    as_of = bar.date.date()
                else:
                    as_of = bar.date

                if as_of < start_date or as_of > end_date:
                    continue

                result.append(
                    PriceBar(
                        symbol=symbol,
                        as_of=as_of,
                        open=float(bar.open),
                        high=float(bar.high),
                        low=float(bar.low),
                        close=float(bar.close),
                        volume=float(bar.volume),
                    )
                )

            result.sort(key=lambda bar: bar.as_of)
            return result
        finally:
            if ib.isConnected():
                ib.disconnect()
