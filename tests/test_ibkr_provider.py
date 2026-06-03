from datetime import date, datetime

from quantlab.providers.ibkr import IbkrMarketDataProvider


class DummyBar:
    def __init__(self, as_of, open_, high, low, close, volume):
        self.date = as_of
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume


class DummyIB:
    def __init__(self):
        self.connected = False

    def connect(self, host, port, clientId, timeout):
        self.connected = True

    def qualifyContracts(self, contract):
        return [contract]

    def reqHistoricalData(
        self,
        contract,
        endDateTime,
        durationStr,
        barSizeSetting,
        whatToShow,
        useRTH,
        formatDate,
    ):
        return [
            DummyBar(date(2026, 1, 1), 100, 101, 99, 100.5, 1000),
            DummyBar(date(2026, 1, 2), 101, 102, 100, 101.5, 1100),
            DummyBar(datetime(2026, 1, 3, 0, 0), 102, 103, 101, 102.5, 1200),
        ]

    def isConnected(self):
        return self.connected

    def disconnect(self):
        self.connected = False


def test_ibkr_provider_parses_daily_bars(monkeypatch):
    monkeypatch.setattr("quantlab.providers.ibkr.IB", DummyIB)

    provider = IbkrMarketDataProvider(host="127.0.0.1", port=7497, client_id=7)
    bars = provider.get_daily_bars(
        symbol="AAPL",
        start_date=date(2026, 1, 2),
        end_date=date(2026, 1, 3),
    )

    assert len(bars) == 2
    assert bars[0].as_of.isoformat() == "2026-01-02"
    assert bars[0].close == 101.5
    assert bars[1].as_of.isoformat() == "2026-01-03"
    assert bars[1].close == 102.5


def test_ibkr_provider_initializes():
    provider = IbkrMarketDataProvider(host="127.0.0.1", port=7497, client_id=9)

    assert provider.host == "127.0.0.1"
    assert provider.port == 7497
    assert provider.client_id == 9
