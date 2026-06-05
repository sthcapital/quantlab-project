"""IBKR TWS market data provider via ib_insync.

Two usage modes:

    1. Per-request connections (simple, good for single-symbol scripts):

        provider = IbkrProvider(host, port, client_id)
        bars = provider.get_daily_bars("AAPL", start, end)

    2. Persistent shared connection (fast for universe scans, avoids repeated
       TWS handshakes):

        with IbkrProvider(host, port, client_id) as provider:
            results = run_universe_backtest(provider, symbols, ...)

Pacing rules (IBKR enforces max 60 historical requests per 10-minute window):
    - 2-second sleep after every fetch (_INTER_REQUEST_SLEEP)
    - Up to 2 retries with 31-second wait on empty/pacing responses
    - Bar caching: parquet written after first fetch, read on subsequent calls
"""

from __future__ import annotations

import logging
import socket
import time
from datetime import date
from pathlib import Path
from typing import Sequence

from ib_insync import IB, Stock

from quantlab.providers.base import Bar, MarketDataProvider

logger = logging.getLogger(__name__)

_INTER_REQUEST_SLEEP: float = 2.0   # seconds between every historical data call
_PACING_RETRY_SLEEP: float = 31.0   # seconds to wait before retrying after an empty response
_MAX_RETRIES: int = 2               # max retry attempts on empty response


# ── Pre-flight check ───────────────────────────────────────────────────────────

def ping_tws(
    host: str = "127.0.0.1",
    port: int = 7497,
    timeout: float = 5.0,
) -> bool:
    """
    Return True if TWS / IB Gateway is reachable at host:port.

    Uses a raw TCP probe — no IBKR handshake required. Safe to call before
    creating an IB() connection.

    Usage in scripts::

        if not ping_tws(args.host, args.port):
            raise SystemExit(f"TWS not reachable at {args.host}:{args.port}")
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# ── Provider ───────────────────────────────────────────────────────────────────

class IbkrProvider(MarketDataProvider):
    """Fetches daily OHLCV bars and spot prices from Interactive Brokers TWS."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        timeout: int = 10,
        spot_client_id: int | None = None,
    ) -> None:
        """
        Args:
            host:           TWS / IB Gateway host.
            port:           TWS port (7497 paper, 7496 live) or Gateway port
                            (4002 paper, 4001 live).
            client_id:      TWS client ID for historical data requests.
            timeout:        Connection timeout in seconds.
            spot_client_id: TWS client ID for get_spot_price. Defaults to
                            client_id + 50. Must differ from client_id to avoid
                            collision when both are open simultaneously.
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        self.timeout = timeout
        self.spot_client_id = spot_client_id if spot_client_id is not None else client_id + 50
        self._shared_ib: IB | None = None   # persistent connection shared across requests

    # ── Persistent connection management ──────────────────────────────────────

    def connect(self) -> "IbkrProvider":
        """Open a persistent connection. Returns self for method chaining."""
        if self._shared_ib is None:
            self._shared_ib = IB()
        if not self._shared_ib.isConnected():
            self._shared_ib.connect(
                self.host, self.port,
                clientId=self.client_id,
                timeout=self.timeout,
            )
            logger.info(
                "IbkrProvider connected to %s:%d (client_id=%d)",
                self.host, self.port, self.client_id,
            )
        return self

    def disconnect(self) -> None:
        """Close the persistent connection if open."""
        if self._shared_ib and self._shared_ib.isConnected():
            self._shared_ib.disconnect()
            logger.info("IbkrProvider disconnected")
        self._shared_ib = None

    def __enter__(self) -> "IbkrProvider":
        return self.connect()

    def __exit__(self, *_) -> None:
        self.disconnect()

    def _get_ib(self) -> tuple[IB, bool]:
        """Return (ib_instance, is_temporary).

        Returns the shared persistent connection when available (i.e. inside a
        ``with`` block or after an explicit ``connect()`` call).  Otherwise
        creates and connects a temporary IB() that the caller must disconnect
        after use.
        """
        if self._shared_ib is not None and self._shared_ib.isConnected():
            return self._shared_ib, False
        ib = IB()
        ib.connect(self.host, self.port, clientId=self.client_id, timeout=self.timeout)
        return ib, True

    # ── Bar cache ──────────────────────────────────────────────────────────────

    def _cache_path(self, symbol: str) -> Path:
        from quantlab.storage import DATA_PROCESSED, ensure_dirs
        ensure_dirs()
        return DATA_PROCESSED / f"{symbol}_bars.parquet"

    def _load_from_cache(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> list[Bar] | None:
        """
        Return bars from parquet cache if the file fully covers [start_date, end_date].

        Returns None on any miss (file absent, partial coverage, read error).
        """
        path = self._cache_path(symbol)
        if not path.exists():
            return None
        try:
            import pyarrow.parquet as pq
            rows = pq.read_table(path).to_pydict()
            if not rows.get("date"):
                return None
            file_dates = [date.fromisoformat(d) for d in rows["date"]]
            if min(file_dates) > start_date or max(file_dates) < end_date:
                logger.debug(
                    "%s: cache miss — file covers %s..%s, need %s..%s",
                    symbol, min(file_dates), max(file_dates), start_date, end_date,
                )
                return None
            bars = [
                Bar(
                    as_of=date.fromisoformat(rows["date"][i]),
                    open=rows["open"][i],
                    high=rows["high"][i],
                    low=rows["low"][i],
                    close=rows["close"][i],
                    volume=rows["volume"][i],
                )
                for i in range(len(rows["date"]))
                if start_date <= date.fromisoformat(rows["date"][i]) <= end_date
            ]
            logger.debug("%s: cache hit — %d bars from %s", symbol, len(bars), path.name)
            return bars
        except Exception as exc:
            logger.debug("%s: cache read failed — %s", symbol, exc)
            return None

    def _save_to_cache(self, symbol: str, bars: list[Bar]) -> None:
        """Write bars to parquet. Silently skips if pyarrow is unavailable."""
        try:
            from quantlab.storage import save_bars_parquet
            save_bars_parquet(symbol, bars)
        except Exception as exc:
            logger.debug("%s: cache write failed — %s", symbol, exc)

    # ── Duration helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _duration_str(start_date: date, end_date: date) -> str:
        """
        Compute the IBKR durationStr that covers the full [start_date, end_date] window.

        Adds a 7-day buffer for weekends and holidays so the request always
        reaches back far enough.
        """
        calendar_days = (end_date - start_date).days + 7
        if calendar_days <= 365:
            return f"{calendar_days} D"
        years = (calendar_days // 365) + 1
        return f"{years} Y"

    @staticmethod
    def _end_date_time(end_date: date) -> str:
        """
        Return the IBKR endDateTime string for the given end_date.

        IBKR convention: empty string means "now" (use when end_date is today
        or in the future). For historical end dates, supply the explicit
        timestamp so the fetch window is anchored to the correct day.
        """
        if end_date >= date.today():
            return ""
        return end_date.strftime("%Y%m%d 23:59:59")

    # ── Fetch with pacing ──────────────────────────────────────────────────────

    def _fetch_raw_bars(
        self,
        ib: IB,
        contract,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> list:
        """
        Request historical bars with retry on empty/pacing responses.

        IBKR sometimes returns an empty list when the pacing limit is hit rather
        than raising an explicit error. Retrying after _PACING_RETRY_SLEEP
        seconds handles this transparently.
        """
        duration = self._duration_str(start_date, end_date)
        end_dt = self._end_date_time(end_date)

        for attempt in range(_MAX_RETRIES + 1):
            if attempt > 0:
                logger.warning(
                    "%s: empty response on attempt %d/%d — "
                    "sleeping %.0fs (possible pacing violation)",
                    symbol, attempt, _MAX_RETRIES, _PACING_RETRY_SLEEP,
                )
                time.sleep(_PACING_RETRY_SLEEP)

            raw_bars = ib.reqHistoricalData(
                contract,
                endDateTime=end_dt,
                durationStr=duration,
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            )

            if raw_bars:
                return raw_bars

        logger.warning("%s: no bars returned after %d attempts", symbol, _MAX_RETRIES + 1)
        return []

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_daily_bars(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> Sequence[Bar]:
        """
        Return daily OHLCV bars for symbol between start_date and end_date.

        Checks the parquet cache first. On a cache miss, fetches from IBKR,
        writes to cache, and returns. A 2-second pacing sleep is inserted after
        every IBKR request.

        Args:
            symbol:     Ticker symbol (e.g. "AAPL").
            start_date: First bar date (inclusive).
            end_date:   Last bar date (inclusive).

        Returns:
            List of Bar objects, sorted by date ascending.

        Raises:
            RuntimeError: If the contract cannot be qualified by TWS.
        """
        cached = self._load_from_cache(symbol, start_date, end_date)
        if cached is not None:
            return cached

        ib, is_temporary = self._get_ib()
        try:
            contract = Stock(symbol, "SMART", "USD")
            qualified = ib.qualifyContracts(contract)
            if not qualified:
                raise RuntimeError(f"Could not qualify contract for {symbol}")

            raw_bars = self._fetch_raw_bars(ib, qualified[0], symbol, start_date, end_date)

            # Respect IBKR pacing: sleep between consecutive requests
            time.sleep(_INTER_REQUEST_SLEEP)

            start_str = start_date.isoformat()
            end_str = end_date.isoformat()

            bars = [
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

            if bars:
                self._save_to_cache(symbol, bars)

            return bars

        finally:
            if is_temporary and ib.isConnected():
                ib.disconnect()

    def get_spot_price(self, symbol: str) -> float | None:
        """
        Return the current market price for symbol.

        Uses spot_client_id (default: client_id + 50) to avoid colliding with
        a concurrent historical data connection on client_id.
        """
        ib = IB()
        try:
            ib.connect(
                self.host, self.port,
                clientId=self.spot_client_id,
                timeout=self.timeout,
            )
            ib.reqMarketDataType(3)  # fall back to delayed data if live unavailable
            stock = Stock(symbol, "SMART", "USD")
            qualified = ib.qualifyContracts(stock)
            if not qualified:
                return None
            ticker = ib.reqTickers(qualified[0])[0]
            for candidate in [ticker.marketPrice(), ticker.last, ticker.close]:
                if candidate is not None and candidate == candidate and candidate > 0:
                    return float(candidate)
            # Last resort: most recent historical close
            raw = ib.reqHistoricalData(
                qualified[0],
                endDateTime="",
                durationStr="5 D",
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            )
            return float(raw[-1].close) if raw else None
        finally:
            if ib.isConnected():
                ib.disconnect()
