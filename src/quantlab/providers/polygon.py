"""
quantlab.providers.polygon — Polygon.io market data provider.

Implements MarketDataProvider for daily OHLCV bars, plus Polygon-specific
endpoints needed for institutional breadth analysis:

    get_daily_bars()      — historical OHLCV for one symbol
    get_grouped_daily()   — ALL US stocks in a single API call (breadth)
    get_previous_close()  — latest previous-day close
    get_ticker_details()  — market cap, sector, industry

API key is read from the POLYGON_API_KEY environment variable.
Grouped daily results are cached to data/processed/breadth/{date}.parquet
so the same date is never re-fetched.

Rate limits: Polygon free tier = 5 req/min.  Paid tiers are much higher.
The provider adds a configurable inter-request sleep (default 0.25s).
"""

from __future__ import annotations

import os
import time
import logging
from datetime import date, datetime
from typing import Sequence

import requests

from quantlab.providers.base import Bar, MarketDataProvider

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.polygon.io"


class PolygonProvider(MarketDataProvider):
    """
    Polygon.io REST API provider.

    Args:
        api_key:        Polygon API key.  Defaults to ``POLYGON_API_KEY`` env var.
        request_sleep:  Seconds to sleep between requests (respects rate limits).
    """

    def __init__(
        self,
        api_key: str | None = None,
        request_sleep: float = 0.25,
    ) -> None:
        self.api_key      = api_key or os.environ.get("POLYGON_API_KEY", "")
        self.request_sleep = request_sleep
        self._session = requests.Session()
        if not self.api_key:
            logger.warning("POLYGON_API_KEY not set — API calls will fail with 403")

    # ── Internal HTTP helper ───────────────────────────────────────────────────

    def _get(self, path: str, params: dict | None = None) -> dict:
        """Execute a GET request and return parsed JSON.  Raises on HTTP error."""
        url = f"{_BASE_URL}{path}"
        p   = dict(params or {})
        p["apiKey"] = self.api_key
        resp = self._session.get(url, params=p, timeout=30)
        resp.raise_for_status()
        time.sleep(self.request_sleep)
        return resp.json()

    @staticmethod
    def _bar_from_agg(item: dict, bar_date: date) -> Bar:
        """Convert a Polygon aggregates dict to a Bar."""
        return Bar(
            as_of  = bar_date,
            open   = float(item.get("o", 0)),
            high   = float(item.get("h", 0)),
            low    = float(item.get("l", 0)),
            close  = float(item.get("c", 0)),
            volume = float(item.get("v", 0)),
        )

    # ── MarketDataProvider interface ───────────────────────────────────────────

    def get_daily_bars(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> Sequence[Bar]:
        """
        Fetch daily OHLCV bars from /v2/aggs/ticker/{symbol}/range/1/day.

        Returns bars sorted ascending by date.
        """
        path = (
            f"/v2/aggs/ticker/{symbol}/range/1/day"
            f"/{start_date.isoformat()}/{end_date.isoformat()}"
        )
        data = self._get(path, {"adjusted": "true", "sort": "asc", "limit": 50000})

        bars: list[Bar] = []
        for item in data.get("results", []):
            t_ms     = item.get("t", 0)
            bar_date = datetime.utcfromtimestamp(t_ms / 1000).date() if t_ms else start_date
            if start_date <= bar_date <= end_date:
                bars.append(self._bar_from_agg(item, bar_date))

        return bars

    # ── Polygon-specific endpoints ─────────────────────────────────────────────

    def get_grouped_daily(self, trade_date: date) -> dict[str, Bar]:
        """
        Fetch ALL US stocks for a single trading day in one request.

        Uses /v2/aggs/grouped/locale/us/market/stocks/{date}.
        Results are cached to data/processed/breadth/{date}.parquet.

        Returns:
            Dict {symbol: Bar} — typically 12,000–13,000 entries.
        """
        cached = self._load_breadth_cache(trade_date)
        if cached is not None:
            logger.debug("Breadth cache hit for %s (%d symbols)", trade_date, len(cached))
            return cached

        path = f"/v2/aggs/grouped/locale/us/market/stocks/{trade_date.isoformat()}"
        data = self._get(path, {"adjusted": "true"})

        result: dict[str, Bar] = {}
        for item in data.get("results", []):
            symbol = item.get("T", "")
            if not symbol:
                continue
            t_ms     = item.get("t", 0)
            bar_date = datetime.utcfromtimestamp(t_ms / 1000).date() if t_ms else trade_date
            result[symbol] = self._bar_from_agg(item, bar_date)

        logger.info("Grouped daily %s: %d symbols fetched", trade_date, len(result))
        self._save_breadth_cache(trade_date, result)
        return result

    def get_previous_close(self, symbol: str) -> Bar | None:
        """
        Fetch the most recent previous trading day's close via
        /v2/aggs/ticker/{symbol}/prev.

        Returns None when no data is available.
        """
        try:
            data    = self._get(f"/v2/aggs/ticker/{symbol}/prev", {"adjusted": "true"})
            results = data.get("results", [])
            if not results:
                return None
            item     = results[0]
            t_ms     = item.get("t", 0)
            bar_date = datetime.utcfromtimestamp(t_ms / 1000).date() if t_ms else date.today()
            return self._bar_from_agg(item, bar_date)
        except Exception as exc:
            logger.warning("get_previous_close(%s) failed: %s", symbol, exc)
            return None

    def get_ticker_details(self, symbol: str) -> dict:
        """
        Fetch reference data via /v3/reference/tickers/{symbol}.

        Returns a dict with keys such as: market_cap, sic_description,
        primary_exchange, name, locale, type, currency_name.
        Returns {} on failure.
        """
        try:
            data = self._get(f"/v3/reference/tickers/{symbol}")
            return data.get("results", {})
        except Exception as exc:
            logger.warning("get_ticker_details(%s) failed: %s", symbol, exc)
            return {}

    # ── Breadth cache helpers ──────────────────────────────────────────────────

    def _breadth_cache_dir(self):
        from quantlab.storage import DATA_PROCESSED
        d = DATA_PROCESSED / "breadth"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _breadth_cache_path(self, trade_date: date):
        return self._breadth_cache_dir() / f"{trade_date.isoformat()}.parquet"

    def _load_breadth_cache(self, trade_date: date) -> dict[str, Bar] | None:
        path = self._breadth_cache_path(trade_date)
        if not path.exists():
            return None
        try:
            import pyarrow.parquet as pq
            rows = pq.read_table(path).to_pydict()
            result: dict[str, Bar] = {}
            for i in range(len(rows["symbol"])):
                bar_date = date.fromisoformat(rows["date"][i])
                result[rows["symbol"][i]] = Bar(
                    as_of  = bar_date,
                    open   = rows["open"][i],
                    high   = rows["high"][i],
                    low    = rows["low"][i],
                    close  = rows["close"][i],
                    volume = rows["volume"][i],
                )
            return result
        except Exception as exc:
            logger.debug("Breadth cache read failed for %s: %s", trade_date, exc)
            return None

    def _save_breadth_cache(self, trade_date: date, data: dict[str, Bar]) -> None:
        if not data:
            return
        path = self._breadth_cache_path(trade_date)
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            table = pa.table({
                "symbol": list(data.keys()),
                "date":   [b.as_of.isoformat() for b in data.values()],
                "open":   [b.open   for b in data.values()],
                "high":   [b.high   for b in data.values()],
                "low":    [b.low    for b in data.values()],
                "close":  [b.close  for b in data.values()],
                "volume": [b.volume for b in data.values()],
            })
            pq.write_table(table, path)
            logger.debug("Breadth cache saved: %s (%d symbols)", path.name, len(data))
        except Exception as exc:
            logger.warning("Breadth cache write failed: %s", exc)
