"""
S3 flat-file pipeline — downloads full-universe daily files once and caches
as Parquet so subsequent access is instant with no API calls.

Confirmed S3 layout (files.massive.com / bucket: flatfiles):
    us_stocks_sip/day_aggs_v1/{year}/{month:02d}/{YYYY-MM-DD}.csv.gz
        ~11,000 symbols, ~214 KB compressed per day
    us_options_opra/day_aggs_v1/{year}/{month:02d}/{YYYY-MM-DD}.csv.gz
        ~265,000 contracts, ~3 MB compressed per day

CSV schema (both files):
    ticker, volume, open, close, high, low, window_start, transactions
    window_start: nanoseconds since Unix epoch (market open time)

Credentials (env vars):
    POLYGON_S3_ACCESS_KEY_ID  — S3 access key
    POLYGON_API_KEY           — used as S3 secret key
"""

from __future__ import annotations

import csv
import gzip
import io
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from quantlab.providers.base import Bar

logger = logging.getLogger(__name__)

_ENDPOINT = "https://files.massive.com"
_BUCKET = "flatfiles"
_STOCKS_PREFIX = "us_stocks_sip/day_aggs_v1"
_OPTIONS_PREFIX = "us_options_opra/day_aggs_v1"


class FlatFileProvider:
    """
    Downloads Polygon/Massive S3 flat files and caches them as Parquet.

    Each date file is downloaded once. All subsequent reads come from the
    local Parquet cache at data/processed/flat_files/.

    Parquet cache files:
        stocks_{YYYY-MM-DD}.parquet  — all US stocks OHLCV for one day
        options_{YYYY-MM-DD}.parquet — all US options OHLCV with parsed fields

    Non-trading days (weekends, holidays) produce a NoSuchKey S3 error that
    callers should handle by skipping or falling back.
    """

    def __init__(
        self,
        access_key: str | None = None,
        secret_key: str | None = None,
        endpoint: str = _ENDPOINT,
        bucket: str = _BUCKET,
    ) -> None:
        import os
        self.access_key = access_key or os.getenv("POLYGON_S3_ACCESS_KEY_ID", "")
        self.secret_key = secret_key or os.getenv("POLYGON_API_KEY", "")
        self.endpoint = endpoint
        self.bucket = bucket
        self._s3 = None  # lazy-initialised boto3 client

    # ── S3 client (lazy) ───────────────────────────────────────────────────────

    def _get_s3(self):
        if self._s3 is None:
            import boto3
            self._s3 = boto3.client(
                "s3",
                endpoint_url=self.endpoint,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
            )
        return self._s3

    # ── S3 key helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def stocks_s3_key(d: date) -> str:
        return f"{_STOCKS_PREFIX}/{d.year}/{d.month:02d}/{d.isoformat()}.csv.gz"

    @staticmethod
    def options_s3_key(d: date) -> str:
        return f"{_OPTIONS_PREFIX}/{d.year}/{d.month:02d}/{d.isoformat()}.csv.gz"

    # ── Local Parquet cache paths ───────────────────────────────────────────────

    def _cache_dir(self) -> Path:
        from quantlab.storage import DATA_PROCESSED
        d = DATA_PROCESSED / "flat_files"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def stocks_cache_path(self, d: date) -> Path:
        return self._cache_dir() / f"stocks_{d.isoformat()}.parquet"

    def options_cache_path(self, d: date) -> Path:
        return self._cache_dir() / f"options_{d.isoformat()}.parquet"

    # ── Raw S3 download ────────────────────────────────────────────────────────

    def _download_csv_gz(self, key: str) -> list[dict]:
        """Download a CSV.gz from S3 and return all rows as a list of dicts."""
        obj = self._get_s3().get_object(Bucket=self.bucket, Key=key)
        raw = obj["Body"].read()
        with gzip.open(io.BytesIO(raw), "rt", newline="") as f:
            return list(csv.DictReader(f))

    # ── Stocks pipeline ────────────────────────────────────────────────────────

    def download_stocks_day(self, d: date) -> dict[str, Bar]:
        """
        Download the full-universe stocks flat file for ``d`` and return
        ``{ticker: Bar}``.

        On cache hit (parquet already exists), the S3 download is skipped.
        Raises ``botocore.exceptions.ClientError`` (NoSuchKey) for non-trading
        days — callers should catch and skip.

        Side effect: saves data/processed/flat_files/stocks_{d}.parquet.
        """
        cache = self.stocks_cache_path(d)
        if cache.exists():
            result = self._load_stocks_parquet(d)
            logger.debug("Stocks flat file %s: %d symbols (cache hit)", d, len(result))
            return result

        key = self.stocks_s3_key(d)
        logger.info("Downloading stocks flat file: %s", key)
        rows = self._download_csv_gz(key)
        self._save_stocks_parquet(d, rows)
        result = _rows_to_bars(rows, d)
        logger.info(
            "Stocks flat file %s: %d symbols | parquet %s",
            d, len(result), _fmt_size(cache.stat().st_size),
        )
        return result

    def _save_stocks_parquet(self, d: date, rows: list[dict]) -> None:
        if not rows:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        def _f(key: str) -> list[float]:
            return [float(r[key]) for r in rows]

        def _i(key: str) -> list[int]:
            return [int(r[key]) for r in rows]

        table = pa.table({
            "ticker":       [r["ticker"] for r in rows],
            "volume":       _f("volume"),
            "open":         _f("open"),
            "close":        _f("close"),
            "high":         _f("high"),
            "low":          _f("low"),
            "window_start": _i("window_start"),
            "transactions": _i("transactions"),
        })
        pq.write_table(table, self.stocks_cache_path(d))

    def _load_stocks_parquet(self, d: date) -> dict[str, Bar]:
        import pyarrow.parquet as pq
        path = self.stocks_cache_path(d)
        if not path.exists():
            return {}
        df = pq.read_table(
            path, columns=["ticker", "open", "high", "low", "close", "volume"]
        ).to_pydict()
        return {
            df["ticker"][i]: Bar(
                as_of=d,
                open=float(df["open"][i]),
                high=float(df["high"][i]),
                low=float(df["low"][i]),
                close=float(df["close"][i]),
                volume=float(df["volume"][i]),
            )
            for i in range(len(df["ticker"]))
        }

    # ── Options pipeline ───────────────────────────────────────────────────────

    def download_options_day(self, d: date) -> list[dict]:
        """
        Download the full options flat file for ``d``, parse each OCC ticker,
        and return all records as a list of dicts.

        On cache hit the S3 download is skipped.
        Raises ``botocore.exceptions.ClientError`` (NoSuchKey) on non-trading days.

        Each returned dict has keys:
            ticker, underlying, expiry, strike, option_type,
            volume, open, close, high, low, window_start, transactions

        Side effect: saves data/processed/flat_files/options_{d}.parquet.
        """
        cache = self.options_cache_path(d)
        if cache.exists():
            records = self._load_options_parquet(d)
            logger.debug("Options flat file %s: %d records (cache hit)", d, len(records))
            return records

        key = self.options_s3_key(d)
        logger.info("Downloading options flat file: %s", key)
        rows = self._download_csv_gz(key)
        records = _parse_option_rows(rows)
        self._save_options_parquet(d, records)
        logger.info(
            "Options flat file %s: %d records | parquet %s",
            d, len(records), _fmt_size(cache.stat().st_size),
        )
        return records

    def _save_options_parquet(self, d: date, records: list[dict]) -> None:
        if not records:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table({
            "ticker":       [r["ticker"]      for r in records],
            "underlying":   [r["underlying"]  for r in records],
            "expiry":       [r["expiry"]      for r in records],
            "strike":       [r["strike"]      for r in records],
            "option_type":  [r["option_type"] for r in records],
            "volume":       [r["volume"]      for r in records],
            "open":         [r["open"]        for r in records],
            "close":        [r["close"]       for r in records],
            "high":         [r["high"]        for r in records],
            "low":          [r["low"]         for r in records],
            "window_start": [r["window_start"]for r in records],
            "transactions": [r["transactions"]for r in records],
        })
        pq.write_table(table, self.options_cache_path(d))

    def _load_options_parquet(
        self, d: date, underlying: Optional[str] = None
    ) -> list[dict]:
        import pyarrow.parquet as pq
        path = self.options_cache_path(d)
        if not path.exists():
            return []
        filters = [("underlying", "=", underlying)] if underlying else None
        df = pq.read_table(path, filters=filters).to_pydict()
        n = len(df.get("ticker", []))
        keys = list(df.keys())
        return [{k: df[k][i] for k in keys} for i in range(n)]

    # ── High-level access ──────────────────────────────────────────────────────

    def get_stocks_bars(
        self, symbol: str, start_date: date, end_date: date
    ) -> list[Bar]:
        """
        Return daily Bar history for ``symbol`` over [start_date, end_date].

        Iterates day by day. Cached parquet files are read without any S3
        calls. Missing files (weekends, holidays) are silently skipped.
        New files are downloaded and cached on first access.
        """
        bars: list[Bar] = []
        current = start_date
        while current <= end_date:
            try:
                day_bars = self.download_stocks_day(current)
                if symbol in day_bars:
                    bars.append(day_bars[symbol])
            except Exception:
                pass  # non-trading day or S3 error — skip
            current += timedelta(days=1)
        return bars

    def get_grouped_daily(self, d: date) -> dict[str, Bar]:
        """
        Return ``{ticker: Bar}`` for all US stocks on date ``d``.

        Equivalent to PolygonProvider.get_grouped_daily() but sourced from the
        S3 flat file. First call downloads ~214 KB; subsequent calls read from
        the local Parquet cache (sub-millisecond).
        """
        return self.download_stocks_day(d)

    def get_options_chain_from_flatfile(
        self, symbol: str, d: date
    ) -> list[dict]:
        """
        Return all option records for underlying ``symbol`` on date ``d``.

        Downloads and caches the full options file on first call for each date.
        Subsequent calls filter the cached Parquet with a predicate pushdown
        (reads only the matching rows from disk).

        Each record dict has: ticker, underlying, expiry, strike, option_type,
        volume, open, close, high, low, window_start, transactions.
        """
        cache = self.options_cache_path(d)
        if not cache.exists():
            self.download_options_day(d)
        return self._load_options_parquet(d, underlying=symbol)


# ── Module-level helpers ───────────────────────────────────────────────────────

def _rows_to_bars(rows: list[dict], d: date) -> dict[str, Bar]:
    bars: dict[str, Bar] = {}
    for row in rows:
        try:
            bars[row["ticker"]] = Bar(
                as_of=d,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
        except (KeyError, ValueError):
            continue
    return bars


def _parse_option_rows(rows: list[dict]) -> list[dict]:
    """Parse raw CSV rows into enriched option records with OCC fields split out."""
    from quantlab.providers.massive_options import MassiveOptionsProvider
    records: list[dict] = []
    for row in rows:
        try:
            ticker = row["ticker"]
            sym, expiry, opt_type, strike = MassiveOptionsProvider.parse_option_ticker(ticker)
            records.append({
                "ticker":       ticker,
                "underlying":   sym,
                "expiry":       expiry.isoformat(),
                "strike":       float(strike),
                "option_type":  opt_type,
                "volume":       float(row["volume"]),
                "open":         float(row["open"]),
                "close":        float(row["close"]),
                "high":         float(row["high"]),
                "low":          float(row["low"]),
                "window_start": int(row["window_start"]),
                "transactions": int(row["transactions"]),
            })
        except (KeyError, ValueError):
            continue
    return records


def _fmt_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes //= 1024
    return f"{n_bytes:.1f} GB"
