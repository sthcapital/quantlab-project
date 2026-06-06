"""
tests/test_flat_files.py — Unit tests for FlatFileProvider.

All tests mock boto3 and the filesystem — no S3 credentials or network needed.
"""

from __future__ import annotations

import csv
import gzip
import io
import sys
import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quantlab.providers.base import Bar
from quantlab.providers.flat_files import (
    FlatFileProvider,
    _rows_to_bars,
    _parse_option_rows,
    _fmt_size,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

SAMPLE_STOCKS_ROWS = [
    {"ticker": "AAPL", "volume": "55000000", "open": "185.00", "close": "187.50",
     "high": "188.10", "low": "184.20", "window_start": "1746072000000000000",
     "transactions": "750000"},
    {"ticker": "MSFT", "volume": "20000000", "open": "415.00", "close": "420.00",
     "high": "421.50", "low": "413.00", "window_start": "1746072000000000000",
     "transactions": "310000"},
    {"ticker": "NVDA", "volume": "40000000", "open": "880.00", "close": "895.00",
     "high": "900.00", "low": "878.00", "window_start": "1746072000000000000",
     "transactions": "520000"},
]

SAMPLE_OPTIONS_ROWS = [
    {"ticker": "O:AAPL250516C00185000", "volume": "1200", "open": "5.40",
     "close": "6.10", "high": "6.30", "low": "5.20",
     "window_start": "1746072000000000000", "transactions": "48"},
    {"ticker": "O:AAPL250516P00180000", "volume": "800", "open": "3.10",
     "close": "2.90", "high": "3.20", "low": "2.80",
     "window_start": "1746072000000000000", "transactions": "31"},
    {"ticker": "O:MSFT250516C00420000", "volume": "500", "open": "8.00",
     "close": "8.50", "high": "8.70", "low": "7.90",
     "window_start": "1746072000000000000", "transactions": "22"},
    {"ticker": "O:BAD", "volume": "10", "open": "1.0", "close": "1.0",
     "high": "1.0", "low": "1.0", "window_start": "1746072000000000000",
     "transactions": "1"},   # malformed ticker — should be skipped
]

TEST_DATE = date(2025, 5, 1)


def _make_gz_bytes(rows: list[dict]) -> bytes:
    """Encode a list of dicts as a CSV.gz bytes object."""
    if not rows:
        return b""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    gz_buf = io.BytesIO()
    with gzip.open(gz_buf, "wt") as f:
        f.write(buf.getvalue())
    return gz_buf.getvalue()


def _make_provider(tmp_path: Path) -> FlatFileProvider:
    """Return a FlatFileProvider whose cache_dir is a temp directory."""
    p = FlatFileProvider(access_key="test_key", secret_key="test_secret")
    # Patch _cache_dir to use tmp_path so tests don't touch the real fs
    p._cache_dir = lambda: tmp_path  # type: ignore[method-assign]
    return p


# ══════════════════════════════════════════════════════════════════════════════
# Module-level helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestHelpers:

    def test_rows_to_bars_basic(self):
        bars = _rows_to_bars(SAMPLE_STOCKS_ROWS, TEST_DATE)
        assert "AAPL" in bars
        assert bars["AAPL"].close == pytest.approx(187.50)
        assert bars["AAPL"].as_of == TEST_DATE
        assert bars["AAPL"].volume == pytest.approx(55_000_000)

    def test_rows_to_bars_skips_bad_rows(self):
        bad_rows = [{"ticker": "X", "open": "abc", "close": "1", "high": "1",
                     "low": "1", "volume": "0"}]
        bars = _rows_to_bars(bad_rows, TEST_DATE)
        assert bars == {}

    def test_parse_option_rows_extracts_fields(self):
        records = _parse_option_rows(SAMPLE_OPTIONS_ROWS)
        # Malformed O:BAD row should be dropped
        assert len(records) == 3
        aapl_call = next(r for r in records if r["ticker"] == "O:AAPL250516C00185000")
        assert aapl_call["underlying"] == "AAPL"
        assert aapl_call["option_type"] == "C"
        assert aapl_call["strike"] == pytest.approx(185.0)
        assert aapl_call["expiry"] == "2025-05-16"
        assert aapl_call["volume"] == pytest.approx(1200.0)

    def test_parse_option_rows_put(self):
        records = _parse_option_rows(SAMPLE_OPTIONS_ROWS)
        put = next(r for r in records if r["ticker"] == "O:AAPL250516P00180000")
        assert put["option_type"] == "P"
        assert put["strike"] == pytest.approx(180.0)

    def test_fmt_size_bytes(self):
        assert _fmt_size(512) == "512.0 B"

    def test_fmt_size_kilobytes(self):
        assert _fmt_size(2048) == "2.0 KB"

    def test_fmt_size_megabytes(self):
        assert _fmt_size(3 * 1024 * 1024) == "3.0 MB"


# ══════════════════════════════════════════════════════════════════════════════
# S3 key helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestS3Keys:

    def test_stocks_key_format(self):
        key = FlatFileProvider.stocks_s3_key(date(2025, 5, 1))
        assert key == "us_stocks_sip/day_aggs_v1/2025/05/2025-05-01.csv.gz"

    def test_stocks_key_zero_pads_month(self):
        key = FlatFileProvider.stocks_s3_key(date(2025, 1, 15))
        assert key == "us_stocks_sip/day_aggs_v1/2025/01/2025-01-15.csv.gz"

    def test_options_key_format(self):
        key = FlatFileProvider.options_s3_key(date(2025, 12, 31))
        assert key == "us_options_opra/day_aggs_v1/2025/12/2025-12-31.csv.gz"

    def test_keys_include_year_and_month_dirs(self):
        key = FlatFileProvider.stocks_s3_key(date(2026, 6, 6))
        parts = key.split("/")
        assert parts[-3] == "2026"
        assert parts[-2] == "06"
        assert parts[-1] == "2026-06-06.csv.gz"


# ══════════════════════════════════════════════════════════════════════════════
# download_stocks_day
# ══════════════════════════════════════════════════════════════════════════════

class TestDownloadStocksDay:

    def test_downloads_and_returns_bars(self, tmp_path):
        p = _make_provider(tmp_path)
        gz_bytes = _make_gz_bytes(SAMPLE_STOCKS_ROWS)

        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {"Body": io.BytesIO(gz_bytes)}
        p._s3 = mock_s3

        bars = p.download_stocks_day(TEST_DATE)

        assert len(bars) == 3
        assert "AAPL" in bars
        assert bars["MSFT"].open == pytest.approx(415.0)
        mock_s3.get_object.assert_called_once()

    def test_creates_parquet_cache(self, tmp_path):
        p = _make_provider(tmp_path)
        gz_bytes = _make_gz_bytes(SAMPLE_STOCKS_ROWS)
        p._s3 = MagicMock()
        p._s3.get_object.return_value = {"Body": io.BytesIO(gz_bytes)}

        p.download_stocks_day(TEST_DATE)

        cache = p.stocks_cache_path(TEST_DATE)
        assert cache.exists()
        assert cache.stat().st_size > 0

    def test_cache_hit_skips_s3(self, tmp_path):
        p = _make_provider(tmp_path)
        gz_bytes = _make_gz_bytes(SAMPLE_STOCKS_ROWS)
        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {"Body": io.BytesIO(gz_bytes)}
        p._s3 = mock_s3

        # First call — downloads
        p.download_stocks_day(TEST_DATE)
        assert mock_s3.get_object.call_count == 1

        # Second call — cache hit, no S3 call
        result = p.download_stocks_day(TEST_DATE)
        assert mock_s3.get_object.call_count == 1   # unchanged
        assert "AAPL" in result

    def test_roundtrip_bar_values(self, tmp_path):
        p = _make_provider(tmp_path)
        gz_bytes = _make_gz_bytes(SAMPLE_STOCKS_ROWS)
        p._s3 = MagicMock()
        p._s3.get_object.return_value = {"Body": io.BytesIO(gz_bytes)}

        bars_first = p.download_stocks_day(TEST_DATE)

        # Load from cache
        bars_cached = p._load_stocks_parquet(TEST_DATE)

        assert bars_first["AAPL"].close == pytest.approx(bars_cached["AAPL"].close)
        assert bars_first["NVDA"].high  == pytest.approx(bars_cached["NVDA"].high)


# ══════════════════════════════════════════════════════════════════════════════
# download_options_day
# ══════════════════════════════════════════════════════════════════════════════

class TestDownloadOptionsDay:

    def test_downloads_and_parses(self, tmp_path):
        p = _make_provider(tmp_path)
        gz_bytes = _make_gz_bytes(SAMPLE_OPTIONS_ROWS)
        p._s3 = MagicMock()
        p._s3.get_object.return_value = {"Body": io.BytesIO(gz_bytes)}

        records = p.download_options_day(TEST_DATE)

        # Malformed O:BAD row dropped
        assert len(records) == 3
        underlyings = {r["underlying"] for r in records}
        assert "AAPL" in underlyings
        assert "MSFT" in underlyings

    def test_creates_parquet_cache(self, tmp_path):
        p = _make_provider(tmp_path)
        gz_bytes = _make_gz_bytes(SAMPLE_OPTIONS_ROWS)
        p._s3 = MagicMock()
        p._s3.get_object.return_value = {"Body": io.BytesIO(gz_bytes)}

        p.download_options_day(TEST_DATE)

        assert p.options_cache_path(TEST_DATE).exists()

    def test_cache_hit_skips_s3(self, tmp_path):
        p = _make_provider(tmp_path)
        gz_bytes = _make_gz_bytes(SAMPLE_OPTIONS_ROWS)
        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {"Body": io.BytesIO(gz_bytes)}
        p._s3 = mock_s3

        p.download_options_day(TEST_DATE)
        assert mock_s3.get_object.call_count == 1

        p.download_options_day(TEST_DATE)
        assert mock_s3.get_object.call_count == 1   # still 1 — served from cache

    def test_record_schema(self, tmp_path):
        p = _make_provider(tmp_path)
        gz_bytes = _make_gz_bytes(SAMPLE_OPTIONS_ROWS)
        p._s3 = MagicMock()
        p._s3.get_object.return_value = {"Body": io.BytesIO(gz_bytes)}

        records = p.download_options_day(TEST_DATE)
        rec = records[0]
        for key in ("ticker", "underlying", "expiry", "strike", "option_type",
                    "volume", "open", "close", "high", "low",
                    "window_start", "transactions"):
            assert key in rec, f"Missing key: {key}"


# ══════════════════════════════════════════════════════════════════════════════
# get_stocks_bars
# ══════════════════════════════════════════════════════════════════════════════

class TestGetStocksBars:

    def test_returns_bars_for_trading_days(self, tmp_path):
        p = _make_provider(tmp_path)

        call_count = 0
        def fake_download(d):
            nonlocal call_count
            call_count += 1
            # Only "trading days" have data
            if d.weekday() < 5:
                return {"AAPL": Bar(as_of=d, open=185.0, high=188.0, low=184.0,
                                    close=187.0, volume=1e7)}
            raise Exception("NoSuchKey")

        p.download_stocks_day = fake_download  # type: ignore[method-assign]

        bars = p.get_stocks_bars("AAPL", date(2025, 5, 1), date(2025, 5, 7))

        # May 1–7 2025: Thu, Fri, Sat, Sun, Mon, Tue, Wed → 5 trading days
        assert len(bars) == 5
        assert all(b.close == pytest.approx(187.0) for b in bars)

    def test_skips_symbol_not_in_day(self, tmp_path):
        p = _make_provider(tmp_path)

        def fake_download(d):
            return {"MSFT": Bar(as_of=d, open=415.0, high=421.0, low=413.0,
                                close=420.0, volume=2e7)}

        p.download_stocks_day = fake_download  # type: ignore[method-assign]

        bars = p.get_stocks_bars("AAPL", date(2025, 5, 1), date(2025, 5, 2))
        assert bars == []   # AAPL not in the fake data

    def test_returns_empty_for_all_failures(self, tmp_path):
        p = _make_provider(tmp_path)
        p.download_stocks_day = lambda d: (_ for _ in ()).throw(Exception("boom"))  # type: ignore[method-assign]

        bars = p.get_stocks_bars("AAPL", date(2025, 5, 1), date(2025, 5, 2))
        assert bars == []

    def test_bars_sorted_ascending(self, tmp_path):
        p = _make_provider(tmp_path)
        prices = {date(2025, 5, 1): 185.0, date(2025, 5, 2): 188.0,
                  date(2025, 5, 5): 190.0}

        def fake_download(d):
            if d in prices:
                return {"AAPL": Bar(as_of=d, open=prices[d], high=prices[d]+1,
                                    low=prices[d]-1, close=prices[d], volume=1e7)}
            raise Exception("no data")

        p.download_stocks_day = fake_download  # type: ignore[method-assign]

        bars = p.get_stocks_bars("AAPL", date(2025, 5, 1), date(2025, 5, 5))
        assert [b.as_of for b in bars] == sorted(b.as_of for b in bars)


# ══════════════════════════════════════════════════════════════════════════════
# get_grouped_daily
# ══════════════════════════════════════════════════════════════════════════════

class TestGetGroupedDaily:

    def test_delegates_to_download_stocks_day(self, tmp_path):
        p = _make_provider(tmp_path)
        gz_bytes = _make_gz_bytes(SAMPLE_STOCKS_ROWS)
        p._s3 = MagicMock()
        p._s3.get_object.return_value = {"Body": io.BytesIO(gz_bytes)}

        result = p.get_grouped_daily(TEST_DATE)

        assert isinstance(result, dict)
        assert "AAPL" in result
        assert isinstance(result["AAPL"], Bar)


# ══════════════════════════════════════════════════════════════════════════════
# get_options_chain_from_flatfile
# ══════════════════════════════════════════════════════════════════════════════

class TestGetOptionsChain:

    def test_filters_by_underlying(self, tmp_path):
        p = _make_provider(tmp_path)
        gz_bytes = _make_gz_bytes(SAMPLE_OPTIONS_ROWS)
        p._s3 = MagicMock()
        p._s3.get_object.return_value = {"Body": io.BytesIO(gz_bytes)}

        # Trigger download + cache
        p.download_options_day(TEST_DATE)

        aapl_chain = p.get_options_chain_from_flatfile("AAPL", TEST_DATE)
        assert all(r["underlying"] == "AAPL" for r in aapl_chain)
        assert len(aapl_chain) == 2  # call + put

    def test_returns_empty_for_unknown_symbol(self, tmp_path):
        p = _make_provider(tmp_path)
        gz_bytes = _make_gz_bytes(SAMPLE_OPTIONS_ROWS)
        p._s3 = MagicMock()
        p._s3.get_object.return_value = {"Body": io.BytesIO(gz_bytes)}

        p.download_options_day(TEST_DATE)

        chain = p.get_options_chain_from_flatfile("TSLA", TEST_DATE)
        assert chain == []

    def test_downloads_if_not_cached(self, tmp_path):
        p = _make_provider(tmp_path)
        gz_bytes = _make_gz_bytes(SAMPLE_OPTIONS_ROWS)
        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {"Body": io.BytesIO(gz_bytes)}
        p._s3 = mock_s3

        # No prior download
        chain = p.get_options_chain_from_flatfile("MSFT", TEST_DATE)

        assert mock_s3.get_object.call_count == 1
        assert len(chain) == 1
        assert chain[0]["underlying"] == "MSFT"


# ══════════════════════════════════════════════════════════════════════════════
# PolygonProvider integration — flat file tried before REST
# ══════════════════════════════════════════════════════════════════════════════

class TestPolygonFlatFileIntegration:

    def test_get_grouped_daily_uses_flat_file_before_rest(self, tmp_path):
        """PolygonProvider should call FlatFileProvider first, not REST."""
        from quantlab.providers.polygon import PolygonProvider

        poly = PolygonProvider(api_key="test")

        # Patch the breadth cache to miss, and FlatFileProvider to succeed
        fake_bars = {
            "AAPL": Bar(as_of=TEST_DATE, open=185.0, high=188.0, low=184.0,
                        close=187.0, volume=5e7),
        }

        with patch.object(poly, "_load_breadth_cache", return_value=None):
            with patch.object(poly, "_save_breadth_cache"):
                with patch(
                    "quantlab.providers.flat_files.FlatFileProvider"
                ) as MockFlat:
                    instance = MockFlat.return_value
                    instance.get_grouped_daily.return_value = fake_bars

                    result = poly.get_grouped_daily(TEST_DATE)

        assert result == fake_bars
        instance.get_grouped_daily.assert_called_once_with(TEST_DATE)

    def test_get_grouped_daily_falls_back_to_rest_on_flat_file_error(self):
        """When FlatFileProvider raises, PolygonProvider falls back to REST."""
        from quantlab.providers.polygon import PolygonProvider

        poly = PolygonProvider(api_key="test")

        rest_response = {
            "results": [
                {"T": "AAPL", "t": 1746072000000, "o": 185.0, "h": 188.0,
                 "l": 184.0, "c": 187.0, "v": 5e7}
            ]
        }

        with patch.object(poly, "_load_breadth_cache", return_value=None):
            with patch.object(poly, "_save_breadth_cache"):
                with patch(
                    "quantlab.providers.flat_files.FlatFileProvider",
                    side_effect=Exception("S3 unavailable"),
                ):
                    with patch.object(poly, "_get", return_value=rest_response):
                        result = poly.get_grouped_daily(TEST_DATE)

        assert "AAPL" in result

    def test_get_grouped_daily_uses_breadth_cache_first(self):
        """Breadth cache hit skips both flat file and REST."""
        from quantlab.providers.polygon import PolygonProvider

        poly = PolygonProvider(api_key="test")
        cached = {"AAPL": Bar(as_of=TEST_DATE, open=1.0, high=1.0, low=1.0,
                              close=1.0, volume=1.0)}

        with patch.object(poly, "_load_breadth_cache", return_value=cached):
            with patch("quantlab.providers.flat_files.FlatFileProvider") as MockFlat:
                result = poly.get_grouped_daily(TEST_DATE)

        MockFlat.assert_not_called()
        assert result is cached
