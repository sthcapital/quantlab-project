"""
quantlab.universe — Tradeable universe builder and manager.

Expands the scanner from a fixed 50-symbol list to the full liquid optionable
US equity universe (typically 2,000–2,500 names).

Build pipeline:
    1. Polygon grouped daily      →  raw ~12,299 US tickers
    2. Price / volume filters     →  ~3,500 liquid names
    3. Symbol quality filters     →  ~2,800 (removes warrants, preferred, ETF tickers)
    4. Optionable check           →  ~2,300 confirmed optionable equities
    5. Sort by dollar volume, cache, return

Optionable check priority (no TWS required):
    1. Polygon /v3/reference/options/contracts (POLYGON_API_KEY, ~12 min first run)
    2. Massive S3 options flat file  — reads the 'underlying' column from the most
       recent cached options Parquet; instant when cache is warm, one S3 download
       otherwise.  No API rate limits.
    3. Static curated list fallback (data/external/optionable_universe.py)

Caching:
    data/processed/universe_{YYYY-MM-DD}.parquet   — full universe list
    data/processed/optionable_{YYYY-MM-DD}.parquet — confirmed optionable symbols
    DuckDB universe_history table                   — filter stats per date

Usage::

    from quantlab.universe import UniverseManager
    from quantlab.providers.polygon import PolygonProvider

    polygon = PolygonProvider()
    mgr     = UniverseManager()

    symbols, stats = mgr.build_tradeable_universe(
        trade_date       = date.today(),
        polygon_provider = polygon,
    )
    print(f"{stats.final_count} optionable stocks from {stats.total_raw} raw")

    # Subsequent calls load from cache instantly
    symbols, stats = mgr.build_tradeable_universe(date.today(), polygon)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Ticker patterns that indicate non-common-stock instruments
_EXCLUDED_SUFFIXES_IN_TICKER = (".W", ".R", ".U", ".RT")   # warrant / right / unit
_EXCLUDED_SUBSTRINGS = ("ETF", "ETP", "ETN")               # exchange-traded products
_MAX_TICKER_LEN = 5                                         # >5 chars = unusual issue

# Default filters
DEFAULT_MIN_PRICE        = 10.0
DEFAULT_MIN_VOLUME       = 100_000.0
DEFAULT_MIN_DOLLAR_VOL   = 5_000_000.0

# ── Hard exclusion list: known non-equity instruments ──────────────────────────
# These pass the basic ticker-quality check (≤5 chars, no dots, no W/R/Z suffix)
# but are not growth stocks and must never appear as long candidates.
EXCLUDE_SYMBOLS: frozenset[str] = frozenset({
    # Volatility products (VIX futures ETPs)
    "VXX", "UVXY", "SVXY", "VIXY", "VXZ",
    # Crypto ETFs / trusts
    "BITI", "ETHD", "SETH", "SBIT", "BTCW", "GBTC", "ETHE",
    # Leveraged inverse gold-miner ETFs
    "DUST", "GDXD", "NUGT", "JNUG", "JDST",
})

# Substrings in a ticker that identify leveraged/inverse products
_LEVERAGED_SUBSTRINGS = ("2X", "3X", "ULTRA", "BEAR", "BULL", "INVERSE", "SHORT")


# ── Exclusion helpers ─────────────────────────────────────────────────────────

def _is_excluded_symbol(sym: str) -> bool:
    """
    Return True if the symbol should never appear in the growth-stock universe.

    Catches leveraged ETFs, volatility products, and crypto ETFs that pass the
    basic apply_symbol_filter() checks (no dots, ≤5 chars, no W/R/Z suffix).
    """
    s = sym.upper()
    if s in EXCLUDE_SYMBOLS:
        return True
    return any(sub in s for sub in _LEVERAGED_SUBSTRINGS)


def _cs_cache_path(trade_date: date) -> "Path":
    from quantlab.storage import DATA_PROCESSED, ensure_dirs
    ensure_dirs()
    return DATA_PROCESSED / f"cs_tickers_{trade_date.isoformat()}.parquet"


def fetch_common_stock_tickers(api_key: str, trade_date: date) -> "set[str] | None":
    """
    Return the set of common-stock (type=CS) tickers from Polygon, cached daily.

    Paginates through GET /v3/reference/tickers?type=CS&active=true until all
    results are collected.  Caches to data/processed/cs_tickers_{date}.parquet
    so subsequent same-day calls are instant.

    Returns None on any failure so callers can skip the filter gracefully.
    """
    import requests

    cache_path = _cs_cache_path(trade_date)
    if cache_path.exists():
        try:
            import pyarrow.parquet as pq
            tbl = pq.read_table(str(cache_path)).to_pydict()
            syms: set[str] = set(tbl.get("symbol", []))
            logger.info("CS tickers cache hit for %s: %d symbols", trade_date, len(syms))
            return syms
        except Exception as exc:
            logger.debug("CS tickers cache read failed: %s", exc)

    logger.info("Fetching common-stock (CS) ticker list from Polygon ...")
    tickers: list[str] = []
    url = "https://api.polygon.io/v3/reference/tickers"
    params: dict = {"type": "CS", "active": "true", "limit": 1000, "apiKey": api_key}

    try:
        session = requests.Session()
        while True:
            r = session.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            tickers.extend(t["ticker"] for t in data.get("results", []) if "ticker" in t)
            next_url = data.get("next_url")
            if not next_url:
                break
            url = next_url
            params = {"apiKey": api_key}

        logger.info("Fetched %d CS tickers from Polygon", len(tickers))

        if tickers:
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
                pq.write_table(pa.table({"symbol": tickers}), cache_path)
                logger.debug("CS tickers cached → %s", cache_path.name)
            except Exception as exc:
                logger.warning("CS tickers cache write failed: %s", exc)

        return set(tickers)

    except Exception as exc:
        logger.warning("CS tickers fetch failed: %s — skipping CS filter", exc)
        return None


# ── Filter result dataclass ────────────────────────────────────────────────────

@dataclass
class UniverseStats:
    """Filter statistics for one universe build run."""
    date: str
    total_raw: int           = 0
    after_price: int         = 0
    after_volume: int        = 0
    after_dollar_vol: int    = 0
    after_symbol_filter: int = 0
    optionable_count: int    = 0
    final_count: int         = 0
    min_price: float         = DEFAULT_MIN_PRICE
    min_volume: float        = DEFAULT_MIN_VOLUME
    min_dollar_volume: float = DEFAULT_MIN_DOLLAR_VOL
    optionable_only: bool    = True

    def summary(self) -> str:
        """One-line human-readable filter summary for scan output."""
        parts = [
            f"Universe: {self.final_count:,} {'optionable ' if self.optionable_only else ''}stocks",
            f"filtered from {self.total_raw:,}",
            f"price>=${self.min_price:.0f}",
            f"dvol>=${self.min_dollar_volume/1_000_000:.0f}M",
        ]
        if self.optionable_only:
            parts.append("options confirmed")
        return " | ".join(parts)


# ── Pure filter functions (fully testable offline) ─────────────────────────────

def apply_price_volume_filter(
    grouped_data: dict,           # {symbol: Bar}
    min_price: float        = DEFAULT_MIN_PRICE,
    min_volume: float       = DEFAULT_MIN_VOLUME,
    min_dollar_volume: float = DEFAULT_MIN_DOLLAR_VOL,
) -> list[tuple[str, float]]:
    """
    Filter bars by price, volume, and dollar-volume thresholds.

    Args:
        grouped_data:      {symbol: Bar} from PolygonProvider.get_grouped_daily().
        min_price:         Minimum closing price (default $10).
        min_volume:        Minimum share volume (default 100,000).
        min_dollar_volume: Minimum price × volume (default $5M).

    Returns:
        List of (symbol, dollar_volume) tuples passing all filters,
        sorted by dollar_volume descending.
    """
    passed: list[tuple[str, float]] = []
    for sym, bar in grouped_data.items():
        if bar.close < min_price:
            continue
        if bar.volume < min_volume:
            continue
        dvol = bar.close * bar.volume
        if dvol < min_dollar_volume:
            continue
        passed.append((sym, dvol))
    return sorted(passed, key=lambda x: x[1], reverse=True)


def apply_symbol_filter(symbols: list[str]) -> list[str]:
    """
    Remove tickers that represent non-common-stock instruments.

    Exclusion rules (applied in order):
        1. Ticker contains '.'         → preferred shares, warrants, ADR units
        2. Ticker length > 5           → unusual/complex issues
        3. Ticker ends with W/R/Z      → warrants, rights, when-issued
        4. Ticker contains ETF/ETP/ETN → exchange-traded products

    Args:
        symbols: Input ticker list.

    Returns:
        Filtered list with only clean common-stock tickers.
    """
    result: list[str] = []
    for sym in symbols:
        s = sym.upper()

        if "." in s:
            continue

        if len(s) > _MAX_TICKER_LEN:
            continue

        # Single trailing letters that flag non-equity issues
        # W = warrant, R = rights, Z = when-issued, Q = bankruptcy
        if len(s) >= 2 and s[-1] in ("W", "R", "Z", "Q") and s[:-1].isalpha():
            continue

        if any(sub in s for sub in _EXCLUDED_SUBSTRINGS):
            continue

        result.append(sym)
    return result


# ── Optionable cache helpers ───────────────────────────────────────────────────

def _optionable_cache_path(trade_date: date) -> Path:
    from quantlab.storage import DATA_PROCESSED, ensure_dirs
    ensure_dirs()
    return DATA_PROCESSED / f"optionable_{trade_date.isoformat()}.parquet"


def _universe_cache_path(trade_date: date) -> Path:
    from quantlab.storage import DATA_PROCESSED, ensure_dirs
    ensure_dirs()
    return DATA_PROCESSED / f"universe_{trade_date.isoformat()}.parquet"


def load_optionable_cache(trade_date: date) -> list[str] | None:
    """Load the IBKR-confirmed optionable symbol list from parquet cache."""
    path = _optionable_cache_path(trade_date)
    if not path.exists():
        return None
    try:
        import pyarrow.parquet as pq
        tbl = pq.read_table(path).to_pydict()
        return list(tbl.get("symbol", []))
    except Exception as exc:
        logger.debug("optionable cache read failed: %s", exc)
        return None


def save_optionable_cache(trade_date: date, symbols: list[str]) -> None:
    """Persist the optionable symbol list to parquet cache."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        path = _optionable_cache_path(trade_date)
        pq.write_table(pa.table({"symbol": symbols}), path)
        logger.debug("optionable cache saved: %d symbols → %s", len(symbols), path.name)
    except Exception as exc:
        logger.warning("optionable cache write failed: %s", exc)


def load_universe_cache(trade_date: date) -> tuple[list[str], UniverseStats] | None:
    """Load the full tradeable universe list from parquet cache."""
    path = _universe_cache_path(trade_date)
    if not path.exists():
        return None
    try:
        import pyarrow.parquet as pq
        tbl  = pq.read_table(path).to_pydict()
        syms = list(tbl.get("symbol", []))
        # Reconstruct a minimal stats object from stored metadata
        stats = UniverseStats(
            date        = trade_date.isoformat(),
            final_count = len(syms),
        )
        return syms, stats
    except Exception as exc:
        logger.debug("universe cache read failed: %s", exc)
        return None


def save_universe_cache(trade_date: date, symbols: list[str],
                        dvols: list[float]) -> None:
    """Persist the universe symbol list with dollar-volumes to parquet."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        path = _universe_cache_path(trade_date)
        pq.write_table(pa.table({
            "symbol":       symbols,
            "dollar_volume": dvols,
        }), path)
        logger.debug("universe cache saved: %d symbols → %s", len(symbols), path.name)
    except Exception as exc:
        logger.warning("universe cache write failed: %s", exc)


# ── Polygon options availability check (preferred — no IBKR needed) ───────────

def check_optionable_polygon(
    candidates: list[str],
    api_key: str,
    trade_date: date,
    max_workers: int = 3,
    sleep_between: float = 0.05,
    show_progress: bool = True,
) -> list[str]:
    """
    Check which symbols have listed options via the Polygon options reference API.

    Uses GET /v3/reference/options/contracts?underlying_ticker={sym}&limit=1.
    If any contract is returned, the symbol is optionable.

    Polygon rate-limit notes:
        - 3 concurrent workers is reliable across all plan tiers.
        - Above 5 concurrent the API returns ERROR responses under load.
        - Each request takes ~0.2–0.7 s; at 3 workers: ~12 min for 3,000 symbols.

    Results are cached to data/processed/optionable_{date}.parquet so
    subsequent same-day calls return instantly.

    Args:
        candidates:     Symbol list to check.
        api_key:        Polygon.io API key (POLYGON_API_KEY env var).
        trade_date:     Date to associate with the cache entry.
        max_workers:    Concurrent HTTP workers (default 3; safe limit).
        sleep_between:  Seconds to sleep per request (default 0.05s).
        show_progress:  Print progress every 200 symbols.

    Returns:
        Subset of candidates confirmed to have listed options.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import requests

    # Check cache first
    cached = load_optionable_cache(trade_date)
    if cached is not None:
        logger.info("Optionable cache hit for %s: %d symbols", trade_date, len(cached))
        cached_set = set(cached)
        return [s for s in candidates if s in cached_set]

    n = len(candidates)
    logger.info(
        "Checking %d symbols for options via Polygon (~%.0f min at %d workers)...",
        n, n / (max_workers * (1 / (sleep_between + 0.3))), max_workers,
    )

    # Thread-local sessions avoid connection conflicts
    _local = threading.local()

    def _sess():
        if not hasattr(_local, "s"):
            s = requests.Session()
            s.headers.update({"Connection": "keep-alive"})
            _local.s = s
        return _local.s

    def _check(sym: str) -> tuple[str, bool]:
        if sleep_between > 0:
            time.sleep(sleep_between)
        try:
            r = _sess().get(
                "https://api.polygon.io/v3/reference/options/contracts",
                params={"underlying_ticker": sym, "limit": 1, "apiKey": api_key},
                timeout=8,
            )
            d = r.json()
            return sym, d.get("status") == "OK" and len(d.get("results", [])) > 0
        except Exception as exc:
            logger.debug("%s: options check failed: %s", sym, exc)
            return sym, False

    optionable: list[str] = []
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_check, s): s for s in candidates}
        for future in as_completed(futures):
            sym, has_opts = future.result()
            if has_opts:
                optionable.append(sym)
            completed += 1
            if show_progress and completed % 200 == 0:
                pct = completed / n * 100
                print(
                    f"  Options check: {completed}/{n} ({pct:.0f}%)"
                    f"  optionable so far: {len(optionable)}",
                    flush=True,
                )

    # Restore dollar-volume order (futures complete out of order)
    candidate_set = set(candidates)
    optionable_ordered = [s for s in candidates if s in set(optionable)]

    logger.info(
        "Polygon options check complete: %d/%d optionable", len(optionable_ordered), n
    )
    save_optionable_cache(trade_date, optionable_ordered)
    return optionable_ordered


# ── Massive S3 options flat-file check (preferred when no Polygon key) ────────

def check_optionable_massive(
    candidates: list[str],
    trade_date: date,
    lookback_days: int = 7,
) -> list[str]:
    """
    Check which symbols have listed options via the Massive S3 options flat files.

    Reads only the ``underlying`` column from the most recent cached options
    Parquet — no API key and no per-symbol HTTP requests required.  On cache
    miss for the most recent day it attempts one S3 download; if that also
    fails it walks back through ``lookback_days`` to find any cached file.

    Falls back to the static optionable list when no flat file is available.

    Args:
        candidates:    Symbol list to check.
        trade_date:    Date to associate with the resulting cache entry.
        lookback_days: How many calendar days to walk back looking for a
                       usable options file (default 7).

    Returns:
        Subset of candidates that appear as an underlying in the options file.
    """
    from quantlab.providers.flat_files import FlatFileProvider
    from datetime import timedelta

    try:
        from quantlab.market_calendar import is_market_open as _is_open
    except ImportError:
        _is_open = lambda d: d.weekday() < 5  # type: ignore[assignment]

    # Return cached result when available
    cached = load_optionable_cache(trade_date)
    if cached is not None:
        logger.info("Optionable cache hit for %s: %d symbols", trade_date, len(cached))
        cached_set = set(cached)
        return [s for s in candidates if s in cached_set]

    flat = FlatFileProvider()

    # Walk back from yesterday (today's file may not be published yet)
    check_date = trade_date - timedelta(days=1)
    options_underlyings: set[str] = set()

    for _ in range(lookback_days):
        if not _is_open(check_date):
            check_date -= timedelta(days=1)
            continue

        cache_path = flat.options_cache_path(check_date)

        # Try local Parquet cache first (read only the underlying column — fast)
        if not cache_path.exists():
            try:
                flat.download_options_day(check_date)
            except Exception as _dl_err:
                logger.debug(
                    "Massive options S3 download failed for %s: %s", check_date, _dl_err
                )
                check_date -= timedelta(days=1)
                continue

        if cache_path.exists():
            try:
                import pyarrow.parquet as pq
                tbl = pq.read_table(str(cache_path), columns=["underlying"])
                options_underlyings = set(tbl.to_pydict()["underlying"])
                logger.info(
                    "Massive options optionable check: %d unique underlyings on %s",
                    len(options_underlyings), check_date,
                )
                break
            except Exception as _read_err:
                logger.debug(
                    "Options Parquet read failed for %s: %s", check_date, _read_err
                )

        check_date -= timedelta(days=1)

    if not options_underlyings:
        logger.warning(
            "Massive options flat file unavailable within %d days of %s"
            " — falling back to static optionable list",
            lookback_days, trade_date,
        )
        return _filter_by_static_optionable(candidates)

    optionable = [s for s in candidates if s in options_underlyings]
    logger.info(
        "Massive options check: %d/%d candidates are optionable",
        len(optionable), len(candidates),
    )
    save_optionable_cache(trade_date, optionable)
    return optionable


# ── IBKR options availability check (legacy — kept for reference only) ─────────

def check_optionable_ibkr(
    candidates: list[str],
    ib,
    trade_date: date,
    client_id: int = 61,
    sleep_per_symbol: float = 0.5,
    show_progress: bool = True,
) -> list[str]:
    """
    Verify which symbols have listed options via IBKR reqSecDefOptParams().

    Caches the result so subsequent same-day calls are instant.
    First run typically takes ~20 minutes for 2,500 symbols.

    Args:
        candidates:       Symbols to check.
        ib:               Connected IB() instance (uses existing connection).
        trade_date:       Date to associate with the cache.
        client_id:        Not used (ib already connected); kept for docs clarity.
        sleep_per_symbol: Seconds between IBKR calls (default 0.5s).
        show_progress:    Print progress every 100 symbols.

    Returns:
        Subset of candidates that have confirmed options chains.
    """
    # Return cached result if available
    cached = load_optionable_cache(trade_date)
    if cached is not None:
        logger.info("Optionable cache hit for %s: %d symbols", trade_date, len(cached))
        return [s for s in candidates if s in set(cached)]

    from ib_insync import Stock

    logger.info("Checking %d symbols for options availability (~%.0f min)...",
                len(candidates), len(candidates) * sleep_per_symbol / 60)
    optionable: list[str] = []
    n = len(candidates)

    for i, symbol in enumerate(candidates, 1):
        try:
            stock     = Stock(symbol, "SMART", "USD")
            qualified = ib.qualifyContracts(stock)
            if not qualified:
                time.sleep(sleep_per_symbol * 0.5)
                continue

            chains = ib.reqSecDefOptParams(
                underlyingSymbol  = symbol,
                futFopExchange    = "",
                underlyingSecType = qualified[0].secType,
                underlyingConId   = qualified[0].conId,
            )
            if chains:
                optionable.append(symbol)

            time.sleep(sleep_per_symbol)

            if show_progress and i % 100 == 0:
                pct  = i / n * 100
                eta  = (n - i) * sleep_per_symbol / 60
                print(f"  Options check: {i}/{n} ({pct:.0f}%)  "
                      f"optionable so far: {len(optionable)}  "
                      f"ETA: {eta:.0f}m", flush=True)

        except Exception as exc:
            logger.debug("%s: options check failed: %s", symbol, exc)

    logger.info("Options check complete: %d/%d optionable", len(optionable), n)
    save_optionable_cache(trade_date, optionable)
    return optionable


# ── Static optionable filter (free-tier / no-connection fallback) ─────────────

def _filter_by_static_optionable(candidates: list[str]) -> list[str]:
    """
    Intersect candidates with the curated static optionable universe.

    Used when neither a Polygon API key nor an IBKR connection is available,
    making a dynamic per-symbol options check impractical (Polygon free tier
    allows only 5 req/min on options endpoints).

    The static list covers all S&P 500 constituents plus top Russell 1000 names
    (see data/external/optionable_universe.py).  Replace with the dynamic
    check once a Polygon paid tier or FactSet feed is available.
    """
    try:
        from pathlib import Path as _Path
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(
            "optionable_universe",
            _Path(__file__).parents[2] / "data" / "external" / "optionable_universe.py",
        )
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        known: frozenset[str] = _mod.OPTIONABLE_UNIVERSE
    except Exception as exc:
        logger.warning("Could not load static optionable universe: %s — keeping all candidates", exc)
        return candidates

    filtered = [s for s in candidates if s in known]
    logger.info(
        "Static optionable filter: %d/%d candidates matched known-optionable list",
        len(filtered), len(candidates),
    )
    return filtered


# ── Universe manager ───────────────────────────────────────────────────────────

class UniverseManager:
    """
    Builds and caches the full tradeable optionable US equity universe.

    Typical output: 2,000–2,500 symbols (filtered from ~12,299 US tickers).

    Filter defaults:
        min_price        = $10.00
        min_volume       = 100,000 shares
        min_dollar_volume = $5,000,000
        symbol quality   = no dots, ≤5 chars, no warrants/ETPs
        optionable       = IBKR-confirmed options exist
    """

    def __init__(
        self,
        min_price: float        = DEFAULT_MIN_PRICE,
        min_volume: float       = DEFAULT_MIN_VOLUME,
        min_dollar_volume: float = DEFAULT_MIN_DOLLAR_VOL,
    ) -> None:
        self.min_price        = min_price
        self.min_volume       = min_volume
        self.min_dollar_volume = min_dollar_volume

    # ── Public API ─────────────────────────────────────────────────────────────

    def build_tradeable_universe(
        self,
        trade_date,
        polygon_provider,
        ib=None,
        optionable_only: bool = True,
        force_rebuild: bool   = False,
        polygon_api_key: str | None = None,
    ) -> tuple[list[str], UniverseStats]:
        """
        Build the tradeable universe for a given date.

        Checks the full cache first.  Only fetches Polygon and runs the IBKR
        check when no cache is available for the date (or force_rebuild=True).

        Args:
            trade_date:       Trading date (date or str YYYY-MM-DD).
            polygon_provider: PolygonProvider instance for grouped daily data.
            ib:               Connected IB() instance for options check.
                              Pass None to skip options filtering (faster).
            optionable_only:  Apply IBKR options availability check.
            force_rebuild:    Ignore cache and rebuild from scratch.

        Returns:
            (symbols, stats) — symbol list sorted by dollar_volume desc,
            and a UniverseStats object with filter counts.
        """
        if isinstance(trade_date, str):
            from datetime import date as _date
            trade_date = _date.fromisoformat(trade_date)

        # ── Cache check ────────────────────────────────────────────────────────
        if not force_rebuild:
            cached = load_universe_cache(trade_date)
            if cached is not None:
                symbols, stats = cached
                logger.info("Universe cache hit for %s: %d symbols", trade_date, len(symbols))
                return symbols, stats

        # ── Step 1: Polygon grouped daily (with previous-day fallback on 403) ──
        # Polygon returns 403 for today's data while the market is still open.
        # Walk back through recent trading days until valid data is found.
        # Universe composition doesn't change intraday, so yesterday's filtered
        # list is valid for today's scan.
        import requests as _requests
        from quantlab.market_calendar import prev_trading_day as _prev_day

        _MAX_DAY_LOOKBACK = 5
        actual_date = trade_date
        grouped = None

        for _attempt in range(_MAX_DAY_LOOKBACK):
            if _attempt > 0:
                # Check fallback date cache before hitting the API again
                if not force_rebuild:
                    _fb = load_universe_cache(actual_date)
                    if _fb is not None:
                        _fb_syms, _fb_stats = _fb
                        logger.warning(
                            "Universe: %s unavailable — using cached %s (%d symbols)",
                            trade_date, actual_date, len(_fb_syms),
                        )
                        return _fb_syms, _fb_stats

            logger.info("Fetching grouped daily for %s ...", actual_date)
            try:
                grouped = polygon_provider.get_grouped_daily(actual_date)
                if actual_date != trade_date:
                    logger.warning(
                        "Universe: %s unavailable — fell back to %s (%d raw tickers)",
                        trade_date, actual_date, len(grouped),
                    )
                break
            except _requests.HTTPError as _exc:
                _status = getattr(getattr(_exc, "response", None), "status_code", None)
                if _status == 403:
                    logger.warning(
                        "Grouped daily for %s returned 403 "
                        "(market may still be open) — trying previous trading day",
                        actual_date,
                    )
                    actual_date = _prev_day(actual_date)
                else:
                    raise
        else:
            logger.warning(
                "No grouped daily data available within %d trading days of %s",
                _MAX_DAY_LOOKBACK, trade_date,
            )
            return [], UniverseStats(date=trade_date.isoformat())

        if not grouped:
            logger.warning("No grouped daily data for %s", actual_date)
            return [], UniverseStats(date=actual_date.isoformat())

        stats = UniverseStats(
            date             = actual_date.isoformat(),
            total_raw        = len(grouped),
            min_price        = self.min_price,
            min_volume       = self.min_volume,
            min_dollar_volume = self.min_dollar_volume,
            optionable_only  = optionable_only,
        )
        logger.info("Raw tickers: %d", stats.total_raw)

        # ── Step 2: Price / volume filter ─────────────────────────────────────
        price_vol_passed = apply_price_volume_filter(
            grouped,
            min_price        = self.min_price,
            min_volume       = self.min_volume,
            min_dollar_volume = self.min_dollar_volume,
        )
        candidates   = [sym for sym, _ in price_vol_passed]
        dvol_by_sym  = {sym: dv for sym, dv in price_vol_passed}

        stats.after_price      = sum(1 for s, b in grouped.items() if b.close >= self.min_price)
        stats.after_volume     = sum(1 for s, b in grouped.items()
                                     if b.close >= self.min_price and b.volume >= self.min_volume)
        stats.after_dollar_vol = len(candidates)
        logger.info("After price/vol filter: %d", stats.after_dollar_vol)

        # ── Step 3: Symbol quality filter ─────────────────────────────────────
        candidates = apply_symbol_filter(candidates)
        logger.info("After symbol quality filter: %d", len(candidates))

        # ── Step 3.5: Hard exclusion + Polygon CS-type filter ─────────────────
        # Remove known leveraged ETFs, vol products, and crypto ETFs that pass
        # the basic ticker-quality checks above.
        import os as _os
        candidates = [s for s in candidates if not _is_excluded_symbol(s)]
        logger.info("After hard exclusion filter: %d", len(candidates))

        # When a Polygon API key is available, restrict further to CS-type only,
        # eliminating all ETFs, ETNs, and non-equity instruments in one pass.
        _poly_api_key = polygon_api_key or _os.environ.get("POLYGON_API_KEY", "")
        if _poly_api_key:
            _cs_set = fetch_common_stock_tickers(_poly_api_key, actual_date)
            if _cs_set is not None:
                _pre_cs = len(candidates)
                candidates = [s for s in candidates if s in _cs_set]
                logger.info(
                    "CS type filter: %d → %d (%d non-CS removed)",
                    _pre_cs, len(candidates), _pre_cs - len(candidates),
                )

        stats.after_symbol_filter = len(candidates)
        logger.info("After symbol/exclusion filter: %d", stats.after_symbol_filter)

        # ── Step 4: Options check ──────────────────────────────────────────────
        # Priority (TWS not required for any path):
        #   1. Polygon /v3/reference/options/contracts — per-symbol REST check.
        #      First run ~12 min for 3,000 symbols; subsequent same-day: instant.
        #   2. Massive S3 options flat file — reads 'underlying' column from the
        #      most recent cached Parquet.  No API rate limits; instant on cache hit.
        #   3. Static optionable fallback (S&P 500 + Russell 1000 curated list).
        if optionable_only:
            import os
            api_key = polygon_api_key or os.environ.get("POLYGON_API_KEY", "")
            if api_key:
                candidates = check_optionable_polygon(
                    candidates, api_key, trade_date
                )
            else:
                # Massive flat-file check — no API key needed
                candidates = check_optionable_massive(candidates, trade_date)
            stats.optionable_count = len(candidates)
            logger.info("After options filter: %d", stats.optionable_count)
        else:
            stats.optionable_count = len(candidates)

        # ── Step 5: Final sort by dollar volume ────────────────────────────────
        candidates.sort(key=lambda s: dvol_by_sym.get(s, 0), reverse=True)
        dvols = [dvol_by_sym.get(s, 0.0) for s in candidates]
        stats.final_count = len(candidates)

        # ── Step 6: Cache ──────────────────────────────────────────────────────
        save_universe_cache(actual_date, candidates, dvols)
        self._save_stats_to_db(stats)

        logger.info("Universe built: %d symbols for %s", stats.final_count, trade_date)
        return candidates, stats

    # ── Stats persistence ──────────────────────────────────────────────────────

    def _save_stats_to_db(self, stats: UniverseStats) -> None:
        """Persist filter statistics to DuckDB universe_history table."""
        try:
            from quantlab.storage import get_db
            con = get_db()
            con.execute("""
                INSERT OR REPLACE INTO universe_history (
                    date, total_raw, after_price, after_volume, after_dollar_vol,
                    after_symbol_filter, optionable_count, final_count,
                    min_price, min_volume, min_dollar_volume, optionable_only
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                stats.date, stats.total_raw, stats.after_price,
                stats.after_volume, stats.after_dollar_vol,
                stats.after_symbol_filter, stats.optionable_count,
                stats.final_count, stats.min_price, stats.min_volume,
                stats.min_dollar_volume, stats.optionable_only,
            ])
            con.close()
        except Exception as exc:
            logger.debug("universe_history save failed: %s", exc)
