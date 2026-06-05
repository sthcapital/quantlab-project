"""
quantlab.universe — Tradeable universe builder and manager.

Expands the scanner from a fixed 50-symbol list to the full liquid optionable
US equity universe (typically 2,000–2,500 names).

Build pipeline:
    1. Polygon grouped daily  →  raw ~12,299 US tickers
    2. Price / volume filters →  ~3,500 liquid names
    3. Symbol quality filters →  ~2,800 (removes warrants, preferred, ETF tickers)
    4. IBKR options check     →  ~2,300 confirmed optionable equities
    5. Sort by dollar volume, cache, return

Caching:
    data/processed/universe_{YYYY-MM-DD}.parquet  — full universe list
    data/processed/optionable_{YYYY-MM-DD}.parquet — IBKR-confirmed symbols
    DuckDB universe_history table                  — filter stats per date

IBKR options check (client_id=61):
    Calls reqSecDefOptParams() for each candidate symbol.
    First run ~20 minutes for 2,500 symbols; subsequent same-day runs instant.
    Set ib=None to skip the check (tradeable_no_options mode).

Usage::

    from quantlab.universe import UniverseManager
    from quantlab.providers.polygon import PolygonProvider

    polygon = PolygonProvider()
    mgr     = UniverseManager()

    # Build + cache (first run ~20 min with options check)
    symbols, stats = mgr.build_tradeable_universe(
        trade_date       = date.today(),
        polygon_provider = polygon,
        ib               = ib,   # connected IB() instance; None to skip options check
    )
    print(f"{stats['final_count']} optionable stocks from {stats['total_raw']} raw")

    # Subsequent calls load from cache instantly
    symbols, stats = mgr.build_tradeable_universe(date.today(), polygon, ib=None)
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


# ── IBKR options availability check ───────────────────────────────────────────

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

        # ── Step 1: Polygon grouped daily ─────────────────────────────────────
        logger.info("Fetching grouped daily for %s ...", trade_date)
        grouped = polygon_provider.get_grouped_daily(trade_date)
        if not grouped:
            logger.warning("No grouped daily data for %s", trade_date)
            return [], UniverseStats(date=trade_date.isoformat())

        stats = UniverseStats(
            date             = trade_date.isoformat(),
            total_raw        = len(grouped),
            min_price        = self.min_price,
            min_volume       = self.min_volume,
            min_dollar_volume = self.min_dollar_volume,
            optionable_only  = optionable_only and ib is not None,
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
        stats.after_symbol_filter = len(candidates)
        logger.info("After symbol filter: %d", stats.after_symbol_filter)

        # ── Step 4: IBKR options check ─────────────────────────────────────────
        if optionable_only and ib is not None:
            candidates = check_optionable_ibkr(candidates, ib, trade_date)
            stats.optionable_count = len(candidates)
            logger.info("After options filter: %d", stats.optionable_count)
        else:
            stats.optionable_count = len(candidates)

        # ── Step 5: Final sort by dollar volume ────────────────────────────────
        candidates.sort(key=lambda s: dvol_by_sym.get(s, 0), reverse=True)
        dvols = [dvol_by_sym.get(s, 0.0) for s in candidates]
        stats.final_count = len(candidates)

        # ── Step 6: Cache ──────────────────────────────────────────────────────
        save_universe_cache(trade_date, candidates, dvols)
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
