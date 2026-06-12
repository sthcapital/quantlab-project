"""
quantlab.watchlist — Active setup tracking and forward-return feedback loop.

Every high-conviction scan result (conviction ≥ 0.70) is added to a DuckDB
watchlist table.  The table is the system's memory: it records what was
identified, at what price, and whether it followed through.

Lifecycle of a watchlist entry:
    watching    → setup is live, being monitored
    stopped_out → price fell below the ATR stop level
    expired     → 10 trading days elapsed without stop being hit
    (triggered  → future: order was placed and filled)

The forward-return columns (realized_ret_1d/3d/5d) are filled by
scripts/track_forward_returns.py, which runs after market close each day.
These are the ground-truth labels that validate whether conviction scores
are actually predictive.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

from quantlab.storage import DB_PATH, get_db

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

MIN_CONVICTION_FOR_WATCHLIST = 0.70


def _trading_days_elapsed(
    from_date: date,
    to_date: date | None = None,
    skip_dates: set[date] | None = None,
) -> int:
    """
    Count Mon–Fri trading days between from_date (exclusive) and to_date (inclusive).

    ``skip_dates`` are treated as NEUTRAL — they neither extend nor break a
    streak's inactivity window.  Used for days when OUR universe build was
    degenerate (sanity-gate refused): a symbol must not be pruned because the
    infrastructure failed to scan it.
    """
    if to_date is None:
        to_date = date.today()
    count = 0
    current = from_date + timedelta(days=1)
    while current <= to_date:
        if current.weekday() < 5 and not (skip_dates and current in skip_dates):
            count += 1
        current += timedelta(days=1)
    return count


def _degenerate_build_dates(con) -> set[date]:
    """
    Dates whose universe build was refused by the sanity gate
    (universe_history.gate_accepted = FALSE).  These days are neutral for
    streak/staleness accounting — the scan ran on degraded infrastructure,
    not on a real read of the market.
    """
    try:
        rows = con.execute(
            "SELECT date FROM universe_history WHERE gate_accepted = FALSE"
        ).fetchall()
        out: set[date] = set()
        for (d,) in rows:
            if isinstance(d, str):
                d = date.fromisoformat(d)
            elif hasattr(d, "date") and not isinstance(d, date):
                d = d.date()
            out.add(d)
        return out
    except Exception:
        return set()


def _layers_fired(scan_result) -> str:
    """Return a compact comma-separated string of conviction layers that contributed."""
    layers: list[str] = []

    if getattr(scan_result, "regime_bullish", False):
        layers.append("REGIME")

    ea = getattr(scan_result, "earnings_acceleration", 0.0)
    if ea >= 0.5:
        layers.append("EARN")

    ar = getattr(scan_result, "accumulation_ratio", 0.0)
    if ar >= 0.6:
        layers.append("ACCUM")

    cv = getattr(scan_result, "climactic_volume", 0.0)
    if cv >= 0.7:
        layers.append("CLIM")

    ab = getattr(scan_result, "absorption", 0.0)
    if ab >= 0.6:
        layers.append("ABS")

    vc = getattr(scan_result, "volume_character", 0.0)
    if vc >= 0.6:
        layers.append("VOL_CHAR")

    if getattr(scan_result, "wyckoff_spring", False):
        layers.append("SPRING")

    if getattr(scan_result, "multi_lookback_confirmed", False):
        layers.append("MULTI_LB")

    if getattr(scan_result, "sector_cluster", False):
        layers.append("SECTOR_CLUSTER")

    opt = getattr(scan_result, "options_conviction", None)
    if opt is not None and opt >= 0.6:
        layers.append("OPTIONS")

    news_cat = getattr(scan_result, "news_category", "none")
    news_n   = getattr(scan_result, "news_count", 0)
    if news_n > 0 and news_cat not in ("none", "other", "downgrade"):
        layers.append(f"NEWS:{news_cat}")

    return ",".join(layers) if layers else "signal"


# ── Core API ───────────────────────────────────────────────────────────────────

def add_to_watchlist(
    scan_result,
    min_conviction: float = MIN_CONVICTION_FOR_WATCHLIST,
    note: str = "",
    size_factor: float = 1.0,
) -> bool:
    """
    Add a scan result to the watchlist if conviction meets the threshold.

    The watch_id (symbol_YYYY-MM-DD) acts as a primary key so the same
    symbol cannot be added twice on the same calendar day.

    Args:
        scan_result:    ScanResult dataclass instance.
        min_conviction: Minimum conviction score to accept (default 0.70).
        note:           Optional audit note stored in breadth_override_note
                        (e.g. "Added pre-breadth-load — tape=BEAR at time of scan").
        size_factor:    Position-size multiplier from the regime exposure policy
                        (1.0 full size; 0.5 half size in RECOVERY/NEUTRAL).

    Returns:
        True if the entry was inserted; False if skipped (below threshold or
        already present for today).
    """
    if scan_result.conviction_score < min_conviction:
        return False

    today      = date.today()
    watch_id   = f"{scan_result.symbol}_{today.isoformat()}"
    layers     = _layers_fired(scan_result)

    # Raw conviction inputs (JSON) — preserved for future score re-weighting
    components_json = ""
    try:
        import json
        from quantlab.execution import conviction_components
        components_json = json.dumps(conviction_components(scan_result))
    except Exception:
        pass

    # 2R price target: entry + 2 * (entry - stop)
    _ep   = scan_result.entry_close
    _stop = scan_result.atr_stop
    target_price = (
        round(_ep + 2 * (_ep - _stop), 4)
        if _ep and _stop and _ep > _stop
        else None
    )

    try:
        con = get_db()
        # INSERT OR IGNORE keeps the first entry if run twice today
        # current_price is seeded with the entry close so a position is NEVER
        # monitored with a null price — same-session entries used to sit at
        # NULL until the next EOD tracker run (SNEX/KO incidents: stop checks
        # silently skipped on day 0)
        con.execute("""
            INSERT OR IGNORE INTO watchlist (
                watch_id, symbol, date_added, entry_price, atr_stop,
                conviction_score, signal_layers, lookback, signal_type,
                target_price, status, date_updated, breadth_override_note,
                size_factor, conviction_components,
                current_price, days_on_watch
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'watching', ?, ?, ?, ?, ?, 0)
        """, [
            watch_id,
            scan_result.symbol,
            today.isoformat(),
            scan_result.entry_close,
            scan_result.atr_stop,
            scan_result.conviction_score,
            layers,
            getattr(scan_result, "lookback", None),
            getattr(scan_result, "signal_type", None),
            target_price,
            today.isoformat(),
            note,
            size_factor,
            components_json,
            scan_result.entry_close,
        ])
        con.close()
        return True
    except Exception as e:
        print(f"[watchlist] insert failed for {scan_result.symbol}: {e}")
        return False


def set_watchlist_note(watch_id: str, note: str) -> None:
    """
    Update the breadth_override_note on an existing watchlist entry.

    Used to annotate entries that were added before breadth data was available,
    or to flag any other post-hoc context that should travel with the record.

    Args:
        watch_id: The watchlist primary key (e.g. "ABT_2026-06-04").
        note:     The note to store (replaces any existing note).
    """
    try:
        con = get_db()
        con.execute(
            "UPDATE watchlist SET breadth_override_note=?, date_updated=? WHERE watch_id=?",
            [note, date.today().isoformat(), watch_id],
        )
        con.close()
        print(f"[watchlist] note set on {watch_id}: {note!r}")
    except Exception as e:
        print(f"[watchlist] set_watchlist_note failed for {watch_id}: {e}")


def get_active_watchlist(db_path: str | None = None) -> list[dict[str, Any]]:
    """
    Return all entries currently in 'watching' status, ordered by conviction desc.

    Args:
        db_path: Override DB path (for testing).

    Returns:
        List of dicts with all watchlist columns.
    """
    _COLS = [
        "watch_id", "symbol", "date_added", "entry_price", "atr_stop",
        "conviction_score", "signal_layers", "lookback", "signal_type",
        "price_1d", "price_3d", "price_5d",
        "realized_ret_1d", "realized_ret_3d", "realized_ret_5d",
        "current_price", "unrealized_ret", "days_on_watch", "status", "date_updated",
        "breadth_override_note", "target_price",
    ]
    try:
        import duckdb
        path = db_path or str(DB_PATH)
        con = duckdb.connect(path)
        from quantlab.storage import _ensure_schema
        _ensure_schema(con)
        rows = con.execute(f"""
            SELECT {', '.join(_COLS)}
            FROM watchlist
            WHERE status = 'watching'
            ORDER BY conviction_score DESC, date_added DESC
        """).fetchall()
        con.close()
        return [dict(zip(_COLS, row)) for row in rows]
    except Exception as e:
        print(f"[watchlist] get_active failed: {e}")
        return []


def _fetch_ib_price(ib_connection, symbol: str, Stock, retries: int = 2,
                    warmup_wait: float = 1.5) -> float | None:
    """
    Current price from IBKR, preferring last trade over close/midpoint.

    Retries once after a short wait: a symbol entered the same session has a
    cold delayed-data subscription whose first snapshot is often all-NaN
    (the SNEX/KO null-price bug).  Returns None when IBKR yields nothing.
    """
    import time
    try:
        ib_connection.reqMarketDataType(3)
        stock = Stock(symbol, "SMART", "USD")
        qualified = ib_connection.qualifyContracts(stock)
        if not qualified:
            return None
        for attempt in range(retries):
            ticker = ib_connection.reqTickers(qualified[0])[0]
            for candidate in [ticker.last, ticker.close, ticker.marketPrice()]:
                if candidate is not None and candidate == candidate and candidate > 0:
                    return float(candidate)
            if attempt < retries - 1:
                time.sleep(warmup_wait)   # subscription warm-up
    except Exception as exc:
        logger.debug("%s: IBKR price fetch failed: %s", symbol, exc)
    return None


def _latest_flatfile_close(symbol: str, max_back_days: int = 5) -> float | None:
    """
    Snapshot fallback: the symbol's close from the most recent cached stocks
    flat file (final EOD data, never an empty subscription).  At worst one
    session stale — still infinitely better than monitoring blind.
    """
    try:
        from datetime import timedelta

        import pyarrow.parquet as pq

        from quantlab.providers.flat_files import FlatFileProvider
        flat = FlatFileProvider()
        d = date.today()
        for _ in range(max_back_days):
            path = flat.stocks_cache_path(d)
            if path.exists():
                tbl = pq.read_table(path, columns=["ticker", "close"],
                                    filters=[("ticker", "=", symbol)]).to_pydict()
                if tbl["close"]:
                    return float(tbl["close"][0])
            d -= timedelta(days=1)
    except Exception as exc:
        logger.debug("%s: flat-file close fallback failed: %s", symbol, exc)
    return None


def update_watchlist_prices(ib_connection) -> list[dict[str, Any]]:
    """
    Refresh current prices for all active watchlist entries via IBKR.

    For each entry:
    - Fetches the current spot price using get_spot_price().
    - Computes unrealized_ret = (current − entry) / entry.
    - Sets status = 'stopped_out' if current_price ≤ effective stop.
      The effective stop is the ATR stop tightened by the regime policy's
      stop_tighten_factor (BEAR halves the entry→stop distance; all other
      regimes leave stops unchanged).
    - Sets status = 'expired'     if trading_days_on_watch ≥ 10.
    - Updates days_on_watch and date_updated.

    Args:
        ib_connection: Active IB() instance (already connected).

    Returns:
        List of dicts summarising each updated entry.
    """
    from ib_insync import Stock

    active = get_active_watchlist()
    if not active:
        return []

    # Regime policy: BEAR tape tightens stops on open positions
    _stop_factor = 1.0
    _tape = ""
    try:
        from quantlab.risk.regime_policy import get_regime_rule
        from quantlab.signals.breadth import get_latest_snapshot
        _snap = get_latest_snapshot()
        if _snap:
            _tape = _snap.tape
            _stop_factor = get_regime_rule(_tape).stop_tighten_factor
            if _stop_factor != 1.0:
                logger.info(
                    "Regime %s: stops tightened ×%.2f of entry→stop distance",
                    _tape, _stop_factor,
                )
    except Exception:
        pass

    today   = date.today()
    updates: list[dict] = []

    for entry in active:
        symbol      = entry["symbol"]
        entry_price = entry["entry_price"]
        atr_stop    = entry["atr_stop"]
        date_added  = entry["date_added"]

        if isinstance(date_added, str):
            date_added = date.fromisoformat(date_added)

        # Fetch current price — IBKR with one warm-up retry (a same-session
        # entry's delayed-data subscription often returns NaN on the first
        # snapshot: the SNEX/KO bug), then flat-file close as snapshot
        # fallback.  A position must NEVER be monitored with a null price.
        current = _fetch_ib_price(ib_connection, symbol, Stock)
        price_source = "ibkr"
        if current is None:
            current = _latest_flatfile_close(symbol)
            price_source = "flatfile-close"
            if current is not None:
                logger.warning(
                    "%s: IBKR price unavailable — monitoring on flat-file "
                    "close %.2f (last completed session)", symbol, current,
                )
        if current is None:
            logger.error(
                "%s: NO price from any source — position is being monitored "
                "blind; stop at %s cannot trigger this cycle",
                symbol, f"{atr_stop:.2f}" if atr_stop else "?",
            )
            continue

        # Sanity check: price must be within ±20%/+50% of entry to be trusted.
        # Pre-market bid/ask midpoints can be wildly stale; reject them rather
        # than risk a false stop-out (e.g. VOYA $80.13 vs actual low $87.60).
        if entry_price and not (entry_price * 0.80 <= current <= entry_price * 1.50):
            logger.warning(
                "%s: price %.2f rejected as invalid (entry=%.2f) — skipping stop check",
                symbol, current, entry_price,
            )
            continue

        unrealized = (current - entry_price) / entry_price if entry_price else 0.0
        days       = _trading_days_elapsed(date_added, today)

        # 2R target price — stored value preferred; fall back to computed
        target_price = entry.get("target_price")
        if target_price is None and entry_price and atr_stop and entry_price > atr_stop:
            target_price = entry_price + 2 * (entry_price - atr_stop)

        # Effective stop: tighten the entry→stop distance by the regime factor
        # (factor 1.0 → unchanged; 0.5 in BEAR → stop halfway between entry and
        # original stop, exiting weak positions sooner)
        try:
            from quantlab.risk.regime_policy import effective_stop_price
            effective_stop = effective_stop_price(entry_price, atr_stop, _stop_factor)
        except Exception:
            effective_stop = atr_stop

        if effective_stop and current <= effective_stop:
            new_status = "stopped_out"
        elif target_price and current >= target_price:
            new_status = "target_hit"
            logger.info(
                "%s: 2R target hit at $%.2f — consider taking profits",
                symbol, current,
            )
        elif days >= 10:
            new_status = "expired"
        else:
            new_status = "watching"

        try:
            con = get_db()
            con.execute("""
                UPDATE watchlist
                SET current_price=?, unrealized_ret=?, days_on_watch=?,
                    status=?, date_updated=?
                WHERE watch_id=?
            """, [current, unrealized, days, new_status, today.isoformat(),
                  entry["watch_id"]])
            con.close()
        except Exception as e:
            print(f"[watchlist] update failed for {symbol}: {e}")
            continue

        updates.append({
            "symbol":        symbol,
            "current_price": current,
            "unrealized_ret": unrealized,
            "days":          days,
            "status":        new_status,
            "entry_price":   entry_price,
            "atr_stop":      atr_stop,
            "effective_stop": effective_stop,
        })

    return updates


def update_forward_return(
    watch_id: str,
    horizon_days: int,
    price: float,
    ret: float,
    db_path: str | None = None,
) -> None:
    """
    Record the realised price and return for a given forward-return horizon.

    Called by track_forward_returns.py when a watchlist entry reaches its
    1-day, 3-day, or 5-day measurement point.

    Args:
        watch_id:     Primary key of the watchlist entry.
        horizon_days: 1, 3, or 5.
        price:        Closing price on the horizon date.
        ret:          (price − entry_price) / entry_price.
        db_path:      Override DB path (for testing).
    """
    col_price = {1: "price_1d", 3: "price_3d", 5: "price_5d"}.get(horizon_days)
    col_ret   = {1: "realized_ret_1d", 3: "realized_ret_3d",
                 5: "realized_ret_5d"}.get(horizon_days)
    if col_price is None:
        return
    try:
        import duckdb
        path = db_path or str(DB_PATH)
        con = duckdb.connect(path)
        from quantlab.storage import _ensure_schema
        _ensure_schema(con)
        con.execute(
            f"UPDATE watchlist SET {col_price}=?, {col_ret}=? WHERE watch_id=?",
            [price, ret, watch_id],
        )
        con.close()
    except Exception as e:
        print(f"[watchlist] forward return update failed ({watch_id}, {horizon_days}d): {e}")


# ── Institutional pre-breakout watchlist ──────────────────────────────────────

_IWL_COLS = [
    "symbol", "first_seen", "last_seen", "consecutive_days", "stage",
    "conviction_score", "entry_price", "options_signal", "volume_dry_up",
    "earnings_score", "peg_score", "breakout_volume_score", "tape", "notes",
    "updated_at", "explosion_score", "explosion_components",
    "conviction_components", "breakout_volume_ratio",
]


class InstitutionalWatchlist:
    """
    Persistent multi-day tracking of pre-breakout candidates.

    Backed by the DuckDB `institutional_watchlist` table.  Each symbol is
    keyed once (PRIMARY KEY symbol) and updated in place — `consecutive_days`
    tracks how many daily scans the symbol has appeared in consecutively.

    Conviction bonus: +0.05 per consecutive day, capped at +0.20.

    Usage::

        iwl = InstitutionalWatchlist()
        iwl.upsert("NVDA", scan_result)
        for entry in iwl.get_multi_day(min_days=2):
            print(entry["symbol"], entry["consecutive_days"])
        iwl.remove_stale(max_days_inactive=5)
    """

    def __init__(self, db_path: str | None = None) -> None:
        import duckdb
        from quantlab.storage import _ensure_schema
        self._path = str(db_path) if db_path else str(DB_PATH)
        # Ensure schema exists on construction
        con = duckdb.connect(self._path)
        _ensure_schema(con)
        con.close()

    def _con(self):
        import duckdb
        from quantlab.storage import _ensure_schema
        con = duckdb.connect(self._path)
        _ensure_schema(con)
        return con

    # ── upsert ─────────────────────────────────────────────────────────────────

    def upsert(self, symbol: str, scan_result) -> dict:
        """
        Insert a new entry or update an existing one.

        If `symbol` is new: consecutive_days=1, first_seen=today.
        If `symbol` exists and last_seen < today: consecutive_days += 1.
        If `symbol` exists and last_seen == today: scores updated, days unchanged.

        Returns a dict with the final stored values.
        """
        today = date.today()
        today_str = today.isoformat()

        base_conviction = getattr(scan_result, "conviction_score", 0.0)
        stage           = getattr(scan_result, "stage", 0)

        # Unknown (0), topping (3), declining (4): never store anywhere
        if stage in (0, 3, 4):
            return {"symbol": symbol, "consecutive_days": 0, "conviction_score": 0.0}

        # Stage 1 (basing) → basing_watchlist; Stage 2 (advancing) → institutional_watchlist
        _table = "basing_watchlist" if stage == 1 else "institutional_watchlist"

        entry_price     = getattr(scan_result, "entry_close", None)
        _edgar_accel    = getattr(scan_result, "edgar_acceleration", None)
        earnings_score  = (
            _edgar_accel if _edgar_accel is not None
            else getattr(scan_result, "earnings_acceleration", None)
        )
        # None = not computable — stored as NULL, never coerced to 0.0
        peg_score       = getattr(scan_result, "peg_score", None)
        bvs             = getattr(scan_result, "breakout_volume_score", 0.0)
        # Raw Weinstein ratio — None = not measurable, stored as NULL
        bv_ratio        = getattr(scan_result, "breakout_volume_ratio", None)
        # Volume dry-up proxy: low volume_trend + decent absorption signals drying
        vol_trend       = getattr(scan_result, "volume_trend", 1.0)
        absorption      = getattr(scan_result, "absorption", 0.0)
        volume_dry_up   = vol_trend < 0.4 and absorption >= 0.3
        _opt_score      = getattr(scan_result, "options_score", None)
        options_signal  = (
            getattr(scan_result, "unusual_options_score", 0.0) >= 0.5
            or (_opt_score is not None and _opt_score >= 0.6)
        )
        explosion_score      = getattr(scan_result, "explosion_score", None)
        explosion_components = getattr(scan_result, "explosion_components", 0)

        # Raw conviction inputs (JSON) — candidate record keeps the data a
        # future conviction re-weighting needs
        components_json = ""
        try:
            import json as _json
            from quantlab.execution import conviction_components as _cc
            components_json = _json.dumps(_cc(scan_result))
        except Exception:
            pass

        tape = ""
        try:
            from quantlab.signals.breadth import get_latest_snapshot
            snap = get_latest_snapshot()
            if snap:
                tape = snap.tape
        except Exception:
            pass

        try:
            con = self._con()
            row = con.execute(
                f"SELECT consecutive_days, last_seen, options_signal, updated_at "
                f"FROM {_table} WHERE symbol = ?",
                [symbol],
            ).fetchone()

            if row is None:
                consecutive_days = 1
                stored_conviction = min(1.0, base_conviction + 0.05 * consecutive_days)
                con.execute(
                    f"""
                    INSERT INTO {_table}
                        (symbol, first_seen, last_seen, consecutive_days, stage,
                         conviction_score, entry_price, options_signal, volume_dry_up,
                         earnings_score, peg_score, breakout_volume_score, tape, notes,
                         updated_at, explosion_score, explosion_components,
                         conviction_components, breakout_volume_ratio)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', CURRENT_TIMESTAMP, ?, ?, ?, ?)
                    """,
                    [
                        symbol, today_str, today_str, consecutive_days, stage,
                        stored_conviction, entry_price, options_signal, volume_dry_up,
                        earnings_score, peg_score, bvs, tape, explosion_score,
                        explosion_components, components_json, bv_ratio,
                    ],
                )
            else:
                prev_days, last_seen, prev_opts, prev_updated = row
                if isinstance(last_seen, str):
                    last_seen = date.fromisoformat(last_seen)
                elif hasattr(last_seen, "date"):
                    last_seen = last_seen.date()
                # Increment only when appearing on a new trading day
                consecutive_days = prev_days + 1 if last_seen < today else prev_days
                bonus = min(0.20, 0.05 * consecutive_days)
                stored_conviction = min(1.0, base_conviction + bonus)

                # Preserve an options_signal set earlier today by the intraday
                # monitor (set_options_signal).  The evening scan recomputes the
                # flag from flat files that often aren't published yet at scan
                # time, so overwriting here would erase every intraday detection
                # before the report is generated.  Flags older than today are
                # stale scan-time values and are recomputed normally.
                if isinstance(prev_updated, str):
                    prev_updated_date = date.fromisoformat(prev_updated[:10])
                elif hasattr(prev_updated, "date"):
                    prev_updated_date = prev_updated.date()
                else:
                    prev_updated_date = prev_updated
                if bool(prev_opts) and prev_updated_date == today and not options_signal:
                    options_signal = True
                    # Re-apply the monitor's conviction bonus the recompute dropped
                    stored_conviction = min(1.0, stored_conviction + 0.08)
                con.execute(
                    f"""
                    UPDATE {_table} SET
                        last_seen=?, consecutive_days=?, stage=?,
                        conviction_score=?, entry_price=?, options_signal=?,
                        volume_dry_up=?, earnings_score=?, peg_score=?,
                        breakout_volume_score=?, tape=?, updated_at=CURRENT_TIMESTAMP,
                        explosion_score=?, explosion_components=?,
                        conviction_components=?, breakout_volume_ratio=?
                    WHERE symbol=?
                    """,
                    [
                        today_str, consecutive_days, stage,
                        stored_conviction, entry_price, options_signal,
                        volume_dry_up, earnings_score, peg_score, bvs, tape,
                        explosion_score, explosion_components, components_json,
                        bv_ratio, symbol,
                    ],
                )
            con.close()
        except Exception as exc:
            print(f"[{_table}] upsert failed for {symbol}: {exc}")
            consecutive_days = 1
            stored_conviction = base_conviction

        return {
            "symbol": symbol,
            "consecutive_days": consecutive_days,
            "conviction_score": stored_conviction,
        }

    # ── queries ────────────────────────────────────────────────────────────────

    def get_candidates(self, min_consecutive_days: int = 1) -> list[dict]:
        """All entries with consecutive_days >= threshold, sorted by days then conviction."""
        try:
            con = self._con()
            rows = con.execute(
                f"SELECT {', '.join(_IWL_COLS)} FROM institutional_watchlist "
                "WHERE consecutive_days >= ? "
                "ORDER BY consecutive_days DESC, conviction_score DESC",
                [min_consecutive_days],
            ).fetchall()
            con.close()
            return [dict(zip(_IWL_COLS, r)) for r in rows]
        except Exception as exc:
            print(f"[institutional_watchlist] get_candidates failed: {exc}")
            return []

    def get_multi_day(self, min_days: int = 2) -> list[dict]:
        """Candidates appearing on min_days+ consecutive scans (highest priority)."""
        return self.get_candidates(min_consecutive_days=min_days)

    def get_basing_candidates(self, min_consecutive_days: int = 1) -> list[dict]:
        """Stage 1 basing stocks from basing_watchlist, sorted by days then conviction."""
        try:
            con = self._con()
            rows = con.execute(
                f"SELECT {', '.join(_IWL_COLS)} FROM basing_watchlist "
                "WHERE consecutive_days >= ? "
                "ORDER BY consecutive_days DESC, conviction_score DESC",
                [min_consecutive_days],
            ).fetchall()
            con.close()
            return [dict(zip(_IWL_COLS, r)) for r in rows]
        except Exception as exc:
            print(f"[basing_watchlist] get_basing_candidates failed: {exc}")
            return []

    def remove_stale(self, max_days_inactive: int = 5) -> int:
        """
        Remove symbols not seen in max_days_inactive trading days.

        Days when the universe build was refused by the sanity gate are
        NEUTRAL: they count toward neither activity nor inactivity, so a
        symbol cannot lose its streak because our build was degenerate.
        Returns the count of removed rows.
        """
        today = date.today()
        try:
            con = self._con()
            degenerate_days = _degenerate_build_dates(con)
            rows = con.execute(
                "SELECT symbol, last_seen FROM institutional_watchlist"
            ).fetchall()
            to_remove: list[str] = []
            for sym, last_seen in rows:
                if isinstance(last_seen, str):
                    last_seen = date.fromisoformat(last_seen)
                elif hasattr(last_seen, "date"):
                    last_seen = last_seen.date()
                if _trading_days_elapsed(
                    last_seen, today, skip_dates=degenerate_days,
                ) > max_days_inactive:
                    to_remove.append(sym)
            for sym in to_remove:
                con.execute(
                    "DELETE FROM institutional_watchlist WHERE symbol = ?", [sym]
                )
            con.close()
            return len(to_remove)
        except Exception as exc:
            print(f"[institutional_watchlist] remove_stale failed: {exc}")
            return 0

    def set_options_signal(self, symbol: str, bonus: float = 0.08) -> None:
        """Flag a symbol as having unusual options activity and add a conviction bonus."""
        try:
            con = self._con()
            con.execute(
                """
                UPDATE institutional_watchlist SET
                    options_signal=TRUE,
                    conviction_score=LEAST(1.0, conviction_score + ?),
                    updated_at=CURRENT_TIMESTAMP
                WHERE symbol=?
                """,
                [bonus, symbol],
            )
            con.close()
        except Exception as exc:
            print(f"[institutional_watchlist] set_options_signal failed for {symbol}: {exc}")

    def to_dataframe(self):
        """Return the full watchlist as a pandas DataFrame."""
        import pandas as pd
        rows = self.get_candidates()
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=_IWL_COLS)


def get_watchlist_summary(db_path: str | None = None) -> dict[str, Any]:
    """
    Aggregate statistics on the watchlist — hit rates, avg returns by horizon.

    Returns a dict with keys: total, by_status, ret_1d, ret_3d, ret_5d,
    hit_1d, hit_3d, hit_5d.
    """
    try:
        import duckdb
        path = db_path or str(DB_PATH)
        con = duckdb.connect(path)
        from quantlab.storage import _ensure_schema
        _ensure_schema(con)

        total = con.execute("SELECT COUNT(*) FROM watchlist").fetchone()[0]
        by_status = dict(
            con.execute(
                "SELECT status, COUNT(*) FROM watchlist GROUP BY status"
            ).fetchall()
        )

        def _agg(col):
            row = con.execute(
                f"SELECT AVG({col}), "
                f"       SUM(CASE WHEN {col} > 0 THEN 1 ELSE 0 END) * 1.0 / "
                f"       NULLIF(COUNT({col}), 0) "
                f"FROM watchlist WHERE {col} IS NOT NULL"
            ).fetchone()
            return {"avg": row[0], "hit_rate": row[1]}

        summary = {
            "total": total,
            "by_status": by_status,
            "ret_1d": _agg("realized_ret_1d"),
            "ret_3d": _agg("realized_ret_3d"),
            "ret_5d": _agg("realized_ret_5d"),
        }
        con.close()
        return summary
    except Exception as e:
        print(f"[watchlist] summary failed: {e}")
        return {}
