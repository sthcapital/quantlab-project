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

from datetime import date, timedelta
from typing import Any

from quantlab.storage import DB_PATH, get_db


# ── Helpers ────────────────────────────────────────────────────────────────────

MIN_CONVICTION_FOR_WATCHLIST = 0.70


def _trading_days_elapsed(from_date: date, to_date: date | None = None) -> int:
    """Count Mon–Fri trading days between from_date (exclusive) and to_date (inclusive)."""
    if to_date is None:
        to_date = date.today()
    count = 0
    current = from_date + timedelta(days=1)
    while current <= to_date:
        if current.weekday() < 5:   # Monday=0 … Friday=4
            count += 1
        current += timedelta(days=1)
    return count


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

    opt = getattr(scan_result, "options_conviction", 0.0)
    if opt >= 0.6:
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

    Returns:
        True if the entry was inserted; False if skipped (below threshold or
        already present for today).
    """
    if scan_result.conviction_score < min_conviction:
        return False

    today      = date.today()
    watch_id   = f"{scan_result.symbol}_{today.isoformat()}"
    layers     = _layers_fired(scan_result)

    try:
        con = get_db()
        # INSERT OR IGNORE keeps the first entry if run twice today
        con.execute("""
            INSERT OR IGNORE INTO watchlist (
                watch_id, symbol, date_added, entry_price, atr_stop,
                conviction_score, signal_layers, lookback, signal_type,
                status, date_updated, breadth_override_note
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'watching', ?, ?)
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
            today.isoformat(),
            note,
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
        "breadth_override_note",
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


def update_watchlist_prices(ib_connection) -> list[dict[str, Any]]:
    """
    Refresh current prices for all active watchlist entries via IBKR.

    For each entry:
    - Fetches the current spot price using get_spot_price().
    - Computes unrealized_ret = (current − entry) / entry.
    - Sets status = 'stopped_out' if current_price ≤ atr_stop.
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

    today   = date.today()
    updates: list[dict] = []

    for entry in active:
        symbol      = entry["symbol"]
        entry_price = entry["entry_price"]
        atr_stop    = entry["atr_stop"]
        date_added  = entry["date_added"]

        if isinstance(date_added, str):
            date_added = date.fromisoformat(date_added)

        # Fetch current price
        try:
            ib_connection.reqMarketDataType(3)
            stock = Stock(symbol, "SMART", "USD")
            qualified = ib_connection.qualifyContracts(stock)
            if not qualified:
                continue
            ticker = ib_connection.reqTickers(qualified[0])[0]
            current = None
            for candidate in [ticker.marketPrice(), ticker.last, ticker.close]:
                if candidate is not None and candidate == candidate and candidate > 0:
                    current = float(candidate)
                    break
            if current is None:
                continue
        except Exception:
            continue

        unrealized = (current - entry_price) / entry_price if entry_price else 0.0
        days       = _trading_days_elapsed(date_added, today)

        if atr_stop and current <= atr_stop:
            new_status = "stopped_out"
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
