"""
scripts/track_forward_returns.py — Daily post-close forward-return tracker.

Run after market close each trading day.  For every active watchlist entry
that has reached its 1-day, 3-day, or 5-day measurement horizon, this
script fetches the closing price from IBKR, records the realised return in
DuckDB, and prints a summary showing what delivered and what failed.

This is the feedback loop that proves (or disproves) whether conviction
scores are predictive.  After accumulating 30+ entries, query DuckDB to
group by signal_layers and measure hit rates per layer.

Usage:
    python scripts/track_forward_returns.py
    python scripts/track_forward_returns.py --no-ibkr   # skip price fetch, just print
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from datetime import date

from quantlab.utils import get_config, setup_logging
from quantlab.watchlist import (
    _trading_days_elapsed,
    get_active_watchlist,
    update_forward_return,
    update_watchlist_prices,
    get_watchlist_summary,
)


HORIZONS = [1, 3, 5]          # trading-day horizons to track
HORIZON_LABELS = {1: "1D", 3: "3D", 5: "5D"}


def _fmt(v: float | None, pct: bool = True) -> str:
    if v is None:
        return "  --  "
    if pct:
        return f"{v * 100:+.2f}%"
    return f"{v:.2f}"


def fetch_closing_prices(
    symbols: list[str],
    ib_connection,
) -> dict[str, float]:
    """Fetch delayed closing prices for a list of symbols via IBKR."""
    from ib_insync import Stock

    ib_connection.reqMarketDataType(3)
    prices: dict[str, float] = {}

    for symbol in symbols:
        try:
            stock = Stock(symbol, "SMART", "USD")
            qualified = ib_connection.qualifyContracts(stock)
            if not qualified:
                continue
            ticker = ib_connection.reqTickers(qualified[0])[0]
            for candidate in [ticker.marketPrice(), ticker.last, ticker.close]:
                if candidate is not None and candidate == candidate and candidate > 0:
                    prices[symbol] = float(candidate)
                    break
        except Exception:
            pass

    return prices


def update_returns_for_today(
    ib_connection,
    today: date | None = None,
) -> list[dict]:
    """
    For every active watchlist entry that has reached a return horizon,
    fetch the current price and record the realised return.

    An entry is at horizon N when trading_days_since_add == N and the
    corresponding realized_ret_Nd column is still NULL.

    Returns list of dicts: one per (entry, horizon) updated.
    """
    today     = today or date.today()
    all_rows  = get_active_watchlist()

    # Also include recently expired/stopped entries that might still need return capture
    import duckdb
    from quantlab.storage import DB_PATH, _ensure_schema
    con = duckdb.connect(str(DB_PATH))
    _ensure_schema(con)
    recent_closed = con.execute("""
        SELECT watch_id, symbol, date_added, entry_price,
               realized_ret_1d, realized_ret_3d, realized_ret_5d, status
        FROM watchlist
        WHERE status IN ('stopped_out', 'expired')
          AND date_added >= ?
    """, [(today - __import__("datetime").timedelta(days=10)).isoformat()]).fetchall()
    con.close()

    # Build a unified list (active + recently closed missing return data)
    _CLOSED_COLS = ["watch_id", "symbol", "date_added", "entry_price",
                    "realized_ret_1d", "realized_ret_3d", "realized_ret_5d", "status"]
    need_update: list[dict] = list(all_rows)
    for row in recent_closed:
        d = dict(zip(_CLOSED_COLS, row))
        if not any(e["watch_id"] == d["watch_id"] for e in need_update):
            need_update.append(d)

    if not need_update:
        return []

    # Determine which entries need price fetches
    targets: dict[str, list[tuple]] = {}  # symbol → [(watch_id, entry_price, horizon)]
    for entry in need_update:
        date_added = entry["date_added"]
        if isinstance(date_added, str):
            from datetime import date as _date
            date_added = _date.fromisoformat(date_added)
        days_elapsed = _trading_days_elapsed(date_added, today)

        for h in HORIZONS:
            ret_key = f"realized_ret_{h}d"
            if days_elapsed >= h and entry.get(ret_key) is None:
                sym = entry["symbol"]
                if sym not in targets:
                    targets[sym] = []
                targets[sym].append((entry["watch_id"], entry["entry_price"], h))

    if not targets:
        return []

    # Fetch closing prices for all symbols that need them
    symbols_needed = list(targets.keys())
    prices = fetch_closing_prices(symbols_needed, ib_connection)

    updates: list[dict] = []
    for symbol, tasks in targets.items():
        current = prices.get(symbol)
        if current is None:
            continue
        for watch_id, entry_price, horizon in tasks:
            ret = (current - entry_price) / entry_price if entry_price else 0.0
            update_forward_return(watch_id, horizon, current, ret)
            updates.append({
                "watch_id":    watch_id,
                "symbol":      symbol,
                "horizon":     horizon,
                "price":       current,
                "ret":         ret,
                "entry_price": entry_price,
            })

    return updates


def print_returns_summary(updates: list[dict]) -> None:
    """Print a compact table of forward-return updates."""
    if not updates:
        print("  No return horizons reached today.")
        return

    print(f"\n  {'watch_id':<24}  {'horizon':>8}  {'entry':>8}  "
          f"{'close':>8}  {'return':>8}")
    print(f"  {'─'*62}")

    for u in sorted(updates, key=lambda x: (x["horizon"], x["symbol"])):
        ret_str = _fmt(u["ret"])
        direction = "▲" if u["ret"] > 0 else "▼"
        print(
            f"  {u['watch_id']:<24}  {HORIZON_LABELS[u['horizon']]:>8}  "
            f"{u['entry_price']:>8.2f}  {u['price']:>8.2f}  {ret_str:>8}  {direction}"
        )


def print_full_summary() -> None:
    """Print current watchlist state and all available forward-return statistics."""
    summary = get_watchlist_summary()
    if not summary:
        print("  No watchlist data.")
        return

    print(f"\n  {'─'*56}")
    print(f"  Watchlist totals: {summary.get('total', 0)}")
    by_status = summary.get("by_status", {})
    for status, n in sorted(by_status.items()):
        print(f"    {status:<16} : {n}")

    for label, key in [("1D", "ret_1d"), ("3D", "ret_3d"), ("5D", "ret_5d")]:
        agg = summary.get(key, {})
        avg = agg.get("avg")
        hit = agg.get("hit_rate")
        if avg is not None:
            print(f"  {label} avg_ret={_fmt(avg)}  hit_rate={_fmt(hit)}")
        else:
            print(f"  {label} — no data yet")


def main() -> None:
    setup_logging(level="WARNING")
    ibkr_cfg = get_config("ibkr")

    parser = ArgumentParser(description="Track forward returns for watchlist entries.")
    parser.add_argument("--no-ibkr", action="store_true",
                        help="Skip IBKR price fetch; just print current state")
    parser.add_argument("--host",      default=ibkr_cfg["host"])
    parser.add_argument("--port",      type=int, default=ibkr_cfg["port"])
    parser.add_argument("--client-id", type=int, default=ibkr_cfg["spot_client_id"])
    args = parser.parse_args()

    today = date.today()
    print(f"\n{'='*60}")
    print(f"  Forward Return Tracker  {today}")
    print(f"{'='*60}")

    active = get_active_watchlist()
    print(f"\n  Active watchlist: {len(active)} entries watching")

    if active:
        print(f"\n  {'symbol':<8}  {'added':>12}  {'conv':>5}  "
              f"{'days':>5}  {'layers':<28}  {'stop':>8}")
        print(f"  {'─'*76}")
        for e in active:
            date_added = e["date_added"]
            if isinstance(date_added, str):
                from datetime import date as _d
                date_added = _d.fromisoformat(date_added)
            days = _trading_days_elapsed(date_added, today)
            print(
                f"  {e['symbol']:<8}  {str(e['date_added']):>12}  "
                f"{e['conviction_score']:>5.2f}  {days:>5}  "
                f"{(e['signal_layers'] or ''):<28}  "
                f"{e['atr_stop'] or 0:>8.2f}"
            )

    if args.no_ibkr:
        print("\n  --no-ibkr: skipping price fetch")
        print_full_summary()
        return

    # ── Fetch prices and update ────────────────────────────────────────────────
    from quantlab.providers.ibkr import ping_tws
    if not ping_tws(args.host, args.port):
        print(f"\n  TWS not reachable at {args.host}:{args.port} — skipping price fetch")
        print_full_summary()
        return

    from ib_insync import IB
    ib = IB()
    try:
        ib.connect(args.host, args.port, clientId=args.client_id, timeout=10)
        print(f"\n  Connected to IBKR — updating prices ...")

        # Update live prices and statuses
        price_updates = update_watchlist_prices(ib)
        if price_updates:
            print(f"\n  Price updates ({len(price_updates)} symbols):")
            for u in price_updates:
                ret_str = _fmt(u["unrealized_ret"])
                stopped = "  ← STOPPED OUT" if u["status"] == "stopped_out" else ""
                expired  = "  ← EXPIRED"    if u["status"] == "expired"     else ""
                print(f"    {u['symbol']:<8}  "
                      f"entry={u['entry_price']:.2f}  current={u['current_price']:.2f}  "
                      f"unreal={ret_str}  days={u['days']}{stopped}{expired}")

        # Record forward returns for entries hitting their horizon today
        print(f"\n  Checking return horizons ...")
        return_updates = update_returns_for_today(ib, today)
        if return_updates:
            print(f"  {len(return_updates)} horizon(s) recorded:")
            print_returns_summary(return_updates)
        else:
            print("  No horizons reached today.")

    finally:
        if ib.isConnected():
            ib.disconnect()

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n  Cumulative performance:")
    print_full_summary()
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
