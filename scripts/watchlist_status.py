"""
scripts/watchlist_status.py — Watchlist terminal dashboard.

Shows all active and completed watchlist entries, unrealised returns,
distance from ATR stop, signal layers that fired, and running hit-rate
statistics.  Optionally fetches live prices from IBKR; falls back to
the last cached price when --no-ibkr is passed.

Usage:
    python scripts/watchlist_status.py                # try live IBKR prices
    python scripts/watchlist_status.py --no-ibkr      # cached prices only
    python scripts/watchlist_status.py --host 172.23.208.1
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from datetime import date
from typing import Any

# ── Formatting helpers (pure functions, fully testable offline) ────────────────

def fmt_pct(v: float | None, decimals: int = 2) -> str:
    """Format a decimal fraction as a percentage string."""
    if v is None:
        return "    --"
    return f"{v * 100:+.{decimals}f}%"


def fmt_return(v: float | None) -> str:
    """Return with ✓ prefix when positive, spaces otherwise."""
    if v is None:
        return "      --"
    label = "✓ " if v > 0 else "  "
    return f"{label}{v * 100:+.2f}%"


def stop_distance(current: float | None, stop: float | None) -> float | None:
    """% distance from ATR stop: positive = above stop, negative = below."""
    if current is None or not stop or stop <= 0:
        return None
    return (current - stop) / current * 100


def near_stop(current: float | None, stop: float | None,
              threshold_pct: float = 2.0) -> bool:
    """Return True when current price is within threshold_pct% of the stop."""
    dist = stop_distance(current, stop)
    return dist is not None and dist < threshold_pct


def fmt_layers(layers: str | None) -> str:
    """Abbreviate long layers string to fit in table column."""
    if not layers:
        return "—"
    # Shorten common prefixes so the column stays readable
    short = (layers
             .replace("MULTI_LB", "MLB")
             .replace("NEWS:", "N:")
             .replace("VOL_CHAR", "VC"))
    return short[:30]


def fmt_dur(secs: int) -> str:
    """Format seconds into 'Xh Ym' string."""
    h, r = divmod(max(0, secs), 3600)
    m = r // 60
    return f"{h}h {m}m" if h else f"{m}m"


# ── Price fetching ─────────────────────────────────────────────────────────────

def _prices_from_cache(symbols: list[str]) -> dict[str, float]:
    """Return last known close from parquet cache for each symbol."""
    from quantlab.storage import DATA_PROCESSED
    import pyarrow.parquet as pq
    prices: dict[str, float] = {}
    for sym in symbols:
        path = DATA_PROCESSED / f"{sym}_bars.parquet"
        if not path.exists():
            continue
        try:
            rows = pq.read_table(path).to_pydict()
            if rows.get("close"):
                prices[sym] = float(rows["close"][-1])
        except Exception:
            pass
    return prices


def _prices_from_ibkr(symbols: list[str], host: str, port: int,
                       client_id: int) -> dict[str, float]:
    """Fetch delayed market prices from IBKR for a list of symbols."""
    from ib_insync import IB, Stock
    prices: dict[str, float] = {}
    ib = IB()
    try:
        ib.connect(host, port, clientId=client_id, timeout=10)
        ib.reqMarketDataType(3)
        for sym in symbols:
            try:
                stock = Stock(sym, "SMART", "USD")
                qualified = ib.qualifyContracts(stock)
                if not qualified:
                    continue
                ticker = ib.reqTickers(qualified[0])[0]
                for candidate in [ticker.marketPrice(), ticker.last, ticker.close]:
                    if candidate is not None and candidate == candidate and candidate > 0:
                        prices[sym] = float(candidate)
                        break
            except Exception:
                pass
    except Exception as e:
        print(f"  [IBKR] Connection failed: {e} — using cached prices", file=sys.stderr)
    finally:
        if ib.isConnected():
            ib.disconnect()
    return prices


def fetch_prices(symbols: list[str], use_ibkr: bool,
                 host: str, port: int, client_id: int) -> dict[str, float]:
    """
    Fetch current prices from IBKR when available, falling back to parquet cache.
    Merges both sources so cached fills gaps left by IBKR misses.
    """
    cached = _prices_from_cache(symbols)
    if not use_ibkr:
        return cached
    from quantlab.providers.ibkr import ping_tws
    if not ping_tws(host, port, timeout=3.0):
        return cached
    live = _prices_from_ibkr(symbols, host, port, client_id)
    return {**cached, **live}   # live prices overwrite cache


# ── Section renderers ──────────────────────────────────────────────────────────

_SEP  = "═" * 74
_DASH = "─" * 74


def _section(title: str) -> None:
    print(f"\n{_SEP}")
    print(f"  {title}")
    print(_SEP)


def print_active(entries: list[dict], prices: dict[str, float]) -> None:
    """Render the active (watching) watchlist entries."""
    _section("Active Watchlist  —  watching")

    if not entries:
        print("  (no active entries)")
        return

    hdr = (f"  {'#':>2}  {'symbol':<8}  {'added':>10}  {'entry':>7}  "
           f"{'current':>8}  {'return':>9}  {'days':>4}  "
           f"{'stop':>7}  {'dist':>6}  layers")
    print(hdr)
    print(f"  {_DASH}")

    for i, e in enumerate(entries, 1):
        sym          = e["symbol"]
        da           = date.fromisoformat(str(e["date_added"]))
        entry_p      = e["entry_price"]
        atr_stop     = e["atr_stop"] or 0.0
        conv         = e["conviction_score"]
        layers       = fmt_layers(e.get("signal_layers"))

        current = prices.get(sym) or e.get("current_price")
        if current and entry_p:
            ret = (current - entry_p) / entry_p
        else:
            ret = e.get("unrealized_ret")

        dist  = stop_distance(current, atr_stop)
        alarm = "  ⚠" if near_stop(current, atr_stop) else "   "
        ret_s = fmt_return(ret)
        cur_s = f"{current:>8.2f}" if current else "      --"
        dist_s = f"{dist:>+.1f}%" if dist is not None else "    --"

        from quantlab.watchlist import _trading_days_elapsed
        days = _trading_days_elapsed(da)

        print(
            f"  {i:>2}.  {sym:<8}  {str(da):>10}  {entry_p:>7.2f}  "
            f"{cur_s}  {ret_s}  {days:>4}  "
            f"{atr_stop:>7.2f}  {dist_s:>6}{alarm}  {layers}"
        )


def print_completed(entries: list[dict]) -> None:
    """Render expired and stopped-out entries with final returns."""
    _section("Completed Entries  —  expired / stopped_out")

    if not entries:
        print("  (no completed entries yet)")
        return

    hdr = (f"  {'symbol':<8}  {'added':>10}  {'status':<12}  "
           f"{'entry':>7}  {'1D ret':>8}  {'3D ret':>8}  {'5D ret':>8}  "
           f"{'days':>4}  layers")
    print(hdr)
    print(f"  {_DASH}")

    for e in entries:
        sym    = e["symbol"]
        da     = e["date_added"]
        status = e["status"]
        entry  = e["entry_price"]
        r1     = e.get("realized_ret_1d")
        r3     = e.get("realized_ret_3d")
        r5     = e.get("realized_ret_5d")
        days   = e.get("days_on_watch") or 0
        layers = fmt_layers(e.get("signal_layers"))

        stop_flag = "  🛑" if status == "stopped_out" else ""

        print(
            f"  {sym:<8}  {str(da):>10}  {status:<12}  "
            f"{entry:>7.2f}  {fmt_pct(r1):>8}  "
            f"{fmt_pct(r3):>8}  {fmt_return(r5):>8}  "
            f"{days:>4}  {layers}{stop_flag}"
        )


def print_statistics(summary: dict) -> None:
    """Render running performance statistics."""
    _section("Running Statistics")

    total     = summary.get("total", 0)
    by_status = summary.get("by_status", {})
    watching  = by_status.get("watching", 0)
    stopped   = by_status.get("stopped_out", 0)
    expired   = by_status.get("expired", 0)
    completed = stopped + expired

    print(f"  Total signals added    : {total}")
    print(f"  Currently watching     : {watching}")
    print(f"  Completed (exp+stop)   : {completed}  "
          f"({expired} expired, {stopped} stopped_out)")

    print()

    for label, key in [("1D", "ret_1d"), ("3D", "ret_3d"), ("5D", "ret_5d")]:
        agg  = summary.get(key, {})
        avg  = agg.get("avg")
        hit  = agg.get("hit_rate")
        if avg is None:
            print(f"  {label} avg / hit rate   : -- / --  (no data yet)")
        else:
            bar = "█" * int((hit or 0) * 10)
            print(f"  {label} avg / hit rate   : {avg*100:+.2f}%  /  "
                  f"{(hit or 0)*100:.0f}%  {bar}")

    # Best and worst completed 5D trades
    import duckdb
    from quantlab.storage import DB_PATH
    try:
        con = duckdb.connect(str(DB_PATH))
        rows = con.execute("""
            SELECT symbol, realized_ret_5d, entry_price, status
            FROM watchlist
            WHERE realized_ret_5d IS NOT NULL
            ORDER BY realized_ret_5d DESC
        """).fetchall()
        con.close()

        if rows:
            best  = rows[0]
            worst = rows[-1]
            print()
            print(f"  Best  completed trade  : {best[0]:<8} "
                  f"{fmt_return(best[1])}  (entry {best[2]:.2f})")
            print(f"  Worst completed trade  : {worst[0]:<8} "
                  f"{fmt_return(worst[1])}  (entry {worst[2]:.2f})")
    except Exception:
        pass

    print(f"\n  {_DASH}")
    if completed == 0:
        print("  Conviction validation begins after first entry expires (10 trading days).")
    elif completed < 10:
        print(f"  Sample size {completed} — directional only until ≥ 10 completed trades.")
    else:
        print(f"  Sample size {completed} — statistically meaningful.")


# ── Main ───────────────────────────────────────────────────────────────────────

def run_dashboard(use_ibkr: bool = False,
                  host: str = "172.23.208.1",
                  port: int = 7497,
                  client_id: int = 51) -> None:
    """
    Render the complete watchlist dashboard.
    Can be called from morning.sh or interactively.
    """
    import duckdb
    from quantlab.storage import DB_PATH, _ensure_schema
    from quantlab.watchlist import get_active_watchlist, get_watchlist_summary

    # ── Load all watchlist data ────────────────────────────────────────────────
    active = get_active_watchlist()

    con = duckdb.connect(str(DB_PATH))
    _ensure_schema(con)
    completed_rows = con.execute("""
        SELECT watch_id, symbol, date_added, entry_price, atr_stop,
               conviction_score, signal_layers, lookback,
               realized_ret_1d, realized_ret_3d, realized_ret_5d,
               current_price, unrealized_ret, days_on_watch, status, date_updated
        FROM watchlist
        WHERE status IN ('stopped_out', 'expired')
        ORDER BY date_added DESC
    """).fetchall()
    con.close()

    _COMP_COLS = ["watch_id","symbol","date_added","entry_price","atr_stop",
                  "conviction_score","signal_layers","lookback",
                  "realized_ret_1d","realized_ret_3d","realized_ret_5d",
                  "current_price","unrealized_ret","days_on_watch","status","date_updated"]
    completed = [dict(zip(_COMP_COLS, r)) for r in completed_rows]

    # ── Fetch prices ───────────────────────────────────────────────────────────
    all_syms = list({e["symbol"] for e in active})
    prices   = fetch_prices(all_syms, use_ibkr, host, port, client_id)

    # ── Summary ────────────────────────────────────────────────────────────────
    now_str = __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{_SEP}")
    print(f"  QuantLab Watchlist Dashboard  —  {now_str}")
    price_src = "live IBKR" if use_ibkr and prices else "parquet cache"
    print(f"  Price source: {price_src}")
    print(_SEP)

    # ── Sections ───────────────────────────────────────────────────────────────
    print_active(active, prices)
    print_completed(completed)
    print_statistics(get_watchlist_summary())
    print()


def main() -> None:
    from quantlab.utils import get_config
    ibkr = get_config("ibkr")

    parser = ArgumentParser(description="Watchlist terminal dashboard.")
    parser.add_argument("--no-ibkr",   action="store_true",
                        help="Use cached prices only; skip IBKR connection")
    parser.add_argument("--host",      default=ibkr["host"])
    parser.add_argument("--port",      type=int, default=ibkr["port"])
    parser.add_argument("--client-id", type=int, default=ibkr["spot_client_id"])
    args = parser.parse_args()

    run_dashboard(
        use_ibkr   = not args.no_ibkr,
        host       = args.host,
        port       = args.port,
        client_id  = args.client_id,
    )


if __name__ == "__main__":
    main()
