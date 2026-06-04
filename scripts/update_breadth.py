"""
scripts/update_breadth.py — Daily post-close breadth updater.

Fetches grouped daily data for all US stocks from Polygon.io, computes
the complete BreadthSnapshot including rolling McClellan Oscillator and
10-day ratio, stores to DuckDB, and prints a one-line breadth summary.

Run after market close every trading day.  Results are cached to parquet
so re-running the same date is instant.

Usage:
    python scripts/update_breadth.py                      # today
    python scripts/update_breadth.py --date 2026-06-03    # specific date
    python scripts/update_breadth.py --no-polygon         # DuckDB only (re-compute rolling)
"""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import date, timedelta

from quantlab.signals.breadth import (
    compute_market_breadth,
    rolling_breadth,
    save_breadth_snapshot,
    load_recent_snapshots,
    get_latest_snapshot,
    BreadthSnapshot,
)
from quantlab.market_calendar import is_market_open
from quantlab.utils import setup_logging


def _prev_trading_day(d: date) -> date:
    """Return the previous Mon-Fri that is not a US market holiday."""
    prev = d - timedelta(days=1)
    while prev.weekday() >= 5 or not is_market_open(prev):
        prev -= timedelta(days=1)
    return prev


def fetch_and_store(
    trade_date: date,
    polygon_provider,
    use_prev: bool = True,
) -> BreadthSnapshot | None:
    """
    Fetch today's grouped daily from Polygon, compute breadth, store to DuckDB.

    Returns the populated BreadthSnapshot (rolling fields added separately).
    """
    print(f"  Fetching grouped daily for {trade_date} ...")
    try:
        today_data = polygon_provider.get_grouped_daily(trade_date)
    except Exception as exc:
        print(f"  ERROR fetching {trade_date}: {exc}")
        return None

    if not today_data:
        print(f"  No data returned for {trade_date} (market closed or holiday?)")
        return None

    # Previous day data for close-to-close returns
    prev_data = None
    if use_prev:
        prev_date = _prev_trading_day(trade_date)
        try:
            prev_data = polygon_provider.get_grouped_daily(prev_date)
        except Exception:
            pass  # falls back to intraday returns

    snapshot = compute_market_breadth(
        trade_date  = trade_date,
        today_data  = today_data,
        prev_data   = prev_data,
    )
    return snapshot


def recompute_rolling(days: int = 60) -> list[BreadthSnapshot]:
    """
    Load recent snapshots from DuckDB, recompute rolling metrics, save back.
    Called after inserting a new snapshot so McClellan etc. are up to date.
    """
    snapshots = load_recent_snapshots(days)
    if not snapshots:
        return []
    rolling_breadth(snapshots)        # adds ratio_10d, mcclellan_*, ad_line, tape
    for s in snapshots:
        save_breadth_snapshot(s)      # upsert with updated rolling fields
    return snapshots


def main() -> None:
    setup_logging(level="WARNING")

    parser = ArgumentParser(description="Update market breadth data from Polygon.io.")
    parser.add_argument("--date", default=None, help="Trading date YYYY-MM-DD (default: today)")
    parser.add_argument("--no-polygon", action="store_true",
                        help="Skip Polygon fetch; only recompute rolling from DuckDB")
    parser.add_argument("--days-rolling", type=int, default=60,
                        help="Days of history to use for rolling metrics (default 60)")
    args = parser.parse_args()

    trade_date = date.fromisoformat(args.date) if args.date else date.today()

    print(f"\n{'='*62}")
    print(f"  Breadth Update  —  {trade_date}")
    print(f"{'='*62}")

    if not args.no_polygon:
        api_key = __import__("os").environ.get("POLYGON_API_KEY", "")
        if not api_key:
            print("  WARNING: POLYGON_API_KEY not set — skipping Polygon fetch")
            args.no_polygon = True
        else:
            from quantlab.providers.polygon import PolygonProvider
            provider = PolygonProvider(api_key=api_key)
            snap = fetch_and_store(trade_date, provider)
            if snap:
                print(f"  Raw:  A={snap.advances} D={snap.declines}  "
                      f"up4%={snap.up_4pct_count} dn4%={snap.down_4pct_count}  "
                      f"total={snap.total_stocks}")
                save_breadth_snapshot(snap)

    # Recompute rolling metrics from stored history
    print(f"  Recomputing rolling metrics ({args.days_rolling} days) ...")
    updated = recompute_rolling(args.days_rolling)

    # Print latest snapshot summary
    latest = get_latest_snapshot()
    if latest:
        print()
        print(f"  {latest.summary_line()}")
        print()
        print(f"  NH/NL ratio    : {latest.new_high_low_ratio:.2f}")
        print(f"  pct>200SMA     : {latest.pct_above_200sma:.1f}%")
        print(f"  Summation Index: {latest.mcclellan_summation or '--'}")
        print(f"  AD Line        : {latest.ad_line or '--'}")

        if latest.breadth_override if hasattr(latest, 'breadth_override') else False:
            print("\n  ⚠ BEAR MARKET OVERRIDE ACTIVE — score_conviction() returns 0 for all signals")
    else:
        print("  No breadth data in DuckDB yet.")

    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
