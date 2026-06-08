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
    flat_file_provider=None,
    use_prev: bool = True,
) -> BreadthSnapshot | None:
    """
    Fetch today's grouped daily from Polygon, compute breadth, store to DuckDB.

    When flat_file_provider is supplied, loads up to 215 trading days of
    per-symbol bar history from S3 flat files (cached as local Parquet) using
    a 400-calendar-day lookback window.  The extra 15-day buffer over the
    200-bar minimum absorbs flat-file S3 publish lag (last 2-3 days) and
    any other calendar misses, ensuring symbols accumulate 200+ bars for
    the 200-SMA computation.  Passes history_data to compute_market_breadth()
    so that pct_above_20/50/200sma are populated.

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

    # Build history_data for SMA participation metrics via S3 flat files.
    # get_grouped_daily() reads local Parquet cache when available — only older
    # dates hit S3. Log progress every 20 days so the operator can monitor.
    # 400 calendar days → ~276 trading days before the [-200:] cap; the extra
    # buffer absorbs weekends, ~10 US holidays/year, and occasional S3 misses.
    history_data: dict = {}
    if flat_file_provider is not None:
        hist_start = trade_date - timedelta(days=400)  # ~276 trading days of buffer
        hist_end   = trade_date - timedelta(days=1)    # exclude today

        trading_days = _trading_days_between(hist_start, hist_end)
        # Cap at 215 trading days: 200 needed for 200-SMA + ~15-day buffer for
        # flat-file S3 publish lag (last 2-3 days) and any other misses.
        days_to_load = trading_days[-215:]
        total_days   = len(days_to_load)

        print(f"  Loading {total_days} days of history for SMA participation ...")
        symbol_bars: dict[str, list] = {}
        loaded = 0
        for i, d in enumerate(days_to_load):
            if i > 0 and i % 20 == 0:
                print(f"    [{i}/{total_days}] history days loaded ...")
            try:
                day_data = flat_file_provider.get_grouped_daily(d)
                for sym, bar in day_data.items():
                    if sym not in symbol_bars:
                        symbol_bars[sym] = []
                    symbol_bars[sym].append(bar)
                loaded += 1
            except Exception:
                continue  # non-trading day or S3 miss — skip silently
        history_data = symbol_bars
        n_200plus = sum(1 for bars in history_data.values() if len(bars) >= 200)
        print(f"  History loaded: {loaded}/{total_days} days  "
              f"{len(history_data):,} symbols with bar history")
        print(f"  History: {n_200plus:,} symbols with 200+ bars "
              f"(of {len(history_data):,} total)")

    snapshot = compute_market_breadth(
        trade_date   = trade_date,
        today_data   = today_data,
        prev_data    = prev_data,
        history_data = history_data if history_data else None,
    )

    # Compute SPY 200 SMA and slope from history_data
    if history_data and "SPY" in history_data and len(history_data["SPY"]) >= 200:
        spy_bars   = history_data["SPY"]
        spy_closes = [b.close for b in spy_bars[-200:]]
        spy_sma200 = sum(spy_closes) / 200
        spy_today  = today_data.get("SPY")
        if spy_today:
            snapshot.spy_above_200sma = spy_today.close > spy_sma200
            if len(spy_bars) >= 220:
                spy_sma200_20d_ago = sum(b.close for b in spy_bars[-220:-20]) / 200
                snapshot.spy_200sma_slope = round(spy_sma200 - spy_sma200_20d_ago, 4)

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


def _trading_days_between(start: date, end: date) -> list[date]:
    """Return list of Mon-Fri market-open days from start to end inclusive."""
    days: list[date] = []
    current = start
    while current <= end:
        if is_market_open(current):
            days.append(current)
        current += timedelta(days=1)
    return days


def main() -> None:
    setup_logging(level="WARNING")

    parser = ArgumentParser(description="Update market breadth data from Polygon.io.")
    parser.add_argument("--date", default=None, help="Trading date YYYY-MM-DD (default: today)")
    parser.add_argument("--no-polygon", action="store_true",
                        help="Skip Polygon fetch; only recompute rolling from DuckDB")
    parser.add_argument("--days-rolling", type=int, default=60,
                        help="Days of history to use for rolling metrics (default 60)")
    # ── Backfill ──────────────────────────────────────────────────────────────
    parser.add_argument("--backfill", action="store_true",
                        help="Fetch a range of historical dates one at a time with rate-limit delays")
    parser.add_argument("--start-date", default=None,
                        help="Backfill start date YYYY-MM-DD (required with --backfill)")
    parser.add_argument("--end-date", default=None,
                        help="Backfill end date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    trade_date = date.fromisoformat(args.date) if args.date else date.today()

    # ── Backfill mode: iterate trading days sequentially ──────────────────────
    if args.backfill:
        if not args.start_date:
            raise SystemExit("--backfill requires --start-date YYYY-MM-DD")
        start_bf = date.fromisoformat(args.start_date)
        end_bf   = date.fromisoformat(args.end_date) if args.end_date else date.today()
        days     = _trading_days_between(start_bf, end_bf)
        api_key  = __import__("os").environ.get("POLYGON_API_KEY", "")
        if not api_key:
            raise SystemExit("POLYGON_API_KEY not set — cannot backfill")
        from quantlab.providers.polygon import PolygonProvider
        # grouped_daily_sleep=12s respects free-tier 5 req/min limit between dates
        provider = PolygonProvider(api_key=api_key, grouped_daily_sleep=12.0)
        print(f"\n{'='*62}")
        print(f"  Backfill: {start_bf} → {end_bf}  ({len(days)} trading days)")
        print(f"  Rate: 1 request per ~12s (Polygon free tier).  ETA: ~{len(days)*13//60}min")
        print(f"{'='*62}")
        stored = 0
        for i, d in enumerate(days, 1):
            snap = fetch_and_store(d, provider)
            if snap:
                save_breadth_snapshot(snap)
                stored += 1
                print(f"  [{i:>4}/{len(days)}] {d}  A={snap.advances} D={snap.declines}"
                      f"  up4%={snap.up_4pct_count} dn4%={snap.down_4pct_count}")
            else:
                print(f"  [{i:>4}/{len(days)}] {d}  skipped (no data)")
        # Recompute rolling after all dates are stored
        print(f"\n  Stored {stored} snapshots.  Recomputing rolling metrics ...")
        recompute_rolling(max(args.days_rolling, stored + 10))
        latest = get_latest_snapshot()
        if latest:
            print(f"\n  Latest: {latest.summary_line()}")
        print(f"{'='*62}\n")
        return

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
            from quantlab.providers.flat_files import FlatFileProvider
            provider     = PolygonProvider(api_key=api_key)
            flat_provider = FlatFileProvider()   # reads S3 creds from env
            snap = fetch_and_store(trade_date, provider,
                                   flat_file_provider=flat_provider)
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
