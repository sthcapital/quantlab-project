"""
scripts/sync_flat_files.py — Download yesterday's (or a specific day's)
stocks and options flat files from the Massive S3 bucket.

Prints compressed file sizes, record counts, and local Parquet paths.
Files already cached are reported instantly without re-downloading.

Usage:
    python scripts/sync_flat_files.py
    python scripts/sync_flat_files.py --date 2025-05-01
    python scripts/sync_flat_files.py --date 2025-05-01 --no-options
    python scripts/sync_flat_files.py --start 2025-05-01 --end 2025-05-05
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _prev_trading_day() -> date:
    """Return the most recent weekday before today."""
    d = date.today() - timedelta(days=1)
    while d.weekday() >= 5:   # 5=Sat, 6=Sun
        d -= timedelta(days=1)
    return d


def _fmt_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes //= 1024
    return f"{n_bytes:.1f} GB"


def sync_one_day(flat, d: date, skip_options: bool) -> bool:
    """Download and cache flat files for a single trading day. Returns True on success."""
    print(f"\n{'─'*56}")
    print(f"  Date: {d.isoformat()}")
    print(f"{'─'*56}")

    # ── Stocks ────────────────────────────────────────────────────────────────
    cache_path = flat.stocks_cache_path(d)
    try:
        stocks = flat.download_stocks_day(d)
        size = _fmt_size(cache_path.stat().st_size)
        tag = "(cached)" if "cache hit" in "" else ""
        print(f"  Stocks  : {len(stocks):>7,} symbols | parquet {size} → {cache_path.name}")
    except Exception as exc:
        print(f"  Stocks  : FAILED — {exc}")
        return False

    # ── Options ───────────────────────────────────────────────────────────────
    if not skip_options:
        opt_cache = flat.options_cache_path(d)
        try:
            options = flat.download_options_day(d)
            size = _fmt_size(opt_cache.stat().st_size)
            print(f"  Options : {len(options):>7,} records | parquet {size} → {opt_cache.name}")
        except Exception as exc:
            print(f"  Options : FAILED — {exc}")

    return True


def main() -> None:
    parser = ArgumentParser(description="Sync Massive S3 flat files to local Parquet cache.")
    parser.add_argument(
        "--date",
        default=None,
        help="Single trading date YYYY-MM-DD (default: yesterday)",
    )
    parser.add_argument("--start", default=None, help="Start of date range YYYY-MM-DD")
    parser.add_argument("--end",   default=None, help="End of date range YYYY-MM-DD")
    parser.add_argument(
        "--no-options", action="store_true",
        help="Skip the options flat file (much faster; options file is ~3 MB/day)",
    )
    args = parser.parse_args()

    from quantlab.providers.flat_files import FlatFileProvider
    flat = FlatFileProvider()

    # Determine date range
    if args.start and args.end:
        start = date.fromisoformat(args.start)
        end   = date.fromisoformat(args.end)
    elif args.date:
        start = end = date.fromisoformat(args.date)
    else:
        start = end = _prev_trading_day()

    print(f"\n{'═'*56}")
    print(f"  QuantLab S3 Flat File Sync")
    print(f"  Endpoint : {flat.endpoint}")
    print(f"  Bucket   : {flat.bucket}")
    print(f"  Range    : {start} → {end}")
    print(f"  Options  : {'disabled' if args.no_options else 'enabled'}")
    print(f"{'═'*56}")

    total_stocks = 0
    total_options = 0
    days_ok = 0
    days_skipped = 0

    current = start
    while current <= end:
        if current.weekday() >= 5:       # skip weekends in explicit ranges
            current += timedelta(days=1)
            continue

        cache_path = flat.stocks_cache_path(current)
        try:
            stocks = flat.download_stocks_day(current)
            total_stocks += len(stocks)
            sz = _fmt_size(cache_path.stat().st_size)
            print(f"  {current}  stocks {len(stocks):>7,}  parquet {sz}")

            if not args.no_options:
                opt_cache = flat.options_cache_path(current)
                try:
                    options = flat.download_options_day(current)
                    total_options += len(options)
                    sz2 = _fmt_size(opt_cache.stat().st_size)
                    print(f"  {current}  options {len(options):>6,}  parquet {sz2}")
                except Exception as exc:
                    print(f"  {current}  options FAILED — {exc}")

            days_ok += 1

        except Exception as exc:
            print(f"  {current}  SKIPPED — {exc}")
            days_skipped += 1

        current += timedelta(days=1)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*56}")
    print(f"  Days synced  : {days_ok}")
    if days_skipped:
        print(f"  Days skipped : {days_skipped} (non-trading or S3 error)")
    if total_stocks:
        print(f"  Total stocks : {total_stocks:,} symbol-days")
    if total_options and not args.no_options:
        print(f"  Total options: {total_options:,} contract-days")
    cache_dir = flat._cache_dir()
    print(f"  Cache dir    : {cache_dir}")
    print(f"{'═'*56}\n")


if __name__ == "__main__":
    main()
