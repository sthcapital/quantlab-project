"""
scripts/monitor_options.py — Intraday options activity monitor.

Runs every 30 minutes during market hours (9:30 AM – 4:00 PM ET, Mon–Fri).
Scans only the institutional_watchlist symbols — not the full 2,325-symbol
universe — keeping each run under 2 minutes.

On unusual options detection:
  - Sets options_signal=True on the institutional watchlist entry
  - Adds +0.08 conviction bonus (capped at 1.0)
  - Logs the update with timestamp for the daily report

Requires POLYGON_API_KEY in the environment (loaded from .env by daily_scan.sh
or set directly in the shell).

Usage:
    python scripts/monitor_options.py
    python scripts/monitor_options.py --force     # bypass market-hours check
    python scripts/monitor_options.py --dry-run   # print detections without writing
"""

from __future__ import annotations

import os
import sys
from argparse import ArgumentParser
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _is_market_hours() -> bool:
    """Return True when current ET time is within 9:30 AM – 4:00 PM on a weekday."""
    try:
        import pytz
        ny  = pytz.timezone("America/New_York")
        now = datetime.now(ny)
        if now.weekday() >= 5:
            return False
        open_  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
        close_ = now.replace(hour=16, minute=0,  second=0, microsecond=0)
        return open_ <= now <= close_
    except ImportError:
        return True  # pytz unavailable — run regardless


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main() -> None:
    parser = ArgumentParser(description="Intraday options activity monitor.")
    parser.add_argument("--force",   action="store_true",
                        help="Bypass market-hours guard and run unconditionally")
    parser.add_argument("--dry-run", action="store_true",
                        help="Detect but do not write results to DuckDB")
    args = parser.parse_args()

    if not args.force and not _is_market_hours():
        print(f"[{_ts()}] monitor_options: outside market hours — skipping")
        return

    from quantlab.utils import setup_logging
    setup_logging(level="WARNING")

    from quantlab.watchlist import InstitutionalWatchlist
    iwl        = InstitutionalWatchlist()
    candidates = iwl.get_candidates()

    if not candidates:
        print(f"[{_ts()}] monitor_options: institutional watchlist empty — nothing to monitor")
        return

    polygon_key = os.environ.get("POLYGON_API_KEY", "")
    if not polygon_key:
        print(f"[{_ts()}] monitor_options: POLYGON_API_KEY not set — skipping")
        return

    today = date.today()
    print(f"\n[{_ts()}] monitor_options: checking {len(candidates)} watchlist symbols ...")

    try:
        from quantlab.providers.massive_options import MassiveOptionsProvider
        mp = MassiveOptionsProvider(api_key=polygon_key)
    except Exception as exc:
        print(f"[{_ts()}] monitor_options: MassiveOptionsProvider unavailable: {exc}")
        return

    alerts: list[str] = []

    for entry in candidates:
        sym         = entry["symbol"]
        entry_price = entry.get("entry_price") or 0.0
        if entry_price <= 0:
            continue
        try:
            opt_score = mp.compute_options_score(sym, entry_price)
            if opt_score >= 0.6:
                flag = "▲" if opt_score >= 0.8 else "~"
                msg  = (
                    f"  {flag} {sym:<8}  opt_score={opt_score:.2f}  "
                    f"conv_was={entry.get('conviction_score',0):.2f}  "
                    f"days={entry.get('consecutive_days',1)}"
                )
                print(msg)
                alerts.append(sym)
                if not args.dry_run:
                    iwl.set_options_signal(sym, bonus=0.08)
        except Exception as exc:
            # Options data unavailable for this symbol — skip silently
            pass

    if alerts:
        print(f"[{_ts()}] monitor_options: flagged {len(alerts)} symbol(s): "
              f"{', '.join(alerts)}")
    else:
        print(f"[{_ts()}] monitor_options: no unusual options activity detected")


if __name__ == "__main__":
    main()
