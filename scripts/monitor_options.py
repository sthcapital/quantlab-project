"""
scripts/monitor_options.py — Intraday options activity monitor.

Runs every 30 minutes during market hours (9:30 AM – 4:00 PM ET, Mon–Fri).
Scans only the institutional_watchlist symbols — not the full 2,325-symbol
universe — keeping each run under 2 minutes.

Recalibrated two-pass detection (2026-06; replaces the absolute-threshold
scorer that flagged 347/357 symbols on 2026-06-11):

  Pass 1 — per-symbol: today's total call volume z-scored against the
  symbol's OWN trailing 20-session flat-file baseline, blended with
  continuous PCR / IV-skew tilts (signals/options_relative.py).

  Pass 2 — cross-sectional: "unusual" = the day's scores strictly above the
  configured percentile (scanner.options_unusual_percentile, default p90),
  capping the daily flag rate at ~10% by construction.

On a flagged symbol:
  - Sets options_signal=True on the institutional watchlist entry
  - Adds the +0.08 conviction bonus ONLY when
    scanner.options_signal_gating_enabled is True (default False:
    display-only until the recalibration output is reviewed)
  - Persists unusual_flag to options_snapshots for the report's
    signal-rate header line

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


# Streak length at which the baseline-inflation diagnostic logs — a campaign
# this long has materially fed its own trailing baseline.
_STREAK_DIAG_THRESHOLD = 5


def _log_baseline_inflation(today: date) -> None:
    """
    DIAGNOSTIC ONLY — never changes scoring.

    A multi-day flag campaign inflates the symbol's own 20-session baseline,
    so persistent accumulation gradually un-flags itself.  For every symbol
    with flag_streak ≥ threshold, log today's z against the live baseline
    AND against the baseline frozen at episode start (first_flagged_date) —
    making the decay measurable so a future decision (e.g. episode-frozen
    baselines) is made on data.
    """
    try:
        import duckdb
        from quantlab.storage import DB_PATH
        from quantlab.providers.flat_files import FlatFileProvider
        from quantlab.signals.options_relative import frozen_vs_live_zscores

        con = duckdb.connect(str(DB_PATH), read_only=True)
        rows = con.execute(
            """
            SELECT symbol, flag_streak, first_flagged_date, call_volume
            FROM options_snapshots
            WHERE snap_date = ? AND flag_streak >= ?
            ORDER BY flag_streak DESC
            """,
            [today, _STREAK_DIAG_THRESHOLD],
        ).fetchall()
        con.close()
        if not rows:
            return

        flat = FlatFileProvider()
        live_hist = flat.get_call_volume_history(today, n_sessions=20)
        for sym, streak, episode_start, call_vol in rows:
            if episode_start is None or call_vol is None:
                continue
            if hasattr(episode_start, "date") and not isinstance(episode_start, date):
                episode_start = episode_start.date()
            frozen_hist = flat.get_call_volume_history(episode_start, n_sessions=20)
            z_live, z_frozen = frozen_vs_live_zscores(
                call_vol,
                [d.get(sym, 0.0) for d in live_hist],
                [d.get(sym, 0.0) for d in frozen_hist],
            )
            _zl = f"{z_live:.2f}" if z_live is not None else "—"
            _zf = f"{z_frozen:.2f}" if z_frozen is not None else "—"
            print(f"  [diag] baseline inflation: {sym:<8} streak={streak}  "
                  f"z_live={_zl}  z_frozen={_zf}  (episode {episode_start})")
    except Exception as exc:
        print(f"[{_ts()}] monitor_options: baseline-inflation diag failed: {exc}")


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

    from quantlab.utils import get_config
    from quantlab.providers.flat_files import FlatFileProvider
    from quantlab.signals.options_relative import cross_sectional_flags

    scanner_cfg = get_config("scanner")
    pctl     = float(scanner_cfg.get("options_unusual_percentile", 90.0))
    gating   = bool(scanner_cfg.get("options_signal_gating_enabled", False))
    min_base = float(scanner_cfg.get("options_min_baseline_contracts", 75))
    max_pcr  = float(scanner_cfg.get("options_gate_max_pcr", 1.5))

    # Per-symbol baselines: trailing 20 cached flat-file sessions (one parquet
    # read per session, all underlyings at once; never hits S3).
    history = FlatFileProvider().get_call_volume_history(today, n_sessions=20)
    if not history:
        print(f"[{_ts()}] monitor_options: WARNING — no cached options flat files "
              f"before {today}; per-symbol baselines unavailable, nothing can flag")

    # Pass 1 — per-symbol relative scores
    scores:     dict[str, float | None] = {}
    zscores:    dict[str, float | None] = {}
    base_means: dict[str, float | None] = {}
    pcrs:       dict[str, float | None] = {}
    for entry in candidates:
        sym         = entry["symbol"]
        entry_price = entry.get("entry_price") or 0.0
        if entry_price <= 0:
            continue
        baseline = [day.get(sym, 0.0) for day in history]
        try:
            res = mp.compute_relative_options_score(sym, entry_price, baseline)
            if res is not None:
                scores[sym]     = res["rel_score"]
                zscores[sym]    = res["vol_zscore"]
                base_means[sym] = (sum(baseline) / len(baseline)) if baseline else None
                pcrs[sym]       = res.get("pcr")
        except Exception:
            # Options data unavailable for this symbol — skip silently
            pass

    # Pass 2 — cross-sectional gate: unusual = top-percentile of the day's
    # scores AND ≥2σ above the symbol's own baseline (cap, not quota) AND a
    # baseline liquid enough that the spike can mean accumulation AND not
    # put-dominated flow (PCR ceiling — direction-aware LONG confirmation)
    flagged = cross_sectional_flags(
        scores, percentile_cut=pctl, zscores=zscores,
        baseline_means=base_means, min_baseline=min_base,
        pcrs=pcrs, max_pcr=max_pcr,
    )
    # Cleared volume/liquidity but blocked by the PCR ceiling — tagged as
    # future short-side signal data (SHORT_SIGNAL_ENABLED is off)
    put_dominated = cross_sectional_flags(
        scores, percentile_cut=pctl, zscores=zscores,
        baseline_means=base_means, min_baseline=min_base,
    ) - flagged

    if not args.dry_run:
        mp.mark_unusual_flags(flagged, put_dominated=put_dominated)
        _log_baseline_inflation(today)

    for sym in sorted(flagged, key=lambda s: -(scores[s] or 0.0)):
        print(f"  ▲ {sym:<8}  rel_score={scores[sym]:.4f}")
        if not args.dry_run:
            iwl.set_options_signal(sym, bonus=0.08 if gating else 0.0)

    n_scored = sum(1 for v in scores.values() if v is not None)
    rate = (len(flagged) / n_scored) if n_scored else 0.0
    print(f"[{_ts()}] monitor_options: Options: {len(flagged)}/{n_scored} unusual, "
          f"{rate:.1%}  (gate p{pctl:g}, gating "
          f"{'ENABLED' if gating else 'display-only'})")


if __name__ == "__main__":
    main()
