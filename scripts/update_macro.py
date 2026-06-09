"""
scripts/update_macro.py — Daily macro regime snapshot (FRED + CBOE VIX).

Fetches FRED yield spreads, credit spreads, Treasury rates, and crude oil,
then enriches with CBOE VIX.  Classifies macro regime and stores the
complete snapshot to the DuckDB macro_snapshots table.  Idempotent —
re-running on the same date overwrites the existing row.

FRED series fetched:
    T10Y2Y        — 10Y minus 2Y Treasury spread (yield curve)
    T10Y3M        — 10Y minus 3M Treasury spread
    BAMLH0A0HYM2  — High Yield OAS credit spread
    DGS10         — 10Y Treasury constant maturity
    FEDFUNDS      — Effective federal funds rate
    DCOILWTICO    — WTI crude oil price

Regime classification:
    "stress"   — ≥2 of: yield curve inverted, HY spread >5%, VIX >25
    "risk_off" — 1 of above
    "risk_on"  — none

Usage:
    python scripts/update_macro.py                    # today
    python scripts/update_macro.py --date 2026-06-01  # specific date

Requires: FRED_API_KEY env var (free at fred.stlouisfed.org)
          VIX from CBOE CDN — no key required.
"""

from __future__ import annotations

import os
import sys
from argparse import ArgumentParser
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main() -> None:
    parser = ArgumentParser(
        description="Update daily macro regime snapshot (FRED + CBOE VIX)."
    )
    parser.add_argument(
        "--date", default=None,
        help="Reference date YYYY-MM-DD (default: today)",
    )
    args = parser.parse_args()
    as_of = date.fromisoformat(args.date) if args.date else date.today()

    fred_key = os.environ.get("FRED_API_KEY", "")

    print(f"\n{'='*60}")
    print(f"  Macro Update — {as_of}")
    print(f"{'='*60}")

    # ── VIX from CBOE CDN (no API key required) ─────────────────────────────
    vix_close: float | None = None
    try:
        from quantlab.providers.cboe import fetch_vix_history, classify_vix_regime
        vix_bars = fetch_vix_history(as_of - timedelta(days=7), as_of)
        if vix_bars:
            vix_close = vix_bars[-1].close
            vix_label, _ = classify_vix_regime(vix_close)
            print(f"  VIX     : {vix_close:.2f}  ({vix_label})")
        else:
            print("  VIX     : no recent data from CBOE CDN")
    except Exception as exc:
        print(f"  VIX     : unavailable ({exc})")

    # ── FRED macro data ──────────────────────────────────────────────────────
    from quantlab.providers.fred import (
        MacroSnapshot,
        classify_macro_regime,
        _store_snapshot,
    )

    if not fred_key:
        print("  FRED    : FRED_API_KEY not set — storing VIX-only snapshot")
        snap = MacroSnapshot(as_of=as_of, vix_close=vix_close)
        snap.macro_regime = classify_macro_regime(snap)
        _store_snapshot(snap)
        print(f"  Regime  : {snap.macro_regime}")
        print(f"{'='*60}\n")
        return

    try:
        from quantlab.providers.fred import fetch_macro_snapshot
        snap = fetch_macro_snapshot(api_key=fred_key, as_of_date=as_of)

        # fetch_macro_snapshot auto-stores to DB (without VIX).
        # Enrich snap with VIX, reclassify, and re-store so VIX is persisted.
        if vix_close is not None:
            snap.vix_close = vix_close
            snap.macro_regime = classify_macro_regime(snap)
            _store_snapshot(snap)

        def _fmt_pct(v: float | None, label: str) -> str:
            return f"  {label:<9}: {v:+.2f}%" if v is not None else f"  {label:<9}: N/A"

        print(_fmt_pct(snap.yield_spread_10y2y, "10Y2Y"))
        print(_fmt_pct(snap.hy_credit_spread,   "HY OAS"))
        print(_fmt_pct(snap.fed_funds_rate,      "Fed Fds"))
        if snap.wti_crude is not None:
            print(f"  WTI      : ${snap.wti_crude:.2f}")
        print(f"  Regime   : {snap.macro_regime}")

    except Exception as exc:
        print(f"  FRED    : fetch failed — {exc}")
        # Still persist at least VIX
        snap = MacroSnapshot(as_of=as_of, vix_close=vix_close)
        snap.macro_regime = classify_macro_regime(snap)
        _store_snapshot(snap)
        print(f"  Regime  : {snap.macro_regime}  (VIX-only)")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
