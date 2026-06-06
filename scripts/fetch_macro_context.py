"""
scripts/fetch_macro_context.py — Daily macro dashboard.

Fetches CBOE VIX history, FRED macro series, and (optionally) SEC EDGAR
fundamentals for a ticker, then prints a human-readable dashboard.

Usage:
    python scripts/fetch_macro_context.py --start 2024-01-01 --end 2026-06-06
    python scripts/fetch_macro_context.py --start 2024-01-01 --end 2026-06-06 --csv
    python scripts/fetch_macro_context.py --ticker AAPL
"""

from __future__ import annotations

import csv
import os
import sys
from argparse import ArgumentParser
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _fmt(val: float | None, fmt: str = ".2f", suffix: str = "") -> str:
    if val is None:
        return "--"
    return f"{val:{fmt}}{suffix}"


def _regime_flag(val: float | None, threshold: float, invert: bool = False) -> str:
    if val is None:
        return ""
    triggered = val < threshold if not invert else val > threshold
    return "  ⚠" if triggered else "  ✓"


def print_dashboard(
    vix_bars,
    snap,
    fundamental_snap=None,
) -> None:
    today = date.today().isoformat()
    print(f"\n{'=' * 54}")
    print(f"  QUANTLAB MACRO DASHBOARD  {today}")
    print(f"{'=' * 54}")

    # ── VIX ───────────────────────────────────────────────
    print("\nVIX  (CBOE — cboe.com/tradable_products/vix)")
    if vix_bars:
        latest = vix_bars[-1]
        closes = [b.close for b in vix_bars]
        from quantlab.providers.cboe import classify_vix_regime
        label, score = classify_vix_regime(latest.close)
        print(f"  Latest close : {latest.close:.2f}  → {label.upper()} (score {score})")
        print(f"  Period high  : {max(closes):.2f}")
        print(f"  Period low   : {min(closes):.2f}")
        print(f"  Bars in range: {len(vix_bars)}")
    else:
        print("  No VIX data available for the requested period.")

    # ── FRED ──────────────────────────────────────────────
    print("\nMACRO  (FRED — fred.stlouisfed.org)")
    if snap is not None:
        print(
            f"  10Y-2Y spread  : {_fmt(snap.yield_spread_10y2y, '+.2f', '%')}"
            f"{_regime_flag(snap.yield_spread_10y2y, 0.0)}"
        )
        print(
            f"  10Y-3M spread  : {_fmt(snap.yield_spread_10y3m, '+.2f', '%')}"
            f"{_regime_flag(snap.yield_spread_10y3m, 0.0)}"
        )
        print(
            f"  HY credit OAS  : {_fmt(snap.hy_credit_spread, '.2f', '%')}"
            f"{_regime_flag(snap.hy_credit_spread, 5.0, invert=True)}"
        )
        print(f"  10Y Treasury   : {_fmt(snap.treasury_10y, '.2f', '%')}")
        print(f"  Fed Funds      : {_fmt(snap.fed_funds_rate, '.2f', '%')}")
        print(f"  WTI Crude      : ${_fmt(snap.wti_crude, '.2f')}")
    else:
        print("  FRED_API_KEY not set — skipped.")
        print("  Set FRED_API_KEY env var (free at fred.stlouisfed.org/docs/api).")

    # ── Regime classification ─────────────────────────────
    print("\nREGIME CLASSIFICATION")
    if snap is not None:
        from quantlab.providers.fred import classify_macro_regime
        if vix_bars:
            snap.vix_close = vix_bars[-1].close
            snap.macro_regime = classify_macro_regime(snap)
        warnings = sum([
            snap.yield_spread_10y2y is not None and snap.yield_spread_10y2y < 0,
            snap.hy_credit_spread is not None and snap.hy_credit_spread > 5.0,
            snap.vix_close is not None and snap.vix_close > 25.0,
        ])
        print(f"  Macro warnings : {warnings} of 3")
        print(f"  Macro regime   : {snap.macro_regime.upper()}")
    if vix_bars:
        from quantlab.providers.cboe import classify_vix_regime
        label, score = classify_vix_regime(vix_bars[-1].close)
        print(f"  VIX regime     : {label.upper()} (score {score})")

    # ── Fundamentals (optional) ───────────────────────────
    if fundamental_snap is not None:
        from quantlab.providers.edgar import compute_earnings_acceleration
        accel = compute_earnings_acceleration(fundamental_snap)
        fs = fundamental_snap

        def _mm(v: float | None) -> str:
            if v is None:
                return "--"
            if abs(v) >= 1e9:
                return f"${v/1e9:.1f}B"
            if abs(v) >= 1e6:
                return f"${v/1e6:.1f}M"
            return f"{v:.4f}"

        def _pct(v: float | None) -> str:
            return f"{v*100:+.1f}%" if v is not None else "--"

        print(f"\nFUNDAMENTALS  (SEC EDGAR — {fs.ticker}  CIK {fs.cik})")
        print(f"  Revenue        : {_mm(fs.revenue)}  ({_pct(fs.revenue_qoq_growth)} QoQ)")
        print(f"  Net income     : {_mm(fs.net_income)}  ({_pct(fs.net_income_qoq_growth)} QoQ)")
        print(f"  EPS diluted    : {_fmt(fs.eps_diluted, '.4f')}  ({_pct(fs.eps_qoq_growth)} QoQ)")
        print(f"  Total assets   : {_mm(fs.total_assets)}")
        print(f"  Total debt     : {_mm(fs.total_debt)}")
        print(f"  Op. cash flow  : {_mm(fs.operating_cashflow)}")
        print(f"  CapEx          : {_mm(fs.capex)}")
        print(f"  Shares out     : {_mm(fs.shares_out)}")
        print(f"  Earnings accel : {accel:.2f}  ({'accelerating' if accel > 0.55 else 'decelerating' if accel < 0.45 else 'neutral'})")

    print()


def write_csv(vix_bars, snap, output_dir: Path, as_of: date) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = as_of.isoformat()

    # VIX CSV
    if vix_bars:
        vix_path = output_dir / f"vix_history_{stamp}.csv"
        with vix_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["date", "open", "high", "low", "close"])
            for b in vix_bars:
                w.writerow([b.date.isoformat(), b.open, b.high, b.low, b.close])
        print(f"VIX CSV → {vix_path}")

    # Macro snapshot CSV
    if snap is not None:
        macro_path = output_dir / f"macro_snapshot_{stamp}.csv"
        with macro_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            w.writerow(["as_of", snap.as_of.isoformat()])
            w.writerow(["yield_spread_10y2y", snap.yield_spread_10y2y])
            w.writerow(["yield_spread_10y3m", snap.yield_spread_10y3m])
            w.writerow(["hy_credit_spread", snap.hy_credit_spread])
            w.writerow(["treasury_10y", snap.treasury_10y])
            w.writerow(["fed_funds_rate", snap.fed_funds_rate])
            w.writerow(["wti_crude", snap.wti_crude])
            w.writerow(["vix_close", snap.vix_close])
            w.writerow(["macro_regime", snap.macro_regime])
        print(f"Macro CSV → {macro_path}")


def main() -> None:
    parser = ArgumentParser(description="Fetch and display a daily macro dashboard.")
    parser.add_argument(
        "--start", default="2024-01-01",
        help="VIX history start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end", default=date.today().isoformat(),
        help="VIX history end date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--ticker", default=None,
        help="Optional ticker for SEC EDGAR fundamentals (e.g. AAPL)",
    )
    parser.add_argument(
        "--csv", action="store_true",
        help="Write CSV files to output/ directory",
    )
    parser.add_argument(
        "--fred-key", default=os.getenv("FRED_API_KEY", ""),
        help="FRED API key (overrides FRED_API_KEY env var)",
    )
    args = parser.parse_args()

    from datetime import datetime
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    # ── Fetch VIX ─────────────────────────────────────────
    print("Fetching VIX history from CBOE...", end=" ", flush=True)
    try:
        from quantlab.providers.cboe import fetch_vix_history
        vix_bars = fetch_vix_history(start, end)
        print(f"{len(vix_bars)} bars")
    except Exception as exc:
        print(f"FAILED ({exc})")
        vix_bars = []

    # ── Fetch FRED ────────────────────────────────────────
    snap = None
    fred_key = args.fred_key
    if fred_key:
        print("Fetching macro snapshot from FRED...", end=" ", flush=True)
        try:
            from quantlab.providers.fred import fetch_macro_snapshot
            snap = fetch_macro_snapshot(fred_key, end)
            print("OK")
        except Exception as exc:
            print(f"FAILED ({exc})")
    else:
        print("FRED_API_KEY not set — skipping macro snapshot.")

    # ── Fetch EDGAR (optional) ────────────────────────────
    fundamental_snap = None
    if args.ticker:
        print(f"Fetching fundamentals from SEC EDGAR for {args.ticker}...", end=" ", flush=True)
        try:
            from quantlab.providers.edgar import fetch_fundamentals
            fundamental_snap = fetch_fundamentals(args.ticker)
            print("OK")
        except Exception as exc:
            print(f"FAILED ({exc})")

    # ── Print dashboard ───────────────────────────────────
    print_dashboard(vix_bars, snap, fundamental_snap)

    # ── Write CSV ─────────────────────────────────────────
    if args.csv:
        from pathlib import Path
        output_dir = Path(__file__).parent.parent / "output"
        write_csv(vix_bars, snap, output_dir, end)


if __name__ == "__main__":
    main()
