"""
scripts/analyse_wyckoff_filter.py — Base quality filter signal analysis.

Loads a backtest run from DuckDB, re-scores each signal with Wyckoff's
base_quality_score() using the cached parquet bars, then prints a side-by-side
forward-return comparison: confirmed (BQ >= threshold) vs plain (BQ < threshold).

Usage:
    python scripts/analyse_wyckoff_filter.py --symbol AAPL
    python scripts/analyse_wyckoff_filter.py --symbol AAPL --threshold 0.5
    python scripts/analyse_wyckoff_filter.py --symbol AAPL --run-id AAPL_breakout_20260604_124636
"""

from __future__ import annotations

from argparse import ArgumentParser
from datetime import date

import duckdb
import pyarrow.parquet as pq

from quantlab.providers.base import Bar
from quantlab.signals.wyckoff import base_quality_score
from quantlab.storage import DB_PATH, DATA_PROCESSED
from quantlab.utils import setup_logging


# ── Helpers ────────────────────────────────────────────────────────────────────

def _avg(vals: list) -> float | None:
    v = [x for x in vals if x is not None]
    return sum(v) / len(v) if v else None

def _hit(vals: list) -> float | None:
    v = [x for x in vals if x is not None]
    return sum(1 for x in v if x > 0) / len(v) if v else None

def _pct(v: float | None, d: int = 2) -> str:
    return f"{v * 100:{'+' if d else ''}.{d}f}%" if v is not None else "  N/A"

def _diff_arrow(a: float | None, b: float | None) -> str:
    if a is None or b is None:
        return " "
    return "▲" if a > b else "▼"


def _tbl_row(label: str, c_vals: list, p_vals: list) -> None:
    ca = _avg(c_vals); pa = _avg(p_vals)
    diff = (ca - pa) if (ca is not None and pa is not None) else None
    print(
        f"  {label:<32}  {_pct(ca):>8}  {_pct(pa):>8}  "
        f"{_pct(diff):>8}  {_diff_arrow(ca, pa)}"
    )

def _hit_row(label: str, c_vals: list, p_vals: list) -> None:
    ca = _hit(c_vals); pa = _hit(p_vals)
    diff = (ca - pa) if (ca is not None and pa is not None) else None
    fmt = lambda x: f"{x * 100:.1f}%" if x is not None else "  N/A"
    diff_str = f"{diff * 100:+.1f}%" if diff is not None else "  N/A"
    print(
        f"  {label:<32}  {fmt(ca):>8}  {fmt(pa):>8}  "
        f"{diff_str:>8}  {_diff_arrow(ca, pa)}"
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    setup_logging(level="WARNING")

    parser = ArgumentParser(description="Wyckoff base quality filter analysis.")
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument(
        "--run-id", default=None,
        help="Specific run_id (default: most recent for symbol in trades table)",
    )
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--min-weeks", type=int, default=12)
    args = parser.parse_args()

    # ── Get run_id ─────────────────────────────────────────────────────────────
    con = duckdb.connect(str(DB_PATH))

    if args.run_id:
        run_id = args.run_id
    else:
        row = con.execute("""
            SELECT run_id FROM trades
            WHERE symbol = ?
            GROUP BY run_id
            ORDER BY MAX(entry_date) DESC, COUNT(*) DESC
            LIMIT 1
        """, [args.symbol]).fetchone()
        if row is None:
            raise SystemExit(f"No trades found for {args.symbol} in {DB_PATH}")
        run_id = row[0]

    # ── Load trades ────────────────────────────────────────────────────────────
    trades = con.execute("""
        SELECT entry_date, trade_return, ret_1d, ret_3d, ret_5d, mfe_5d, mae_5d
        FROM trades
        WHERE symbol = ? AND run_id = ?
        ORDER BY entry_date
    """, [args.symbol, run_id]).fetchall()
    con.close()

    if not trades:
        raise SystemExit(f"No trades found for {args.symbol} run_id={run_id}")

    # ── Load bars from parquet cache ───────────────────────────────────────────
    pq_path = DATA_PROCESSED / f"{args.symbol}_bars.parquet"
    if not pq_path.exists():
        raise SystemExit(
            f"Parquet cache not found: {pq_path}\n"
            f"Run a backtest with --save-parquet or use the IBKR provider to cache bars."
        )

    tbl = pq.read_table(pq_path).to_pydict()
    bars = sorted(
        [
            Bar(
                as_of=date.fromisoformat(tbl["date"][i]),
                open=tbl["open"][i], high=tbl["high"][i],
                low=tbl["low"][i],   close=tbl["close"][i],
                volume=tbl["volume"][i],
            )
            for i in range(len(tbl["date"]))
        ],
        key=lambda b: b.as_of,
    )
    date_to_idx = {b.as_of.isoformat(): i for i, b in enumerate(bars)}

    # ── Score each trade ───────────────────────────────────────────────────────
    confirmed: list[tuple] = []
    plain: list[tuple] = []
    bq_all: list[float] = []

    for entry_date, trade_return, r1, r3, r5, mfe, mae in trades:
        entry_str = entry_date.isoformat() if hasattr(entry_date, "isoformat") else str(entry_date)
        idx = date_to_idx.get(entry_str)
        if idx is None:
            continue
        bq = base_quality_score(bars[: idx + 1], min_weeks=args.min_weeks)
        bq_all.append(bq)
        row = (trade_return, r1, r3, r5, mfe, mae, bq)
        (confirmed if bq >= args.threshold else plain).append(row)

    n_c, n_p = len(confirmed), len(plain)
    total = n_c + n_p

    # ── Print results ──────────────────────────────────────────────────────────
    hdr_c = f"BQ≥{args.threshold} (n={n_c})"
    hdr_p = f"BQ<{args.threshold} (n={n_p})"

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  {args.symbol} — Base Quality Filter  (threshold={args.threshold})")
    print(f"  Run: {run_id}  |  Total signals: {total}")
    print(sep)
    print(f"  {'Metric':<32}  {hdr_c:>8}  {hdr_p:>8}  {'diff':>8}  dir")
    print(f"  {'─' * 68}")

    c_r1  = [t[1] for t in confirmed]; p_r1  = [t[1] for t in plain]
    c_r3  = [t[2] for t in confirmed]; p_r3  = [t[2] for t in plain]
    c_r5  = [t[3] for t in confirmed]; p_r5  = [t[3] for t in plain]
    c_mfe = [t[4] for t in confirmed]; p_mfe = [t[4] for t in plain]
    c_mae = [t[5] for t in confirmed]; p_mae = [t[5] for t in plain]
    c_tr  = [t[0] for t in confirmed if t[0] is not None]
    p_tr  = [t[0] for t in plain     if t[0] is not None]

    _tbl_row("1D avg forward return",   c_r1,  p_r1)
    _hit_row("1D hit rate",             c_r1,  p_r1)
    _tbl_row("3D avg forward return",   c_r3,  p_r3)
    _hit_row("3D hit rate",             c_r3,  p_r3)
    _tbl_row("5D avg forward return",   c_r5,  p_r5)
    _hit_row("5D hit rate",             c_r5,  p_r5)
    _tbl_row("MFE 5D avg",              c_mfe, p_mfe)
    _tbl_row("MAE 5D avg",              c_mae, p_mae)

    cwr = (_hit(c_tr) or 0.0); pwr = (_hit(p_tr) or 0.0)
    diff_wr = cwr - pwr
    arrow = "▲" if diff_wr > 0 else "▼"
    print(
        f"  {'Win rate (completed trades)':<32}  "
        f"{cwr*100:>7.1f}%  {pwr*100:>7.1f}%  {diff_wr*100:>+7.1f}%  {arrow}"
    )
    print(f"  {'Completed trades (n)':<32}  {len(c_tr):>8}  {len(p_tr):>8}")
    print(sep)

    if bq_all:
        bq_s = sorted(bq_all)
        n = len(bq_s)
        print(f"\n  BQ score distribution ({n} signals):")
        print(f"  min={min(bq_all):.3f}  p25={bq_s[n//4]:.3f}  "
              f"median={bq_s[n//2]:.3f}  p75={bq_s[3*n//4]:.3f}  max={max(bq_all):.3f}  "
              f"mean={sum(bq_all)/n:.3f}")
    print()


if __name__ == "__main__":
    main()
