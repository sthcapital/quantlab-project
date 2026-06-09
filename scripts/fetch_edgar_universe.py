"""
scripts/fetch_edgar_universe.py — Populate EDGAR fundamentals for all tradeable symbols.

Fetches EPS, revenue, and net-income data from the SEC EDGAR companyfacts API
for every symbol in the tradeable universe, computing acceleration scores and
storing results in the edgar_fundamentals DuckDB table.

SEC fair-use policy: ≤ 10 requests/second.
With typical network latency (0.5–1 s per request) plus the 1-second inter-batch
sleep, effective throughput stays well within the SEC limit.

Estimated runtime:
    2,325 symbols × ~1.2 s/symbol ≈ 45 minutes first run
    Re-runs within 6 days skip already-fresh symbols (instant for stale-free cache).

Usage:
    python scripts/fetch_edgar_universe.py              # full tradeable universe
    python scripts/fetch_edgar_universe.py --limit 20   # first 20 symbols (test)
    python scripts/fetch_edgar_universe.py --force      # re-fetch all, ignore cache age
    python scripts/fetch_edgar_universe.py --universe sp500_sample  # smaller set
"""

from __future__ import annotations

import sys
import time
from argparse import ArgumentParser
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quantlab.execution import load_universe
from quantlab.providers.edgar import (
    _count_consecutive_beats,
    _ensure_edgar_table,
    _save_edgar_cache,
    compute_earnings_acceleration,
    fetch_fundamentals,
)
from quantlab.storage import DB_PATH

BATCH_SIZE  = 50    # symbols per batch
BATCH_SLEEP = 1.0   # seconds between batches (SEC fair-use guard)
_FETCH_METRICS = ["eps_diluted", "net_income", "revenue"]


# ── Per-symbol helpers ─────────────────────────────────────────────────────────

_CACHE_MAX_AGE_DAYS = 6   # re-fetch data older than this many days

def _is_recently_cached(symbol: str, con) -> bool:
    """True when edgar_fundamentals has a fresh entry for symbol (within 6 days).

    6-day window lets the Monday weekly job re-fetch data that is up to a week
    old without re-hitting the SEC for symbols already refreshed this week.
    """
    cutoff = (date.today() - timedelta(days=_CACHE_MAX_AGE_DAYS)).isoformat()
    row = con.execute(
        "SELECT 1 FROM edgar_fundamentals WHERE symbol = ? AND fetch_date >= ?",
        [symbol, cutoff],
    ).fetchone()
    return row is not None


def _process_symbol(symbol: str, force: bool, con) -> str:
    """
    Fetch and store EDGAR fundamentals for one symbol.

    Returns:
        "fetched"  — successfully fetched and stored
        "skipped"  — cached within last 6 days (and force=False)
        "failed"   — not in SEC index or other error
    """
    if not force and _is_recently_cached(symbol, con):
        return "skipped"

    try:
        snap = fetch_fundamentals(symbol, metrics=_FETCH_METRICS)
        score      = compute_earnings_acceleration(snap)
        consecutive = _count_consecutive_beats(
            snap.eps_history or snap.net_income_history
        )
        _save_edgar_cache(symbol, snap, score, consecutive)
        return "fetched"
    except ValueError:
        # Ticker not found in SEC filing index — common for foreign-listed or
        # recently-IPO'd names.  Not a network error; log at debug level only.
        return "failed"
    except Exception:
        return "failed"


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = ArgumentParser(
        description="Populate EDGAR fundamentals for all tradeable symbols."
    )
    parser.add_argument(
        "--universe", default="tradeable",
        help="Universe name or comma-separated symbols (default: tradeable)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N symbols — useful for testing",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-fetch all symbols, ignoring cache age",
    )
    args = parser.parse_args()

    symbols = load_universe(args.universe)
    if args.limit:
        symbols = symbols[: args.limit]

    n         = len(symbols)
    n_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE
    eta_lo    = n * 0.5 / 60
    eta_hi    = n * 1.5 / 60

    print(f"\n{'='*62}")
    print(f"  EDGAR Universe Fundamentals Fetch")
    print(f"  Universe : {args.universe}  ({n} symbols)")
    print(f"  Batch    : {BATCH_SIZE} symbols  |  {BATCH_SLEEP}s sleep between batches")
    print(f"  ETA      : {eta_lo:.0f}–{eta_hi:.0f} min  (cache hits finish instantly)")
    print(f"{'='*62}\n")

    import duckdb

    total_fetched = total_skipped = total_failed = 0

    for batch_idx in range(n_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch_end   = min(batch_start + BATCH_SIZE, n)
        batch       = symbols[batch_start:batch_end]

        # One connection per batch — reduces lock contention across symbols
        con = duckdb.connect(str(DB_PATH))
        _ensure_edgar_table(con)

        for symbol in batch:
            result = _process_symbol(symbol, args.force, con)
            if result == "fetched":
                total_fetched += 1
            elif result == "skipped":
                total_skipped += 1
            else:
                total_failed += 1

        con.close()

        done = batch_end
        if done % 100 == 0 or done == n:
            pct = done / n * 100
            print(
                f"  [{done:>5}/{n}  {pct:5.1f}%]  "
                f"fetched={total_fetched:>4}  "
                f"skipped={total_skipped:>4}  "
                f"failed={total_failed:>4}"
            )

        if batch_idx < n_batches - 1:
            time.sleep(BATCH_SLEEP)

    # ── Summary and sample rows ────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  Complete.")
    print(f"  fetched={total_fetched}  skipped={total_skipped}  failed={total_failed}")
    print(f"{'='*62}")

    # Show a sample of stored rows for verification
    try:
        con = duckdb.connect(str(DB_PATH))
        _ensure_edgar_table(con)
        rows = con.execute(
            """
            SELECT symbol, fetch_date, acceleration_score,
                   revenue_growth, eps_growth, consecutive_beats, eps_diluted
            FROM edgar_fundamentals
            WHERE fetch_date = ?
            ORDER BY acceleration_score DESC NULLS LAST
            LIMIT 10
            """,
            [date.today().isoformat()],
        ).fetchall()
        con.close()

        if rows:
            print(f"\n  Top 10 stored today (by acceleration_score):")
            print(f"  {'Symbol':<8} {'Score':>6} {'RevYoY':>8} {'EpsYoY':>8} "
                  f"{'Beats':>6} {'EPS/q':>8}")
            print(f"  {'-'*56}")
            for r in rows:
                sym, fetch_d, accel, rev_g, eps_g, beats, eps_q = r
                def _fmt(v, pct=False):
                    if v is None:
                        return "    N/A"
                    return f"{v*100:>7.1f}%" if pct else f"{v:>8.4f}"
                print(
                    f"  {sym:<8} {_fmt(accel):>6}  "
                    f"{_fmt(rev_g, pct=True):>8}  {_fmt(eps_g, pct=True):>8}  "
                    f"{(beats or 0):>5}  {_fmt(eps_q):>8}"
                )
        else:
            print("\n  (no rows stored today — all were cache hits or failures)")
    except Exception as exc:
        print(f"  Sample query failed: {exc}")

    print()


if __name__ == "__main__":
    main()
