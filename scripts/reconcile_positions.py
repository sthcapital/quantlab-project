"""
scripts/reconcile_positions.py — reconcile open paper positions to the growth filter.

On growth-filter activation (2026-06-13) the open book may hold names the
filter would never admit (defensive Stage-2 survivors).  This script
cross-references every open position (watchlist.status = 'watching') against the
current growth_universe qualified set and:

  - QUALIFIED  → left open, untouched, tagged "growth-qualified — retained".
  - NOT        → final state journalled to closed_positions (entry, exit=current
                 price, R-multiple, P&L, days held, reason) BEFORE the watchlist
                 row is marked 'closed'.

Report first, act second: the classification is printed for every position; pass
--dry-run to print without mutating state.

Usage:
    python scripts/reconcile_positions.py --dry-run
    python scripts/reconcile_positions.py
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import duckdb

from quantlab.storage import DB_PATH, _ensure_schema

DEFAULT_REASON = "pre-filter defensive — flattened on growth-filter activation"
# SNEX additionally carries the gate-bug provenance.
SPECIAL_REASONS = {
    "SNEX": "entered-via-gate-bug + non-qualifying",
}
RETAINED_NOTE = "growth-qualified — retained"


def _latest_qualified(con) -> tuple[date | None, set[str]]:
    row = con.execute("SELECT max(as_of_date) FROM growth_universe").fetchone()
    as_of = row[0] if row else None
    if as_of is None:
        return None, set()
    syms = {
        r[0] for r in con.execute(
            "SELECT symbol FROM growth_universe "
            "WHERE as_of_date = ? AND growth_qualified",
            [as_of],
        ).fetchall()
    }
    return as_of, syms


def _r_multiple(entry, exit_, stop):
    if entry is None or exit_ is None or stop is None:
        return None
    risk = entry - stop
    if risk <= 0:
        return None
    return round((exit_ - entry) / risk, 4)


def reconcile(db_path: str | None = None, dry_run: bool = False) -> dict:
    con = duckdb.connect(db_path or str(DB_PATH))
    _ensure_schema(con)
    as_of, qualified = _latest_qualified(con)
    today = date.today()

    rows = con.execute(
        """
        SELECT symbol, entry_price, current_price, unrealized_ret, days_on_watch,
               atr_stop, date_added
        FROM watchlist WHERE status = 'watching'
        ORDER BY symbol
        """
    ).fetchall()

    retained, flattened = [], []
    print(f"\n{'='*72}")
    print(f"  RECONCILE OPEN POSITIONS → growth_universe qualified set "
          f"(as_of {as_of}, {len(qualified)} qualified)")
    print(f"{'='*72}")
    print(f"  {'sym':<6}{'entry':>9}{'exit':>9}{'P&L%':>8}{'R':>7}{'days':>6}  decision")

    for sym, entry, cur, ret, days, stop, d_added in rows:
        is_q = sym in qualified
        r_mult = _r_multiple(entry, cur, stop)
        pnl_pct = ret if ret is not None else (
            (cur - entry) / entry if entry else None)
        decision = "RETAIN (qualified)" if is_q else "FLATTEN"
        print(f"  {sym:<6}{entry:>9.2f}{(cur or 0):>9.2f}"
              f"{(pnl_pct*100 if pnl_pct is not None else 0):>7.1f}%"
              f"{(r_mult if r_mult is not None else 0):>7.2f}{(days or 0):>6}  {decision}")

        if is_q:
            retained.append(sym)
            if not dry_run:
                con.execute(
                    "UPDATE watchlist SET breadth_override_note = ?, date_updated = ? "
                    "WHERE symbol = ? AND status = 'watching'",
                    [RETAINED_NOTE, today.isoformat(), sym],
                )
        else:
            reason = SPECIAL_REASONS.get(sym, DEFAULT_REASON)
            flattened.append((sym, reason))
            if not dry_run:
                # Journal BEFORE closing — the audit trail must survive the close.
                con.execute(
                    """
                    INSERT OR REPLACE INTO closed_positions
                        (symbol, entry_date, exit_date, entry_price, exit_price,
                         atr_stop, r_multiple, pnl_pct, days_held, reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [sym, d_added, today.isoformat(), entry, cur, stop,
                     r_mult, pnl_pct, days, reason],
                )
                con.execute(
                    "UPDATE watchlist SET status = 'closed', "
                    "breadth_override_note = ?, current_price = ?, date_updated = ? "
                    "WHERE symbol = ? AND status = 'watching'",
                    [reason, cur, today.isoformat(), sym],
                )

    con.close()
    print(f"\n  retained (qualified): {len(retained)}  {retained}")
    print(f"  flattened: {len(flattened)}  {[s for s, _ in flattened]}")
    if dry_run:
        print("  [DRY RUN] no state changed")
    return {"as_of": as_of, "retained": retained, "flattened": flattened}


def main() -> None:
    ap = ArgumentParser(description="Reconcile open positions to the growth filter.")
    ap.add_argument("--dry-run", action="store_true", help="report only; no mutation")
    args = ap.parse_args()
    reconcile(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
