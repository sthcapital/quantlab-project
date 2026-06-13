"""
scripts/purge_watchlist.py — purge non-qualifying institutional_watchlist entries.

After the growth filter is activated, the candidate watchlist still holds names
ingested under the old (pre-filter) regime.  This removes every
institutional_watchlist entry whose symbol is NOT in the current
growth_universe qualified set, EXCEPT any symbol that is a CURRENTLY-OPEN
position (status='watching') — a live trade's candidate entry is preserved so
reconcile_positions.py owns its fate.  Positions already CLOSED (flattened in
step 2) are not open, so they are purged like any other non-qualifier, leaving
the candidate watchlist strictly growth-qualified.

Every removed symbol is logged to watchlist_purge_log (date, symbol, reason)
BEFORE deletion — the full removed list is auditable.  Removal is a plain
DELETE: institutional_watchlist is keyed by symbol with no dependent state, and
the upsert re-inserts a fresh row (new first_seen, consecutive_days=1) when a
purged name requalifies, so re-entry is never blocked.

Usage:
    python scripts/purge_watchlist.py --dry-run
    python scripts/purge_watchlist.py
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import duckdb

from quantlab.storage import DB_PATH, _ensure_schema

PURGE_REASON = "not in growth_universe qualified set — growth-filter activation purge"


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


def _open_position_symbols(con) -> set[str]:
    """Symbols that are CURRENTLY-OPEN positions (status='watching').

    Only live trades are protected — their candidate entry is preserved so
    reconcile_positions.py owns their fate.  Closed (flattened) positions are
    not open and are purged like any other non-qualifier.
    """
    return {r[0] for r in con.execute(
        "SELECT symbol FROM watchlist WHERE status = 'watching'").fetchall()}


def purge(db_path: str | None = None, dry_run: bool = False) -> dict:
    con = duckdb.connect(db_path or str(DB_PATH))
    _ensure_schema(con)
    as_of, qualified = _latest_qualified(con)
    protected = _open_position_symbols(con)
    today = date.today()

    iwl = [r[0] for r in con.execute(
        "SELECT symbol FROM institutional_watchlist ORDER BY symbol").fetchall()]
    to_remove = [s for s in iwl if s not in qualified and s not in protected]
    retained = [s for s in iwl if s in qualified or s in protected]

    print(f"\n{'='*72}")
    print(f"  PURGE WATCHLIST → growth_universe qualified set "
          f"(as_of {as_of}, {len(qualified)} qualified)")
    print(f"{'='*72}")
    print(f"  institutional_watchlist total : {len(iwl)}")
    print(f"  retained (qualified or open-position): {len(retained)}")
    print(f"    · qualified: {len([s for s in retained if s in qualified])}")
    print(f"    · open-position (protected): "
          f"{sorted(s for s in retained if s in protected)}")
    print(f"  TO REMOVE : {len(to_remove)}")

    if not dry_run:
        for sym in to_remove:
            con.execute(
                "INSERT OR REPLACE INTO watchlist_purge_log "
                "(purge_date, symbol, reason) VALUES (?, ?, ?)",
                [today.isoformat(), sym, PURGE_REASON],
            )
        # DELETE after the log is written.
        ph = ",".join("?" * len(to_remove)) if to_remove else "''"
        if to_remove:
            con.execute(
                f"DELETE FROM institutional_watchlist WHERE symbol IN ({ph})",
                to_remove,
            )
        remaining = con.execute(
            "SELECT count(*) FROM institutional_watchlist").fetchone()[0]
        logged = con.execute(
            "SELECT count(*) FROM watchlist_purge_log WHERE purge_date = ?",
            [today.isoformat()]).fetchone()[0]
        print(f"  removed (logged): {logged}   institutional_watchlist remaining: {remaining}")
    else:
        print("  [DRY RUN] no state changed")

    con.close()
    return {"as_of": as_of, "removed": to_remove, "retained": retained}


def main() -> None:
    ap = ArgumentParser(description="Purge non-qualifying watchlist entries.")
    ap.add_argument("--dry-run", action="store_true", help="report only; no mutation")
    args = ap.parse_args()
    purge(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
