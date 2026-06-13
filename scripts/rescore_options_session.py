"""
scripts/rescore_options_session.py — Re-run a session through the
recalibrated relative options scorer.

For every symbol the intraday monitor snapshotted on the given date
(options_snapshots rows), recomputes:

  - today's total call volume from the session's EOD options flat file
    (final volumes — more accurate than the intraday REST partials the
    monitor saw)
  - the per-symbol z-score vs the symbol's own trailing 20-session
    flat-file baseline
  - the relative composite score (reusing the session's stored PCR and
    IV skew — those were chain-snapshot values that cannot be refetched
    retroactively)
  - the cross-sectional top-percentile gate (default p90)

Dry-run by default: prints the would-be flags without touching DuckDB.
--write persists vol_zscore / rel_score / unusual_flag back onto the
session's options_snapshots rows.

Usage:
    python scripts/rescore_options_session.py --date 2026-06-11
    python scripts/rescore_options_session.py --date 2026-06-11 --write
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main() -> None:
    parser = ArgumentParser(description="Rescore a session with the relative options scorer.")
    parser.add_argument("--date", required=True, help="Session date (YYYY-MM-DD)")
    parser.add_argument("--percentile", type=float, default=None,
                        help="Cross-sectional gate percentile (default: scanner config, 90)")
    parser.add_argument("--write", action="store_true",
                        help="Persist vol_zscore/rel_score/unusual_flag to options_snapshots")
    args = parser.parse_args()

    session = date.fromisoformat(args.date)

    from quantlab.storage import DB_PATH, connect_with_retry
    from quantlab.utils import get_config
    from quantlab.providers.flat_files import FlatFileProvider
    # Shared with the automated finalizer (scripts/finalize_sessions.py) so the
    # manual override and the overnight path can never drift apart.
    from quantlab.options_finalize import compute_session_scores

    pctl = args.percentile if args.percentile is not None else float(
        get_config("scanner").get("options_unusual_percentile", 90.0)
    )

    con = connect_with_retry(DB_PATH, read_only=True)
    n_rows = con.execute(
        "SELECT COUNT(*) FROM options_snapshots WHERE snap_date = ?",
        [session],
    ).fetchone()[0]
    if not n_rows:
        con.close()
        print(f"No options_snapshots rows for {session} — nothing to rescore.")
        return

    flat = FlatFileProvider()
    try:
        ss = compute_session_scores(session, con, flat, pctl)
    except Exception as exc:
        con.close()
        print(f"Options flat file for {session} unavailable: {exc}")
        return
    con.close()

    results        = ss.results
    flagged        = ss.flagged
    put_dominated  = ss.put_dominated   # PCR-ceiling-blocked: future short-side data
    floor_blocked  = ss.floor_blocked   # would have flagged on score+z alone
    print(f"Rescoring {n_rows} symbols for {session} — "
          f"{ss.n_baseline} baseline sessions, gate p{pctl:g}\n")

    scores = {sym: r["rel_score"] for sym, r in results.items()}
    _cfg     = get_config("scanner")
    min_base = float(_cfg.get("options_min_baseline_contracts", 75))
    max_pcr  = float(_cfg.get("options_gate_max_pcr", 1.5))

    def _f2(v, width: int = 6) -> str:
        """NULL-safe column format — '—' for unmeasured values (MISSING ≠ ZERO)."""
        return f"{v:>{width}.2f}" if v is not None else "—".rjust(width)

    n_scored = sum(1 for v in scores.values() if v is not None)
    print(f"{'SYM':<8} {'CALL_VOL':>10} {'BASE_AVG':>10} {'VOL_Z':>7} "
          f"{'PCR':>6} {'SKEW':>6} {'REL':>7} {'LEGACY':>7}")
    for sym in sorted(flagged, key=lambda s: -(scores[s] or 0.0)):
        r = results[sym]
        print(f"{sym:<8} {r['call_volume']:>10,.0f} {r['baseline_mean']:>10,.0f} "
              f"{r['vol_zscore']:>7.2f} {_f2(r['pcr'])} {r['iv_skew']:>6.2f} "
              f"{r['rel_score']:>7.4f} {_f2(r['legacy_score'], 7)}")

    if floor_blocked:
        print(f"\nFloor-blocked (baseline avg < {min_base:g} contracts — "
              f"scored/persisted, no gate credit):")
        for sym in sorted(floor_blocked, key=lambda s: -(scores[s] or 0.0)):
            r = results[sym]
            print(f"  {sym:<8} base_avg={r['baseline_mean']:>8,.0f}  "
                  f"vol_z={r['vol_zscore']:>6.2f}  rel={r['rel_score']:.4f}")

    if put_dominated:
        print(f"\nPut-dominated (PCR > {max_pcr:g} — no LONG flag; tagged as "
              f"short-side signal data):")
        for sym in sorted(put_dominated, key=lambda s: -(scores[s] or 0.0)):
            r = results[sym]
            print(f"  {sym:<8} pcr={_f2(r['pcr'])}  vol_z={r['vol_zscore']:>6.2f}  "
                  f"rel={r['rel_score']:.4f}")

    rate = (len(flagged) / n_scored) if n_scored else 0.0
    print(f"\nOptions: {len(flagged)}/{n_scored} unusual, {rate:.1%}"
          f"   (legacy scorer ≥0.6: "
          f"{sum(1 for r in results.values() if (r['legacy_score'] or 0) >= 0.6)}"
          f"/{len(results)})")
    if n_scored < len(results):
        print(f"Not scored (baseline too short): {len(results) - n_scored}")

    if args.write:
        from quantlab.providers.massive_options import MassiveOptionsProvider
        con = connect_with_retry(DB_PATH)
        # Run the schema migration (adds the relative-scoring columns)
        MassiveOptionsProvider._ensure_table(con)
        for sym, r in results.items():
            con.execute(
                """
                UPDATE options_snapshots
                SET call_volume = ?, vol_zscore = ?, rel_score = ?
                WHERE symbol = ? AND snap_date = ?
                """,
                [r["call_volume"], r["vol_zscore"], r["rel_score"],
                 sym, session],
            )
        con.close()
        # Flags + put_dominated tag + flag freshness (first_flagged_date /
        # flag_streak) all persist through the one shared writer
        MassiveOptionsProvider(api_key="").mark_unusual_flags(
            flagged, snap_date=session, put_dominated=put_dominated,
        )
        print(f"Persisted rescored values for {len(results)} rows.")
    else:
        print("Dry run — use --write to persist.")


if __name__ == "__main__":
    main()
