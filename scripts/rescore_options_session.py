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

    import duckdb
    from quantlab.storage import DB_PATH
    from quantlab.utils import get_config
    from quantlab.providers.flat_files import FlatFileProvider
    from quantlab.signals.options_relative import (
        cross_sectional_flags,
        relative_options_score,
        volume_zscore,
    )

    pctl = args.percentile if args.percentile is not None else float(
        get_config("scanner").get("options_unusual_percentile", 90.0)
    )

    con = duckdb.connect(str(DB_PATH), read_only=True)
    rows = con.execute(
        "SELECT symbol, pcr, iv_skew, options_score FROM options_snapshots "
        "WHERE snap_date = ? ORDER BY symbol",
        [session],
    ).fetchall()
    con.close()

    if not rows:
        print(f"No options_snapshots rows for {session} — nothing to rescore.")
        return

    flat = FlatFileProvider()

    # Session EOD call volumes (downloads the flat file on first access)
    try:
        flat.download_options_day(session)
    except Exception as exc:
        print(f"Options flat file for {session} unavailable: {exc}")
        return
    session_vols = flat.get_underlying_call_volumes(session)

    # Trailing 20-session baselines (cached parquet reads only)
    history = flat.get_call_volume_history(session, n_sessions=20)
    print(f"Rescoring {len(rows)} symbols for {session} — "
          f"{len(history)} baseline sessions, gate p{pctl:g}\n")

    results: dict[str, dict] = {}
    for sym, pcr, iv_skew, legacy_score in rows:
        baseline = [day.get(sym, 0.0) for day in history]
        call_vol = session_vols.get(sym, 0.0)
        vol_z = volume_zscore(call_vol, baseline)
        rel = relative_options_score(vol_z, pcr=pcr, iv_skew=iv_skew)
        results[sym] = {
            "call_volume": call_vol,
            "baseline_mean": (sum(baseline) / len(baseline)) if baseline else None,
            "vol_zscore": vol_z,
            "rel_score": rel,
            "pcr": pcr,
            "iv_skew": iv_skew,
            "legacy_score": legacy_score,
        }

    scores  = {sym: r["rel_score"] for sym, r in results.items()}
    zscores = {sym: r["vol_zscore"] for sym, r in results.items()}
    flagged = cross_sectional_flags(scores, percentile_cut=pctl, zscores=zscores)

    n_scored = sum(1 for v in scores.values() if v is not None)
    print(f"{'SYM':<8} {'CALL_VOL':>10} {'BASE_AVG':>10} {'VOL_Z':>7} "
          f"{'PCR':>6} {'SKEW':>6} {'REL':>7} {'LEGACY':>7}")
    for sym in sorted(flagged, key=lambda s: -(scores[s] or 0.0)):
        r = results[sym]
        print(f"{sym:<8} {r['call_volume']:>10,.0f} {r['baseline_mean']:>10,.0f} "
              f"{r['vol_zscore']:>7.2f} {r['pcr']:>6.2f} {r['iv_skew']:>6.2f} "
              f"{r['rel_score']:>7.4f} {r['legacy_score']:>7.2f}")

    rate = (len(flagged) / n_scored) if n_scored else 0.0
    print(f"\nOptions: {len(flagged)}/{n_scored} unusual, {rate:.1%}"
          f"   (legacy scorer ≥0.6: "
          f"{sum(1 for r in results.values() if (r['legacy_score'] or 0) >= 0.6)}"
          f"/{len(results)})")
    if n_scored < len(results):
        print(f"Not scored (baseline too short): {len(results) - n_scored}")

    if args.write:
        from quantlab.providers.massive_options import MassiveOptionsProvider
        con = duckdb.connect(str(DB_PATH))
        # Run the schema migration (adds the relative-scoring columns)
        MassiveOptionsProvider._ensure_table(con)
        for sym, r in results.items():
            con.execute(
                """
                UPDATE options_snapshots
                SET call_volume = ?, vol_zscore = ?, rel_score = ?,
                    unusual_flag = CASE WHEN ? IS NULL THEN NULL ELSE ? END
                WHERE symbol = ? AND snap_date = ?
                """,
                [r["call_volume"], r["vol_zscore"], r["rel_score"],
                 r["rel_score"], sym in flagged, sym, session],
            )
        con.close()
        print(f"Persisted rescored values for {len(results)} rows.")
    else:
        print("Dry run — use --write to persist.")


if __name__ == "__main__":
    main()
