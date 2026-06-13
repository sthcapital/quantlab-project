"""
quantlab.options_finalize — finalize an options session against the EOD flat file.

Eliminates the manual ``rescore_options_session.py --write`` step.  An options
session is *finalized* when its options_snapshots rows have been rescored
against Polygon's end-of-day options flat file (final OPRA volumes) rather than
the intraday REST partials the monitor saw during the session.

Normal flow given Polygon's publication timing:

  * Evening scan (5:00 PM ET) calls ``finalize_session(today)`` as its first
    step.  The same-day flat file is usually NOT published yet, so the session
    is recorded ``finalized = FALSE`` (basis ``intraday``) and the scan proceeds
    on the monitor's intraday flags.
  * The next MORNING job calls ``sweep_unfinalized()`` — by then yesterday's
    file has landed, so the prior session finalizes (basis ``final``).

Publication vs permission (2026-06-12 incident): a same-day S3 403 looked
identical to an access-denied credential failure.  ``finalize_session`` probes
the object first (HEAD); on a 403 it re-probes a known-published prior date to
prove the credentials still work, and only calls it a credential failure when
that probe also fails.  ``stale_unfinalized_sessions`` surfaces any session
still unfinalized past noon the next trading day so a real permission failure
cannot silently stall finalization forever.

Session status lives in the ``options_session_status`` table; the per-row
rel_score / unusual_flag finalization routes through
``MassiveOptionsProvider.mark_unusual_flags`` exactly as the rescore path does.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Session basis values (also rendered in the report header).
BASIS_FINAL = "final"
BASIS_INTRADAY = "intraday"

# finalize_session outcome statuses.
STATUS_FINAL = "final"                       # rescored against the EOD flat file
STATUS_INTRADAY = "intraday"                 # file not published yet — intraday stands
STATUS_NO_ROWS = "no_rows"                   # no options_snapshots for the session
STATUS_CREDENTIAL_FAILURE = "credential_failure"  # 403 AND prior-date probe failed
STATUS_ERROR = "error"                       # transient/unexpected probe failure


@dataclass
class FinalizeResult:
    session: date
    status: str
    basis: str
    finalized: bool
    n_scored: int = 0
    n_flagged: int = 0
    note: str = ""

    @property
    def credential_failure(self) -> bool:
        return self.status == STATUS_CREDENTIAL_FAILURE

    def summary(self) -> str:
        return (f"{self.session}: {self.status} "
                f"(basis={self.basis}, {self.n_flagged}/{self.n_scored} unusual)"
                + (f" — {self.note}" if self.note else ""))


# ── Session-status table ───────────────────────────────────────────────────────

def ensure_session_status_table(con) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS options_session_status (
            snap_date    DATE PRIMARY KEY,
            finalized    BOOLEAN,
            basis        VARCHAR,
            finalized_at TIMESTAMP,
            note         VARCHAR
        )
    """)


def set_session_status(
    con, snap_date: date, *, finalized: bool, basis: str, note: str = "",
) -> None:
    ensure_session_status_table(con)
    con.execute(
        """
        INSERT OR REPLACE INTO options_session_status
            (snap_date, finalized, basis, finalized_at, note)
        VALUES (?, ?, ?, ?, ?)
        """,
        [snap_date, finalized, basis,
         datetime.now() if finalized else None, note],
    )


def get_session_status(snap_date: date, db_path=None, con=None) -> Optional[dict]:
    """Return {finalized, basis, finalized_at, note} for the session, or None."""
    own = con is None
    if own:
        from quantlab.storage import DB_PATH, connect_with_retry
        con = connect_with_retry(db_path or DB_PATH, read_only=True)
    try:
        row = con.execute(
            "SELECT finalized, basis, finalized_at, note "
            "FROM options_session_status WHERE snap_date = ?",
            [snap_date],
        ).fetchone()
    except Exception:
        return None                          # table absent on an old DB
    finally:
        if own:
            con.close()
    if not row:
        return None
    return {"finalized": bool(row[0]), "basis": row[1],
            "finalized_at": row[2], "note": row[3] or ""}


# ── Core rescore (shared with rescore_options_session.py) ──────────────────────

@dataclass
class SessionScores:
    results: dict           # symbol -> {call_volume, baseline_mean, vol_zscore, rel_score, pcr, iv_skew, legacy_score}
    flagged: set
    put_dominated: set
    floor_blocked: set
    n_scored: int
    n_baseline: int = 0     # trailing flat-file sessions used for the baseline


def compute_session_scores(session: date, con, flat, percentile: float) -> SessionScores:
    """
    Recompute the relative options scores for ``session`` from the EOD flat
    file (final OPRA call volumes) and the symbol's own trailing-20 baseline.

    Assumes the session's flat file is already downloadable (the caller probed
    access first).  Returns the scores plus the cross-sectional gate result —
    the single source of truth shared by the manual rescore script and the
    automated finalizer so the two can never drift.
    """
    from quantlab.utils import get_config
    from quantlab.signals.options_relative import (
        cross_sectional_flags,
        relative_options_score,
        volume_zscore,
    )

    rows = con.execute(
        "SELECT symbol, pcr, iv_skew, options_score FROM options_snapshots "
        "WHERE snap_date = ? ORDER BY symbol",
        [session],
    ).fetchall()

    flat.download_options_day(session)       # cache the EOD file (no-op on hit)
    session_vols = flat.get_underlying_call_volumes(session)
    history = flat.get_call_volume_history(session, n_sessions=20)

    results: dict = {}
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

    cfg = get_config("scanner")
    min_base = float(cfg.get("options_min_baseline_contracts", 75))
    max_pcr = float(cfg.get("options_gate_max_pcr", 1.5))
    scores = {s: r["rel_score"] for s, r in results.items()}
    zscores = {s: r["vol_zscore"] for s, r in results.items()}
    base_means = {s: r["baseline_mean"] for s, r in results.items()}
    pcrs = {s: r["pcr"] for s, r in results.items()}

    flagged = cross_sectional_flags(
        scores, percentile_cut=percentile, zscores=zscores,
        baseline_means=base_means, min_baseline=min_base,
        pcrs=pcrs, max_pcr=max_pcr,
    )
    put_dominated = cross_sectional_flags(
        scores, percentile_cut=percentile, zscores=zscores,
        baseline_means=base_means, min_baseline=min_base,
    ) - flagged
    floor_blocked = cross_sectional_flags(
        scores, percentile_cut=percentile, zscores=zscores,
    ) - flagged - put_dominated

    n_scored = sum(1 for v in scores.values() if v is not None)
    return SessionScores(results, flagged, put_dominated, floor_blocked,
                         n_scored, n_baseline=len(history))


# ── Finalization ───────────────────────────────────────────────────────────────

def _config_percentile(percentile: Optional[float]) -> float:
    if percentile is not None:
        return percentile
    from quantlab.utils import get_config
    return float(get_config("scanner").get("options_unusual_percentile", 90.0))


def _latest_cached_options_date(flat, before: date) -> Optional[date]:
    """Most recent date with a cached options parquet strictly before ``before``."""
    from datetime import timedelta
    d = before - timedelta(days=1)
    limit = before - timedelta(days=30)
    while d >= limit:
        if flat.options_cache_path(d).exists():
            return d
        d -= timedelta(days=1)
    return None


def _persist_final(con, session: date, scores: SessionScores) -> None:
    """Write finalized rel-score columns and route flags through mark_unusual_flags."""
    from quantlab.providers.massive_options import MassiveOptionsProvider

    MassiveOptionsProvider._ensure_table(con)
    for sym, r in scores.results.items():
        con.execute(
            """
            UPDATE options_snapshots
            SET call_volume = ?, vol_zscore = ?, rel_score = ?
            WHERE symbol = ? AND snap_date = ?
            """,
            [r["call_volume"], r["vol_zscore"], r["rel_score"], sym, session],
        )
    # mark_unusual_flags opens its own retrying connection — the shared writer
    # for flags + put_dominated tag + flag freshness (same path as the rescore
    # script's --write).
    MassiveOptionsProvider(api_key="").mark_unusual_flags(
        scores.flagged, snap_date=session, put_dominated=scores.put_dominated,
    )


def finalize_session(
    session: date,
    *,
    percentile: Optional[float] = None,
    db_path=None,
    flat=None,
    con=None,
) -> FinalizeResult:
    """
    Attempt to finalize ``session`` against its EOD options flat file.

    Returns a FinalizeResult and records the outcome in options_session_status.
    Never raises on a missing/denied flat file: it degrades to an unfinalized
    ``intraday`` session (the monitor's intraday flags stand) and, on a 403 it
    cannot attribute to absence, flags a credential failure for the loud alert.
    """
    from quantlab.storage import DB_PATH, connect_with_retry

    own_con = con is None
    con = con or connect_with_retry(db_path or DB_PATH)
    try:
        n_rows = con.execute(
            "SELECT COUNT(*) FROM options_snapshots WHERE snap_date = ?",
            [session],
        ).fetchone()[0]
        if not n_rows:
            return FinalizeResult(session, STATUS_NO_ROWS, BASIS_INTRADAY,
                                  finalized=False, note="no options_snapshots rows")

        if flat is None:
            from quantlab.providers.flat_files import FlatFileProvider
            flat = FlatFileProvider()
        pctl = _config_percentile(percentile)

        access = flat.probe_options_access(session)

        if access == "ok":
            scores = compute_session_scores(session, con, flat, pctl)
            _persist_final(con, session, scores)
            note = "rescored against EOD flat file"
            set_session_status(con, session, finalized=True,
                               basis=BASIS_FINAL, note=note)
            return FinalizeResult(session, STATUS_FINAL, BASIS_FINAL,
                                  finalized=True, n_scored=scores.n_scored,
                                  n_flagged=len(scores.flagged), note=note)

        if access == "not_found":
            note = "EOD flat file not published yet (404)"
            set_session_status(con, session, finalized=False,
                               basis=BASIS_INTRADAY, note=note)
            return FinalizeResult(session, STATUS_INTRADAY, BASIS_INTRADAY,
                                  finalized=False, note=note)

        if access == "denied":
            # 403 masks an absent object on a bucket without ListBucket — prove
            # the credentials still work against a known-published prior date.
            probe_date = _latest_cached_options_date(flat, before=session)
            creds_ok = (probe_date is not None
                        and flat.probe_options_access(probe_date) == "ok")
            if creds_ok:
                note = (f"403 on {session} but credentials verified on "
                        f"{probe_date} — file not published yet (403-masked 404)")
                set_session_status(con, session, finalized=False,
                                   basis=BASIS_INTRADAY, note=note)
                return FinalizeResult(session, STATUS_INTRADAY, BASIS_INTRADAY,
                                      finalized=False, note=note)
            note = ("ACCESS DENIED (403) and credential probe "
                    + (f"on {probe_date} also failed" if probe_date
                       else "impossible (no prior cached date)")
                    + " — check POLYGON_S3_ACCESS_KEY_ID / POLYGON_API_KEY")
            set_session_status(con, session, finalized=False,
                               basis=BASIS_INTRADAY, note=note)
            return FinalizeResult(session, STATUS_CREDENTIAL_FAILURE,
                                  BASIS_INTRADAY, finalized=False, note=note)

        note = "flat-file access probe error — will retry on next sweep"
        set_session_status(con, session, finalized=False,
                           basis=BASIS_INTRADAY, note=note)
        return FinalizeResult(session, STATUS_ERROR, BASIS_INTRADAY,
                              finalized=False, note=note)
    finally:
        if own_con:
            con.close()


# ── Sweep / staleness ──────────────────────────────────────────────────────────

def _status_table_exists(con) -> bool:
    try:
        return con.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_name = 'options_session_status'"
        ).fetchone() is not None
    except Exception:
        return False


# Default sweep window (calendar days back from ``before``): comfortably covers
# the prior trading day plus weekends/holidays without reprocessing months of
# history on a first run.  Older stuck sessions are caught by the staleness
# alert and handled via the manual override.
_SWEEP_LOOKBACK_DAYS = 14


def unfinalized_sessions(
    db_path=None,
    before: Optional[date] = None,
    lookback_days: Optional[int] = None,
    con=None,
) -> list[date]:
    """
    Sessions with options_snapshots rows that are not finalized — no status row
    or status.finalized = FALSE — restricted to ``snap_date < before`` and (when
    ``lookback_days`` is set) to ``snap_date >= before - lookback_days``.  Oldest
    first.  Read-only safe: never creates the status table.
    """
    from datetime import timedelta

    own = con is None
    if own:
        from quantlab.storage import DB_PATH, connect_with_retry
        con = connect_with_retry(db_path or DB_PATH, read_only=True)
    try:
        has_status = _status_table_exists(con)
        params: list = []
        where = ["1 = 1"]
        if before is not None:
            where.append("s.snap_date < ?")
            params.append(before)
        if lookback_days is not None and before is not None:
            where.append("s.snap_date >= ?")
            params.append(before - timedelta(days=lookback_days))
        if has_status:
            sql = (
                "SELECT DISTINCT s.snap_date FROM options_snapshots s "
                "LEFT JOIN options_session_status st ON st.snap_date = s.snap_date "
                "WHERE " + " AND ".join(where)
                + " AND COALESCE(st.finalized, FALSE) = FALSE ORDER BY s.snap_date"
            )
        else:                                 # no pipeline history yet — none finalized
            sql = ("SELECT DISTINCT s.snap_date FROM options_snapshots s "
                   "WHERE " + " AND ".join(where) + " ORDER BY s.snap_date")
        return [r[0] for r in con.execute(sql, params).fetchall()]
    finally:
        if own:
            con.close()


def sweep_unfinalized(
    db_path=None,
    before: Optional[date] = None,
    percentile: Optional[float] = None,
    flat=None,
    lookback_days: int = _SWEEP_LOOKBACK_DAYS,
) -> list[FinalizeResult]:
    """Finalize every recent unfinalized prior session (``before`` defaults to today)."""
    if before is None:
        before = date.today()
    results: list[FinalizeResult] = []
    for session in unfinalized_sessions(db_path=db_path, before=before,
                                        lookback_days=lookback_days):
        results.append(
            finalize_session(session, percentile=percentile,
                             db_path=db_path, flat=flat)
        )
    return results


def stale_unfinalized_sessions(
    now: Optional[datetime] = None, db_path=None, con=None,
) -> list[date]:
    """
    Sessions the finalization pipeline has EXPLICITLY left unfinalized
    (options_session_status.finalized = FALSE) whose deadline — noon the next
    trading day after the session — has already passed.

    Only explicit status rows count: a session with no status row has not been
    through the pipeline (today before the evening scan, or pre-finalization
    history) and is not "stale".  A non-empty result means finalization has
    stalled — almost always a credential/permission failure — and must be
    alerted loudly (health check + report header).  Read-only safe.
    """
    from datetime import time
    from quantlab.market_calendar import next_trading_day

    own = con is None
    if own:
        from quantlab.storage import DB_PATH, connect_with_retry
        con = connect_with_retry(db_path or DB_PATH, read_only=True)
    try:
        if not _status_table_exists(con):
            return []
        rows = con.execute(
            "SELECT snap_date FROM options_session_status "
            "WHERE COALESCE(finalized, FALSE) = FALSE ORDER BY snap_date"
        ).fetchall()
    finally:
        if own:
            con.close()

    now = now or datetime.now()
    stale: list[date] = []
    for (session,) in rows:
        deadline = datetime.combine(next_trading_day(session), time(12, 0))
        if now > deadline:
            stale.append(session)
    return stale
