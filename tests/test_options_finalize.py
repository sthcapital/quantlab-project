"""
quantlab.options_finalize — auto-finalize options sessions against the EOD flat
file (item 3).  Covers the status table, the four finalize_session outcomes
(final / not-yet-published / 403-masked-404 / credential failure), the morning
sweep, and the noon-next-trading-day staleness deadline.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

import quantlab.storage as storage
from quantlab.providers.massive_options import MassiveOptionsProvider
from quantlab.providers.flat_files import FlatFileProvider
import quantlab.options_finalize as fz


# ── Test doubles ───────────────────────────────────────────────────────────────

class FakeFlat:
    """Flat-file provider stub: scripted S3 access + canned volumes."""

    def __init__(self, access, vols, history, cache_dir: Path, cached_dates=()):
        self._access = access          # {date: 'ok'|'not_found'|'denied'|'error'}
        self._vols = vols              # {date: {symbol: call_volume}}
        self._history = history        # [{symbol: call_volume}, ...]
        self._cache_dir = cache_dir
        for d in cached_dates:         # make options_cache_path(d).exists() True
            (cache_dir / f"options_{d.isoformat()}.parquet").write_text("x")

    def probe_options_access(self, d):
        return self._access.get(d, "not_found")

    def download_options_day(self, d):
        return []

    def get_underlying_call_volumes(self, d):
        return self._vols.get(d, {})

    def get_call_volume_history(self, d, n_sessions=20):
        return self._history

    def options_cache_path(self, d):
        return self._cache_dir / f"options_{d.isoformat()}.parquet"


def _err(status, code):
    from botocore.exceptions import ClientError
    return ClientError(
        {"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}},
        "HeadObject",
    )


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def db(tmp_path, monkeypatch):
    """A temp DuckDB seeded with options_snapshots, wired as the default DB."""
    import duckdb
    path = tmp_path / "test.duckdb"
    monkeypatch.setattr(storage, "DB_PATH", path)
    con = duckdb.connect(str(path))
    MassiveOptionsProvider._ensure_table(con)
    con.close()
    return path


def _seed_session(db_path, session: date, symbols):
    """Insert intraday options_snapshots rows (rel_score NULL until finalized)."""
    import duckdb
    con = duckdb.connect(str(db_path))
    for i, sym in enumerate(symbols):
        con.execute(
            "INSERT INTO options_snapshots "
            "(symbol, snap_date, spot_price, pcr, iv_skew, options_score, "
            " call_count, put_count) VALUES (?,?,?,?,?,?,?,?)",
            [sym, session, 100.0 + i, 0.8, 0.1, 0.5, 500, 300],
        )
    con.close()


# ── classify_s3_error: 403 vs 404 ──────────────────────────────────────────────

def test_classify_404_is_not_found():
    assert FlatFileProvider.classify_s3_error(_err(404, "NoSuchKey")) == "not_found"


def test_classify_403_is_denied():
    assert FlatFileProvider.classify_s3_error(_err(403, "AccessDenied")) == "denied"


def test_classify_other_is_error():
    assert FlatFileProvider.classify_s3_error(_err(500, "InternalError")) == "error"


# ── Status table round-trip ─────────────────────────────────────────────────────

def test_session_status_roundtrip(db):
    import duckdb
    con = duckdb.connect(str(db))
    fz.set_session_status(con, date(2026, 6, 12), finalized=True,
                          basis=fz.BASIS_FINAL, note="x")
    con.close()
    st = fz.get_session_status(date(2026, 6, 12))
    assert st["finalized"] is True
    assert st["basis"] == "final"
    assert fz.get_session_status(date(2026, 1, 1)) is None


# ── finalize_session outcomes ───────────────────────────────────────────────────

def _flat_with_volumes(session, cache_dir, access):
    # Baseline ~500 contracts for 20 sessions; one symbol spikes hard so the
    # cross-sectional gate has something to flag and a healthy baseline mean.
    history = [{"AAA": 500.0, "BBB": 500.0, "CCC": 500.0} for _ in range(20)]
    vols = {session: {"AAA": 50000.0, "BBB": 520.0, "CCC": 480.0}}
    return FakeFlat(access, vols, history, cache_dir)


def test_finalize_ok_marks_final_and_writes_scores(db, tmp_path):
    session = date(2026, 6, 12)
    _seed_session(db, session, ["AAA", "BBB", "CCC"])
    flat = _flat_with_volumes(session, tmp_path, {session: "ok"})

    res = fz.finalize_session(session, flat=flat)

    assert res.status == fz.STATUS_FINAL
    assert res.basis == fz.BASIS_FINAL and res.finalized is True
    assert res.n_scored == 3
    st = fz.get_session_status(session)
    assert st["finalized"] is True
    # rel_score columns were populated by the finalize write
    import duckdb
    con = duckdb.connect(str(db))
    n_rel = con.execute(
        "SELECT COUNT(rel_score) FROM options_snapshots WHERE snap_date = ?",
        [session]).fetchone()[0]
    con.close()
    assert n_rel == 3


def test_finalize_not_published_marks_intraday(db, tmp_path):
    session = date(2026, 6, 12)
    _seed_session(db, session, ["AAA", "BBB"])
    flat = FakeFlat({session: "not_found"}, {}, [], tmp_path)

    res = fz.finalize_session(session, flat=flat)

    assert res.status == fz.STATUS_INTRADAY
    assert res.finalized is False
    assert "404" in res.note
    assert fz.get_session_status(session)["finalized"] is False


def test_finalize_403_masked_404_when_creds_verified(db, tmp_path):
    """403 on the target but a prior cached date probes OK → not a cred failure."""
    session = date(2026, 6, 12)
    prior = date(2026, 6, 11)
    _seed_session(db, session, ["AAA", "BBB"])
    flat = FakeFlat(
        access={session: "denied", prior: "ok"},
        vols={}, history=[], cache_dir=tmp_path, cached_dates=[prior],
    )

    res = fz.finalize_session(session, flat=flat)

    assert res.status == fz.STATUS_INTRADAY
    assert res.finalized is False
    assert "verified" in res.note.lower()


def test_finalize_403_credential_failure_when_probe_also_denied(db, tmp_path):
    session = date(2026, 6, 12)
    prior = date(2026, 6, 11)
    _seed_session(db, session, ["AAA", "BBB"])
    flat = FakeFlat(
        access={session: "denied", prior: "denied"},
        vols={}, history=[], cache_dir=tmp_path, cached_dates=[prior],
    )

    res = fz.finalize_session(session, flat=flat)

    assert res.status == fz.STATUS_CREDENTIAL_FAILURE
    assert res.credential_failure is True
    assert res.finalized is False


def test_finalize_no_rows(db, tmp_path):
    flat = FakeFlat({}, {}, [], tmp_path)
    res = fz.finalize_session(date(2026, 6, 9), flat=flat)
    assert res.status == fz.STATUS_NO_ROWS


# ── Sweep ───────────────────────────────────────────────────────────────────────

def test_sweep_finalizes_only_unfinalized_priors(db, tmp_path):
    s1, s2, today = date(2026, 6, 10), date(2026, 6, 11), date(2026, 6, 12)
    _seed_session(db, s1, ["AAA", "BBB", "CCC"])
    _seed_session(db, s2, ["AAA", "BBB", "CCC"])
    # s2 already finalized — sweep must skip it
    import duckdb
    con = duckdb.connect(str(db))
    fz.set_session_status(con, s2, finalized=True, basis=fz.BASIS_FINAL)
    con.close()

    history = [{"AAA": 500.0, "BBB": 500.0, "CCC": 500.0} for _ in range(20)]
    flat = FakeFlat({s1: "ok"}, {s1: {"AAA": 50000.0, "BBB": 520.0, "CCC": 480.0}},
                    history, tmp_path)

    results = fz.sweep_unfinalized(before=today, flat=flat)

    assert [r.session for r in results] == [s1]
    assert results[0].status == fz.STATUS_FINAL


# ── Staleness deadline (noon next trading day) ──────────────────────────────────

def test_stale_flags_session_past_noon_next_trading_day(db):
    session = date(2026, 6, 11)   # Thursday
    import duckdb
    con = duckdb.connect(str(db))
    fz.set_session_status(con, session, finalized=False, basis=fz.BASIS_INTRADAY)
    con.close()
    # Friday 1 PM ET — past noon the next trading day
    now = datetime(2026, 6, 12, 13, 0)
    assert fz.stale_unfinalized_sessions(now=now) == [session]


def test_not_stale_before_noon_next_trading_day(db):
    session = date(2026, 6, 11)
    import duckdb
    con = duckdb.connect(str(db))
    fz.set_session_status(con, session, finalized=False, basis=fz.BASIS_INTRADAY)
    con.close()
    now = datetime(2026, 6, 12, 11, 0)   # Friday 11 AM — deadline not reached
    assert fz.stale_unfinalized_sessions(now=now) == []


def test_finalized_session_never_stale(db):
    session = date(2026, 6, 11)
    import duckdb
    con = duckdb.connect(str(db))
    fz.set_session_status(con, session, finalized=True, basis=fz.BASIS_FINAL)
    con.close()
    now = datetime(2026, 6, 15, 13, 0)
    assert fz.stale_unfinalized_sessions(now=now) == []


def test_session_without_status_row_not_stale(db):
    # A session with snapshots but no status row (pre-pipeline / today before the
    # evening scan) is not "stale" — only explicit finalized=FALSE rows count.
    _seed_session(db, date(2026, 6, 5), ["AAA"])
    now = datetime(2026, 6, 12, 13, 0)
    assert fz.stale_unfinalized_sessions(now=now) == []


# ── Report header integration (item 3c/3d) ──────────────────────────────────────

def _load_generate_report():
    import importlib.util
    root = Path(__file__).parent.parent
    spec = importlib.util.spec_from_file_location(
        "generate_report", root / "scripts" / "generate_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_report_basis_suffix_final_vs_intraday(db):
    gr = _load_generate_report()
    session = date(2026, 6, 12)
    import duckdb
    con = duckdb.connect(str(db))
    fz.set_session_status(con, session, finalized=False, basis=fz.BASIS_INTRADAY)
    con.close()
    assert gr._session_basis_suffix(session) == " (intraday — finalizes overnight)"

    con = duckdb.connect(str(db))
    fz.set_session_status(con, session, finalized=True, basis=fz.BASIS_FINAL)
    con.close()
    assert gr._session_basis_suffix(session) == " (final)"


def test_report_stale_finalization_warning(db):
    gr = _load_generate_report()
    session = date(2026, 6, 11)
    import duckdb
    con = duckdb.connect(str(db))
    fz.set_session_status(con, session, finalized=False, basis=fz.BASIS_INTRADAY)
    con.close()
    # No clean way to inject "now" through the report helper, but 2026-06-11 is
    # far in the past relative to the real clock, so its deadline has passed.
    warn = gr._stale_finalization_warning()
    assert warn is not None and "STALLED" in warn and "2026-06-11" in warn

