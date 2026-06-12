"""
Regression tests for the options signal pipeline.

Covers the 2026-06-11 failure where the intraday monitor flagged 291 symbols
but the report rendered zero options signals:

  1. InstitutionalWatchlist.upsert() must not clobber an options_signal set
     earlier the same day by the intraday monitor (set_options_signal).
  2. The report must emit a WARNING when zero entries carry an options signal
     while options_snapshots has rows for that session.
  3. The daily health check must assert the options monitor produced a
     heartbeat (options_snapshots row count) for the session.
"""

from __future__ import annotations

import importlib.util
from datetime import date, datetime, timedelta
from pathlib import Path

import duckdb
import pytest

from quantlab.execution import ScanResult

_ROOT = Path(__file__).parent.parent


def _load_script(name: str):
    """Import a module from scripts/ (not a package) by file path."""
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_result(symbol="TEST", conviction=0.55, stage=2, entry_close=100.0,
                 unusual_options_score=0.0, options_score=0.0):
    r = ScanResult(
        symbol=symbol, scan_date=date.today().isoformat(),
        signal_type="breakout", signal=True,
        entry_close=entry_close, indicator_value=None, lookback=5,
        conviction_score=conviction, stage=stage,
    )
    r.unusual_options_score = unusual_options_score
    r.options_score = options_score
    return r


_SNAPSHOT_DDL = """
    CREATE TABLE IF NOT EXISTS options_snapshots (
        symbol        VARCHAR NOT NULL,
        snap_date     DATE NOT NULL,
        spot_price    DOUBLE,
        pcr           DOUBLE,
        iv_skew       DOUBLE,
        unusual_calls BOOLEAN,
        options_score DOUBLE,
        call_count    INTEGER,
        put_count     INTEGER,
        PRIMARY KEY (symbol, snap_date)
    )
"""


def _insert_snapshots(db_path, snap_date: date, symbols: list[str],
                      unusual: bool = True, score: float = 0.85) -> None:
    con = duckdb.connect(str(db_path))
    con.execute(_SNAPSHOT_DDL)
    for sym in symbols:
        con.execute(
            "INSERT OR REPLACE INTO options_snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [sym, snap_date, 100.0, 0.5, 0.0, unusual, score, 10, 5],
        )
    con.close()


# ══════════════════════════════════════════════════════════════════════════════
# 1. upsert() must preserve same-day intraday options_signal
# ══════════════════════════════════════════════════════════════════════════════

class TestUpsertPreservesIntradaySignal:

    def test_evening_upsert_keeps_flag_set_by_monitor_today(self, tmp_path):
        """Monitor sets the flag intraday; the evening re-upsert (which finds no
        unusual activity in flat files) must not erase it before the report."""
        from quantlab.watchlist import InstitutionalWatchlist
        db = tmp_path / "test.duckdb"
        iwl = InstitutionalWatchlist(db_path=db)

        iwl.upsert("CELH", _make_result("CELH"))          # morning: no signal yet
        iwl.set_options_signal("CELH", bonus=0.08)        # intraday monitor flags it
        iwl.upsert("CELH", _make_result("CELH"))          # evening scan: scores all 0.0

        entry = next(e for e in iwl.get_candidates() if e["symbol"] == "CELH")
        assert entry["options_signal"] is True

    def test_evening_upsert_reapplies_monitor_conviction_bonus(self, tmp_path):
        from quantlab.watchlist import InstitutionalWatchlist
        db = tmp_path / "test.duckdb"
        iwl = InstitutionalWatchlist(db_path=db)

        iwl.upsert("PATH", _make_result("PATH", conviction=0.50))
        iwl.set_options_signal("PATH", bonus=0.08)
        iwl.upsert("PATH", _make_result("PATH", conviction=0.50))

        entry = next(e for e in iwl.get_candidates() if e["symbol"] == "PATH")
        # day-1 bonus 0.05 + preserved monitor bonus 0.08
        assert entry["conviction_score"] == pytest.approx(0.50 + 0.05 + 0.08, abs=1e-4)

    def test_stale_flag_from_prior_day_is_recomputed(self, tmp_path):
        """A flag whose last update was yesterday is scan-time state, not an
        intraday detection — the upsert recomputes it normally."""
        from quantlab.watchlist import InstitutionalWatchlist
        db = tmp_path / "test.duckdb"
        iwl = InstitutionalWatchlist(db_path=db)

        yesterday = date.today() - timedelta(days=1)
        con = duckdb.connect(str(db))
        con.execute(
            """
            INSERT INTO institutional_watchlist
                (symbol, first_seen, last_seen, consecutive_days, stage,
                 conviction_score, entry_price, options_signal, volume_dry_up,
                 earnings_score, peg_score, breakout_volume_score, tape, notes,
                 updated_at)
            VALUES (?, ?, ?, 1, 2, 0.60, 100.0, TRUE, FALSE,
                    NULL, NULL, NULL, '', '', ?)
            """,
            ["BROS", yesterday.isoformat(), yesterday.isoformat(),
             datetime.combine(yesterday, datetime.min.time())],
        )
        con.close()

        iwl.upsert("BROS", _make_result("BROS"))   # today's scan: no signal
        entry = next(e for e in iwl.get_candidates() if e["symbol"] == "BROS")
        assert entry["options_signal"] is False

    def test_scan_scores_still_set_flag(self, tmp_path):
        from quantlab.watchlist import InstitutionalWatchlist
        iwl = InstitutionalWatchlist(db_path=tmp_path / "test.duckdb")
        iwl.upsert("MRX", _make_result("MRX", unusual_options_score=0.6))
        entry = next(e for e in iwl.get_candidates() if e["symbol"] == "MRX")
        assert entry["options_signal"] is True


# ══════════════════════════════════════════════════════════════════════════════
# 2. Report WARNING when snapshots exist but zero signals render
# ══════════════════════════════════════════════════════════════════════════════

class TestReportOptionsWarning:

    def test_warning_when_snapshots_exist_but_no_flags(self, tmp_path):
        """The exact 2026-06-11 failure: monitor wrote snapshot rows for the
        session but every report row shows '–' — must produce a WARNING that
        names a pipeline fault."""
        gr = _load_script("generate_report")
        db = tmp_path / "test.duckdb"
        snap_date = date.today()
        _insert_snapshots(db, snap_date, ["CELH", "PATH", "MRX"])

        candidates = [
            {"symbol": "CELH", "options_signal": False},
            {"symbol": "PATH", "options_signal": False},
        ]
        warning = gr._options_pipeline_warning(candidates, [], snap_date, str(db))
        assert warning is not None
        assert "pipeline fault" in warning
        assert "3 snapshots" in warning

    def test_warning_when_monitor_never_ran(self, tmp_path):
        gr = _load_script("generate_report")
        db = tmp_path / "test.duckdb"
        duckdb.connect(str(db)).close()   # empty DB — no snapshots table

        warning = gr._options_pipeline_warning(
            [{"symbol": "CELH", "options_signal": False}], [], date.today(), str(db),
        )
        assert warning is not None
        assert "monitor may not have run" in warning

    def test_no_warning_when_any_signal_present(self, tmp_path):
        gr = _load_script("generate_report")
        db = tmp_path / "test.duckdb"
        _insert_snapshots(db, date.today(), ["CELH"])

        candidates = [
            {"symbol": "CELH", "options_signal": True},
            {"symbol": "PATH", "options_signal": False},
        ]
        assert gr._options_pipeline_warning(candidates, [], date.today(), str(db)) is None

    def test_basing_watchlist_signal_counts(self, tmp_path):
        gr = _load_script("generate_report")
        db = tmp_path / "test.duckdb"
        _insert_snapshots(db, date.today(), ["CELH"])

        basing = [{"symbol": "CELH", "options_signal": True}]
        assert gr._options_pipeline_warning([], basing, date.today(), str(db)) is None


# ══════════════════════════════════════════════════════════════════════════════
# 3. Health check asserts an options monitor heartbeat
# ══════════════════════════════════════════════════════════════════════════════

class TestOptionsHeartbeat:

    def test_heartbeat_counts_session_rows(self, tmp_path):
        cdr = _load_script("check_daily_runs")
        db = tmp_path / "test.duckdb"
        snap_date = date.today()
        _insert_snapshots(db, snap_date, ["CELH", "PATH"])
        assert cdr.check_options_heartbeat(snap_date, db) == 2

    def test_heartbeat_zero_for_missing_session(self, tmp_path):
        cdr = _load_script("check_daily_runs")
        db = tmp_path / "test.duckdb"
        _insert_snapshots(db, date.today() - timedelta(days=1), ["CELH"])
        assert cdr.check_options_heartbeat(date.today(), db) == 0

    def test_heartbeat_zero_when_table_missing(self, tmp_path):
        cdr = _load_script("check_daily_runs")
        db = tmp_path / "test.duckdb"
        duckdb.connect(str(db)).close()
        assert cdr.check_options_heartbeat(date.today(), db) == 0

    def test_health_check_fails_on_zero_heartbeat(self, tmp_path, capsys):
        """End-to-end: log shows the monitor cron fired, but no snapshot rows
        were written for the session → exit code 1 on a trading day."""
        cdr = _load_script("check_daily_runs")
        check_date = date(2026, 6, 11)   # Thursday — NYSE open
        db = tmp_path / "test.duckdb"
        duckdb.connect(str(db)).close()

        log = tmp_path / "scan.log"
        log.write_text(
            f"[{check_date} 08:45:01] QuantLab Morning Check\n"
            f"[{check_date} 13:00:08] monitor_options: checking 357 watchlist symbols ...\n"
            f"[{check_date} 16:30:00] Forward Return Tracker\n"
            f"[{check_date} 17:00:05] Starting universe scan\n"
            f"[{check_date} 17:10:00] tape=BULL\n"
        )
        exit_code = cdr.check_and_report(log, check_date, quiet=True, db_path=db)
        assert exit_code == 1

    def test_health_check_passes_with_heartbeat(self, tmp_path):
        cdr = _load_script("check_daily_runs")
        check_date = date(2026, 6, 11)
        db = tmp_path / "test.duckdb"
        _insert_snapshots(db, check_date, ["CELH", "PATH"])

        log = tmp_path / "scan.log"
        log.write_text(
            f"[{check_date} 08:45:01] QuantLab Morning Check\n"
            f"[{check_date} 13:00:08] monitor_options: checking 357 watchlist symbols ...\n"
            f"[{check_date} 16:30:00] Forward Return Tracker\n"
            f"[{check_date} 17:00:05] Starting universe scan\n"
            f"[{check_date} 17:10:00] tape=BULL\n"
        )
        exit_code = cdr.check_and_report(log, check_date, quiet=True, db_path=db)
        assert exit_code == 0

    def test_health_check_fails_when_monitor_log_missing(self, tmp_path):
        """Snapshot rows exist but the monitor never logged a run today —
        still a failure (partial/foreign writes shouldn't mask a dead cron)."""
        cdr = _load_script("check_daily_runs")
        check_date = date(2026, 6, 11)
        db = tmp_path / "test.duckdb"
        _insert_snapshots(db, check_date, ["CELH"])

        log = tmp_path / "scan.log"
        log.write_text(
            f"[{check_date} 08:45:01] QuantLab Morning Check\n"
            f"[{check_date} 16:30:00] Forward Return Tracker\n"
            f"[{check_date} 17:00:05] Starting universe scan\n"
            f"[{check_date} 17:10:00] tape=BULL\n"
        )
        exit_code = cdr.check_and_report(log, check_date, quiet=True, db_path=db)
        assert exit_code == 1
