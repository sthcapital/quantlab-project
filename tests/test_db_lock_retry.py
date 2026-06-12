"""
connect_with_retry — lock-contention backoff for CLI scripts vs the intraday
monitor's short write cycles.

The contention tests hold the DuckDB file lock from a REAL subprocess: DuckDB
shares the database instance within one process, so only a second process
produces the "Could not set lock" IOException the helper retries on.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import duckdb
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quantlab.storage import connect_with_retry


def _hold_lock(db: Path, hold_seconds: float) -> subprocess.Popen:
    """Spawn a subprocess that opens ``db`` read-write and sleeps.

    Blocks until the child confirms it holds the lock (prints 'held').
    """
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import duckdb, sys, time\n"
                f"con = duckdb.connect({str(db)!r})\n"
                "print('held', flush=True)\n"
                f"time.sleep({hold_seconds})\n"
                "con.close()\n"
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert child.stdout is not None
    line = child.stdout.readline().strip()
    assert line == "held", f"lock-holder subprocess failed: {line!r}"
    return child


class TestConnectWithRetry:
    def test_plain_connect_no_contention(self, tmp_path):
        db = tmp_path / "free.duckdb"
        con = connect_with_retry(db)
        con.execute("CREATE TABLE t (x INTEGER)")
        con.close()

    def test_succeeds_after_lock_released(self, tmp_path):
        db = tmp_path / "contended.duckdb"
        duckdb.connect(str(db)).close()  # create the file first

        child = _hold_lock(db, hold_seconds=1.5)
        try:
            # Backoff schedule 0.5s, 1.0s, 2.0s, 4.0s — the lock frees at
            # ~1.5s, so a retry within the first few attempts must win.
            con = connect_with_retry(db, attempts=5, base_delay=0.5)
            con.execute("SELECT 1").fetchone()
            con.close()
        finally:
            child.kill()
            child.wait()

    def test_raises_after_attempts_exhausted(self, tmp_path):
        db = tmp_path / "stuck.duckdb"
        duckdb.connect(str(db)).close()

        child = _hold_lock(db, hold_seconds=30)
        try:
            with pytest.raises(duckdb.IOException, match="lock"):
                connect_with_retry(db, attempts=2, base_delay=0.1)
        finally:
            child.kill()
            child.wait()

    def test_non_lock_error_raises_immediately(self, tmp_path):
        bogus = tmp_path / "no_such_dir" / "missing.duckdb"
        start = time.monotonic()
        with pytest.raises(duckdb.IOException):
            connect_with_retry(bogus, attempts=5, base_delay=2.0)
        # No retries: a missing-directory error must not burn the backoff
        assert time.monotonic() - start < 1.0

    def test_read_only_passthrough(self, tmp_path):
        db = tmp_path / "ro.duckdb"
        con = duckdb.connect(str(db))
        con.execute("CREATE TABLE t AS SELECT 42 AS x")
        con.close()

        ro = connect_with_retry(db, read_only=True)
        assert ro.execute("SELECT x FROM t").fetchone()[0] == 42
        with pytest.raises(duckdb.Error):
            ro.execute("INSERT INTO t VALUES (43)")
        ro.close()
