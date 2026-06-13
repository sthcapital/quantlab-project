"""
scripts/lib/run_lock.sh — flock-based per-job run lock.

2026-06-12 incident: duplicate scheduler entries started two evening scans in
the same second; the loser's DuckDB lock error was swallowed and the regime
gate read an empty tape.  The lock makes a duplicate invocation exit 0 with a
loud suppression message instead of racing.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

HELPER = Path(__file__).parent.parent / "scripts" / "lib" / "run_lock.sh"


def _holder(job: str, hold_seconds: float) -> subprocess.Popen:
    """Spawn a bash process that acquires the run lock and sleeps."""
    proc = subprocess.Popen(
        [
            "bash", "-c",
            f'source "{HELPER}" && acquire_run_lock "{job}" '
            f'&& echo ACQUIRED && sleep {hold_seconds}',
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    line = proc.stdout.readline().strip()
    assert line == "ACQUIRED", f"lock holder failed to start: {line!r}"
    return proc


def _invoke(job: str, log_file: str = "") -> subprocess.CompletedProcess:
    log_arg = f' "{log_file}"' if log_file else ""
    return subprocess.run(
        [
            "bash", "-c",
            f'source "{HELPER}" && acquire_run_lock "{job}"{log_arg} '
            f'&& echo JOB_BODY_RAN',
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )


class TestRunLock:
    def test_duplicate_invocation_suppressed(self):
        job = f"pytest-dup-{os.getpid()}"
        holder = _holder(job, hold_seconds=20)
        try:
            second = _invoke(job)
            # Exits 0 (suppression is not a failure), logs loudly, and the
            # job body never runs
            assert second.returncode == 0
            assert "DUPLICATE INVOCATION SUPPRESSED" in second.stdout
            assert job in second.stdout
            assert "JOB_BODY_RAN" not in second.stdout
        finally:
            holder.kill()
            holder.wait()

    def test_suppression_message_tees_to_log_file(self, tmp_path):
        job = f"pytest-log-{os.getpid()}"
        log = tmp_path / "scan.log"
        holder = _holder(job, hold_seconds=20)
        try:
            second = _invoke(job, log_file=str(log))
            assert second.returncode == 0
            assert "DUPLICATE INVOCATION SUPPRESSED" in log.read_text()
        finally:
            holder.kill()
            holder.wait()

    def test_lock_released_on_exit(self):
        job = f"pytest-rel-{os.getpid()}"
        holder = _holder(job, hold_seconds=0.2)
        holder.wait()
        time.sleep(0.1)
        third = _invoke(job)
        assert third.returncode == 0
        assert "JOB_BODY_RAN" in third.stdout
        assert "SUPPRESSED" not in third.stdout

    def test_different_jobs_do_not_collide(self):
        job_a = f"pytest-a-{os.getpid()}"
        job_b = f"pytest-b-{os.getpid()}"
        holder = _holder(job_a, hold_seconds=20)
        try:
            other = _invoke(job_b)
            assert other.returncode == 0
            assert "JOB_BODY_RAN" in other.stdout
        finally:
            holder.kill()
            holder.wait()
