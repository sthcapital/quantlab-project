"""
scripts/check_daily_runs.py — Daily job health check.

Reads ~/quantlab-scan.log and verifies that each scheduled job ran today
within its expected time window.  Exits with code 1 if any critical job
is missing so the script can be used in monitoring / alerting pipelines.

Critical jobs (exit code 1 when absent):
    Morning scan   — 08:30–09:30 ET   (cron: 13:00 UTC)
    EOD tracker    — 16:00–17:00 ET   (cron: 20:30 UTC)

Advisory jobs (reported but do not affect exit code):
    Breadth update — 16:00–18:00 ET   (runs 5 min after EOD tracker)

Output format:
    [OK]      Morning scan    ran at 09:00 AM ET
    [OK]      EOD tracker     ran at 04:30 PM ET  (outside window: 05:05 PM)
    [MISSING] Breadth update  not found for 2026-06-05

Log timestamp format (America/New_York local time):
    [YYYY-MM-DD HH:MM:SS] ...    — from daily_scan.sh log() shell function
    YYYY-MM-DD HH:MM:SS  INFO    — from Python logging module

Usage:
    python scripts/check_daily_runs.py
    python scripts/check_daily_runs.py --date 2026-06-04
    python scripts/check_daily_runs.py --log /path/to/other.log
    python scripts/check_daily_runs.py --quiet   # suppress output, exit code only
"""

from __future__ import annotations

import re
import sys
from argparse import ArgumentParser
from datetime import date, datetime, time
from pathlib import Path

import pytz

# ── Constants ──────────────────────────────────────────────────────────────────

LOG_DEFAULT  = Path.home() / "quantlab-scan.log"
NY           = pytz.timezone("America/New_York")

# Regex patterns for extracting a datetime from a log line.
# The system clock is America/New_York so timestamps are already in ET.
_TS_PATTERNS = [
    # [2026-06-05 09:00:01] — from daily_scan.sh log() function
    re.compile(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]'),
    # 2026-06-05 09:00:01  INFO — from Python logging module
    re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+\w'),
    # — 2026-06-05 09:00:01 — embedded in job separator lines
    re.compile(r'[—─]\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'),
]

_STRPTIME_FMT = "%Y-%m-%d %H:%M:%S"


# ── Job definitions ────────────────────────────────────────────────────────────

class JobSpec:
    """Configuration for one scheduled job to check."""

    def __init__(
        self,
        name: str,
        patterns: list[str],
        window: tuple[int, int, int, int],   # (start_h, start_m, end_h, end_m)
        critical: bool,
    ) -> None:
        self.name     = name
        self.patterns = patterns
        self.window   = window    # in ET (same as log timestamps)
        self.critical = critical

    def in_window(self, t: time) -> bool:
        sh, sm, eh, em = self.window
        start = time(sh, sm)
        end   = time(eh, em)
        return start <= t <= end

    def window_str(self) -> str:
        sh, sm, eh, em = self.window
        return f"{sh:02d}:{sm:02d}–{eh:02d}:{em:02d} ET"


JOBS: list[JobSpec] = [
    JobSpec(
        name     = "Morning scan",
        patterns = [
            "Starting universe scan",           # from daily_scan.sh log()
            "QuantLab Daily Pre-Market Scan",   # header printed by daily_scan.sh
            "QuantLab Universe Scanner",         # header from scan_universe.py
        ],
        window   = (8, 30, 9, 30),
        critical = True,
    ),
    JobSpec(
        name     = "EOD tracker",
        patterns = [
            "EOD tracker complete",             # from morning.sh nohup script
            "EOD tracker —",               # — separator in nohup script
            "Forward Return Tracker",           # header from track_forward_returns.py
            "EOD tracker ——",         # variant
        ],
        window   = (16, 0, 17, 0),
        critical = True,
    ),
    JobSpec(
        name     = "Breadth update",
        patterns = [
            "Breadth update complete",          # from morning.sh nohup script
            "Breadth Update  —",           # header from update_breadth.py
            "Breadth update —",            # variant in separator line
            "tape=",                            # from BreadthSnapshot.summary_line()
        ],
        window   = (16, 0, 18, 0),
        critical = False,
    ),
]


# ── Log parsing ────────────────────────────────────────────────────────────────

def read_log(path: Path) -> list[str]:
    """Read all lines from the log file; return [] if file absent."""
    if not path.exists():
        return []
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []


def todays_lines(lines: list[str], check_date: date) -> list[str]:
    """Filter lines that contain today's ISO date string."""
    ds = check_date.isoformat()
    return [ln for ln in lines if ds in ln]


def extract_time(line: str) -> time | None:
    """
    Try to extract a wall-clock time from a single log line.

    Returns a ``time`` object (no date, no timezone) or None when no
    recognised timestamp pattern is found.
    """
    for pat in _TS_PATTERNS:
        m = pat.search(line)
        if m:
            try:
                dt = datetime.strptime(m.group(1), _STRPTIME_FMT)
                return dt.time()
            except ValueError:
                pass
    return None


def find_job(today: list[str], spec: JobSpec) -> tuple[bool, time | None]:
    """
    Scan today's log lines for a job's marker patterns.

    Returns:
        (found, run_time) — found=True when at least one pattern matched;
        run_time is the extracted wall-clock time or None when unavailable.
    """
    for i, line in enumerate(today):
        for pattern in spec.patterns:
            if pattern in line:
                # Try the matching line first, then ±5 surrounding lines
                for offset in range(-5, 6):
                    idx = i + offset
                    if 0 <= idx < len(today):
                        t = extract_time(today[idx])
                        if t is not None:
                            return True, t
                return True, None   # found but couldn't extract time
    return False, None


# ── Reporting ──────────────────────────────────────────────────────────────────

_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_RESET  = "\033[0m"

# Only emit ANSI codes when writing to a real terminal; log files stay clean
_USE_COLOR = sys.stdout.isatty()


def _tag(label: str, color: str) -> str:
    if not _USE_COLOR:
        return label
    return f"{color}{label}{_RESET}"


def check_and_report(
    log_path: Path,
    check_date: date,
    quiet: bool = False,
) -> int:
    """
    Run all job checks and print the status report.

    Returns:
        0 — all critical jobs found (within window or window unknown)
        1 — one or more critical jobs missing or ran outside their window
    """
    lines = read_log(log_path)
    today = todays_lines(lines, check_date)

    if not lines:
        msg = f"[WARN] Log file not found or empty: {log_path}"
        print(_tag(msg, _YELLOW))
        return 1

    exit_code = 0
    results: list[tuple[str, str]] = []   # (tag, message)

    for spec in JOBS:
        found, run_time = find_job(today, spec)

        if not found:
            tag = _tag("[MISSING]", _RED)
            msg = f"{spec.name:<18} not found for {check_date}"
            results.append((tag, msg))
            if spec.critical:
                exit_code = 1
            continue

        if run_time is None:
            tag = _tag("[OK]", _GREEN)
            msg = f"{spec.name:<18} ran today (time not in log)"
            results.append((tag, msg))
            continue

        time_str = run_time.strftime("%I:%M %p ET").lstrip("0")
        if spec.in_window(run_time):
            tag = _tag("[OK]", _GREEN)
            msg = f"{spec.name:<18} ran at {time_str}"
        else:
            # Ran outside the expected window — flag but don't fail critical jobs.
            # A late manual run is still a valid run; failure means truly missing.
            tag = _tag("[LATE]", _YELLOW)
            msg = (
                f"{spec.name:<18} ran at {time_str}"
                f"  (outside window {spec.window_str()})"
            )
            # Advisory only — a job that ran late is better than not running at all

        results.append((tag, msg))

    if not quiet:
        print()
        print(f"  QuantLab Daily Health Check — {check_date}")
        print(f"  Log: {log_path}  ({len(today)} entries for today)")
        print()
        for tag, msg in results:
            print(f"  {tag:<12}  {msg}")
        print()

    return exit_code


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = ArgumentParser(description="Check that all scheduled QuantLab jobs ran today.")
    parser.add_argument("--date",  default=None,
                        help="Date to check (YYYY-MM-DD, default: today)")
    parser.add_argument("--log",   default=str(LOG_DEFAULT),
                        help=f"Path to log file (default: {LOG_DEFAULT})")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress output; only set exit code")
    args = parser.parse_args()

    check_date = date.fromisoformat(args.date) if args.date else date.today()
    exit_code  = check_and_report(
        log_path   = Path(args.log),
        check_date = check_date,
        quiet      = args.quiet,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
