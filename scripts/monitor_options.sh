#!/usr/bin/env bash
# =============================================================================
# scripts/monitor_options.sh — run-locked wrapper for monitor_options.py.
#
# Cron invokes this instead of the Python script directly so a duplicate
# scheduler entry (the 2026-06-12 Task Scheduler + cron double-fire) can
# never run two monitors concurrently: the second invocation logs
# "DUPLICATE INVOCATION SUPPRESSED" and exits 0.
#
# Arguments are passed through to monitor_options.py (--force, --dry-run).
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${LOG_FILE:-$HOME/quantlab-scan.log}"

# shellcheck source=lib/run_lock.sh
source "$PROJECT_DIR/scripts/lib/run_lock.sh"
acquire_run_lock "monitor-options" "$LOG_FILE"

# Lock fd is inherited across exec — held until the monitor exits.
exec python "$PROJECT_DIR/scripts/monitor_options.py" "$@"
