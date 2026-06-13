# =============================================================================
# scripts/lib/run_lock.sh — per-job flock run lock (source, then call).
#
# WHY (2026-06-12 incident): Windows Task Scheduler and cron both scheduled
# the same jobs after the cron timezone fix, so two evening-scan processes
# started within the same second.  One lost DuckDB lock contention, the
# error was swallowed at debug level, and the regime gate read an empty
# tape (the UNKNOWN incident).  The Task Scheduler duplicates are disabled,
# but the system must stay robust to ANY future duplicate invocation.
#
# Usage:
#     source "$PROJECT_DIR/scripts/lib/run_lock.sh"
#     acquire_run_lock "evening-scan" "$LOG_FILE"
#
# Semantics:
#   - flock -n on /tmp/quantlab-<job>.lock; the fd stays open for the
#     process lifetime (inherited across exec), so the lock is released
#     only when the job exits.
#   - If the lock is already held, logs "DUPLICATE INVOCATION SUPPRESSED"
#     loudly (tee'd to the log file when given) and exits 0 — a suppressed
#     duplicate is not a failure.
# =============================================================================

acquire_run_lock() {
    local job="$1"
    local log_file="${2:-}"
    local lockfile="/tmp/quantlab-${job}.lock"
    local lock_fd

    exec {lock_fd}>"$lockfile"
    if ! flock -n "$lock_fd"; then
        local msg
        msg="[$(date '+%Y-%m-%d %H:%M:%S')] ${job}: DUPLICATE INVOCATION SUPPRESSED — another ${job} process holds ${lockfile}; exiting cleanly"
        if [[ -n "$log_file" ]]; then
            echo "$msg" | tee -a "$log_file"
        else
            echo "$msg"
        fi
        exit 0
    fi
    # Lock acquired — fd intentionally left open for the process lifetime.
}
