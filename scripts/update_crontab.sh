#!/usr/bin/env bash
# =============================================================================
# scripts/update_crontab.sh — Rewrite the QuantLab crontab with Eastern-time
# schedule fields for a host whose cron runs in local America/New_York time.
#
# HISTORY (2026-06-12 incident): this script previously emitted UTC cron
# fields on the assumption that the system clock was UTC.  The host clock is
# actually Eastern (America/Toronto), and cron evaluates crontab fields in
# LOCAL time — so every job fired 4 hours late (morning check 12:45 PM ET,
# evening scan 9:00 PM ET).  The health check's [LATE] tags were truthful.
# Scheduling directly in local Eastern time fixes the offset and makes the
# old DST auto-update machinery unnecessary: a local-time cron follows DST
# transitions automatically.
#
# Usage:
#   bash scripts/update_crontab.sh           # install ET-local schedule
#   bash scripts/update_crontab.sh --dry-run # print without installing
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
LOG_FILE="$HOME/quantlab-scan.log"
DRY_RUN=""
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN="true"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

# ── Guard: the schedule below is only correct on an Eastern-time host ─────────
# Eastern is UTC-4 (EDT) or UTC-5 (EST).  Anything else means the host TZ was
# changed; refuse to install a silently-shifted schedule.
UTC_OFFSET="$(date +%z)"
TZ_ABBREV="$(date +%Z)"
if [[ "$UTC_OFFSET" != "-0400" && "$UTC_OFFSET" != "-0500" ]]; then
    echo "[$(ts)] ERROR: host UTC offset is ${UTC_OFFSET} (${TZ_ABBREV})," >&2
    echo "        not Eastern (-0400/-0500). The ET-local schedule below"   >&2
    echo "        would fire at the wrong wall-clock times. Fix the host"   >&2
    echo "        timezone (or adapt this script) before installing."       >&2
    exit 1
fi

echo "[$(ts)] update_crontab.sh"
echo "  Host clock   : ${TZ_ABBREV} (UTC${UTC_OFFSET}) — cron fields are LOCAL Eastern time"
echo "  Morning check: 08:45 AM ET  (cron: 45 8 * * 1-5)"
echo "  Evening scan : 05:00 PM ET  (cron: 0 17 * * 1-5)"
echo "  Evening scan : 06:00 PM ET  Sunday only (cron: 0 18 * * 0)"
echo "  EOD tracker  : 04:30 PM ET  (cron: 30 16 * * 1-5)"
echo "  Health check : 06:15 PM ET  (cron: 15 18 * * 1-5)"
echo "  Options      : 9:30 AM – 3:30 PM ET every 30 min (cron: */30 9-15 * * 1-5)"

# ── Build new crontab content ──────────────────────────────────────────────────
NEW_CRONTAB="# QuantLab automated trading schedule
# Last updated : $(ts) by update_crontab.sh
# Timezone     : LOCAL Eastern time (host: ${TZ_ABBREV}, UTC${UTC_OFFSET})
# All fields below are Eastern wall-clock times — cron evaluates the crontab
# in the host's local timezone, which is verified to be Eastern at install
# time.  DST needs no special handling: the local clock shifts with it.

# ── Morning check (08:45 AM ET, Mon–Fri) — lightweight, < 2 min ──────────────
45 8 * * 1-5 /bin/bash -lc 'source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate quantlab && cd ${PROJECT_DIR} && bash scripts/morning.sh' >> ${LOG_FILE} 2>&1

# ── Evening scan (05:00 PM ET, Mon–Fri) — full tradeable-universe scan ────────
# (Universe size floats daily with the build — ~1,000–2,300 symbols; the gate
# in quantlab/universe.py refuses degenerate builds.)
0 17 * * 1-5 /bin/bash -lc 'source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate quantlab && cd ${PROJECT_DIR} && bash scripts/evening_scan.sh' >> ${LOG_FILE} 2>&1

# ── Evening scan (06:00 PM ET, Sunday) — weekend data refresh ─────────────────
0 18 * * 0 /bin/bash -lc 'source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate quantlab && cd ${PROJECT_DIR} && bash scripts/evening_scan.sh' >> ${LOG_FILE} 2>&1

# ── End-of-day forward return tracker (04:30 PM ET, Mon–Fri) ─────────────────
30 16 * * 1-5 /bin/bash -lc 'source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate quantlab && cd ${PROJECT_DIR} && python scripts/track_forward_returns.py' >> ${LOG_FILE} 2>&1

# ── Intraday options monitor (every 30 min, 9:30 AM – 4:00 PM ET, Mon–Fri) ────
# Script self-checks market hours (9:30 AM – 4:00 PM) and exits early if
# outside — the 9:00 AM fire is skipped by that guard.
# Invoked via the run-locked wrapper: a duplicate scheduler entry logs
# "DUPLICATE INVOCATION SUPPRESSED" and exits 0 instead of racing the lock.
*/30 9-15 * * 1-5 /bin/bash -lc 'source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate quantlab && cd ${PROJECT_DIR} && [[ -f .env ]] && set -a && source .env && set +a; bash scripts/monitor_options.sh' >> ${LOG_FILE} 2>&1

# ── Daily health check (06:15 PM ET, Mon–Fri) ────────────────────────────────
# Runs after evening scan completes; exits 1 if any critical job is missing.
15 18 * * 1-5 /bin/bash -lc 'source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate quantlab && cd ${PROJECT_DIR} && python scripts/check_daily_runs.py' >> ${LOG_FILE} 2>&1
"

if [[ -n "$DRY_RUN" ]]; then
    echo ""
    echo "── DRY RUN — crontab that would be installed: ──────────────"
    echo "$NEW_CRONTAB"
    echo "────────────────────────────────────────────────────────────"
    echo "[$(ts)] Dry run complete — crontab NOT changed."
else
    echo "$NEW_CRONTAB" | crontab -
    echo "[$(ts)] Crontab updated successfully."
fi
