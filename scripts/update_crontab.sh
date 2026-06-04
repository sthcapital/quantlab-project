#!/usr/bin/env bash
# =============================================================================
# scripts/update_crontab.sh — Rewrite the QuantLab crontab with DST-correct
# UTC scan times for the America/New_York timezone.
#
# Run this manually whenever clocks change, or let cron call it automatically
# on the DST transition dates (wired into the crontab this script produces).
#
# The script self-embeds the DST auto-update triggers so they survive every
# rewrite:
#   Spring forward (2nd Sunday of March)  : 3:00 AM EDT = 07:00 UTC
#   Fall back     (1st Sunday of November): 3:00 AM EST = 08:00 UTC
#
# Usage:
#   bash scripts/update_crontab.sh          # update for today's offset
#   bash scripts/update_crontab.sh --dry-run # print without installing
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
LOG_FILE="$HOME/quantlab-scan.log"
DRY_RUN=""
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN="true"

# ── Activate conda so market_calendar is importable ───────────────────────────
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate quantlab
cd "$PROJECT_DIR"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

# ── Query current DST offset via Python ───────────────────────────────────────
SCHED="$(python3 - <<'PYEOF'
import sys
sys.path.insert(0, "src")
from quantlab.market_calendar import cron_schedule_for_date
from datetime import date
s = cron_schedule_for_date(date.today())
print(f"{s['scan_cron']}")
print(f"{s['eod_cron']}")
print(f"{s['tz_name']}")
print(f"{s['utc_offset']}")
print(f"{s['scan_utc']}")
print(f"{s['eod_utc']}")
PYEOF
)"

SCAN_CRON=$(echo "$SCHED" | sed -n '1p')
EOD_CRON=$(echo  "$SCHED" | sed -n '2p')
TZ_NAME=$(echo   "$SCHED" | sed -n '3p')
UTC_OFFSET=$(echo "$SCHED" | sed -n '4p')
SCAN_UTC=$(echo  "$SCHED" | sed -n '5p')
EOD_UTC=$(echo   "$SCHED" | sed -n '6p')

echo "[$(ts)] update_crontab.sh"
echo "  Timezone : ${TZ_NAME} (UTC${UTC_OFFSET})"
echo "  Scan     : 09:00 AM ET  =  ${SCAN_UTC}   (cron: ${SCAN_CRON} * * 1-5)"
echo "  EOD      : 04:30 PM ET  =  ${EOD_UTC}   (cron: ${EOD_CRON} * * 1-5)"

# ── Build new crontab content ──────────────────────────────────────────────────
NEW_CRONTAB="# QuantLab automated trading schedule
# Last updated : $(ts) by update_crontab.sh
# Timezone     : ${TZ_NAME} (UTC${UTC_OFFSET})
# Scan UTC     : ${SCAN_UTC}   EOD UTC : ${EOD_UTC}
# Re-run this script manually after a DST change, or let the lines below do it.

# ── Morning pre-market scan (09:00 AM ET, Mon–Fri) ────────────────────────────
${SCAN_CRON} * * 1-5 /bin/bash -lc 'source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate quantlab && cd ${PROJECT_DIR} && bash scripts/daily_scan.sh --with-news' >> ${LOG_FILE} 2>&1

# ── End-of-day forward return tracker (04:30 PM ET, Mon–Fri) ─────────────────
${EOD_CRON} * * 1-5 /bin/bash -lc 'source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate quantlab && cd ${PROJECT_DIR} && python scripts/track_forward_returns.py' >> ${LOG_FILE} 2>&1

# ── DST auto-update ────────────────────────────────────────────────────────────
# Spring forward: 2nd Sunday of March (day 8–14), 3:00 AM EDT = 07:00 UTC
0 7 8-14 3 0 /bin/bash -lc 'source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate quantlab && cd ${PROJECT_DIR} && bash scripts/update_crontab.sh' >> ${LOG_FILE} 2>&1

# Fall back: 1st Sunday of November (day 1–7), 3:00 AM EST = 08:00 UTC
0 8 1-7 11 0 /bin/bash -lc 'source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate quantlab && cd ${PROJECT_DIR} && bash scripts/update_crontab.sh' >> ${LOG_FILE} 2>&1
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
