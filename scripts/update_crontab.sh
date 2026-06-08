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
import pytz
s = cron_schedule_for_date(date.today())
print(f"{s['scan_cron']}")
print(f"{s['eod_cron']}")
print(f"{s['tz_name']}")
print(f"{s['utc_offset']}")
print(f"{s['scan_utc']}")
print(f"{s['eod_utc']}")
# Options monitor window: 9:00 AM – 4:30 PM ET in UTC
# utc_offset is e.g. "-0400" (EDT) or "-0500" (EST)
offset_hours = abs(int(s['utc_offset'][:3]))   # 4 for EDT, 5 for EST
opt_start = 9  + offset_hours   # 9 AM ET in UTC = 13 (EDT) or 14 (EST)
opt_end   = 17 + offset_hours   # 5 PM ET in UTC (generous window) = 21 (EDT) or 22 (EST)
print(f"{opt_start}")
print(f"{opt_end}")
PYEOF
)"

SCAN_CRON=$(echo  "$SCHED" | sed -n '1p')
EOD_CRON=$(echo   "$SCHED" | sed -n '2p')
TZ_NAME=$(echo    "$SCHED" | sed -n '3p')
UTC_OFFSET=$(echo "$SCHED" | sed -n '4p')
SCAN_UTC=$(echo   "$SCHED" | sed -n '5p')
EOD_UTC=$(echo    "$SCHED" | sed -n '6p')
OPT_START=$(echo  "$SCHED" | sed -n '7p')
OPT_END=$(echo    "$SCHED" | sed -n '8p')

echo "[$(ts)] update_crontab.sh"
echo "  Timezone : ${TZ_NAME} (UTC${UTC_OFFSET})"
echo "  Scan     : 09:00 AM ET  =  ${SCAN_UTC}   (cron: ${SCAN_CRON} * * 1-5)"
echo "  EOD      : 04:30 PM ET  =  ${EOD_UTC}   (cron: ${EOD_CRON} * * 1-5)"
echo "  Options  : 9:00 AM – 5:00 PM ET every 30 min  (cron: */30 ${OPT_START}-${OPT_END} * * 1-5)"

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

# ── Intraday options monitor (every 30 min, 9:00 AM – 5:00 PM ET, Mon–Fri) ────
# Script self-checks market hours (9:30 AM – 4:00 PM) and exits early if outside.
# Wide UTC window (${OPT_START}–${OPT_END}) handles EDT/EST drift automatically.
*/30 ${OPT_START}-${OPT_END} * * 1-5 /bin/bash -lc 'source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate quantlab && cd ${PROJECT_DIR} && [[ -f .env ]] && set -a && source .env && set +a; python scripts/monitor_options.py' >> ${LOG_FILE} 2>&1

# ── Daily health check (05:15 PM ET, Mon–Fri) ────────────────────────────────
# Runs after all other jobs complete; exits 1 if any critical job is missing.
# EDT: 05:15 PM = 21:15 UTC  |  EST: 05:15 PM = 22:15 UTC
# The UTC time is derived from the EOD offset + 45 minutes.
$(python3 -c "
import pytz; from datetime import datetime
NY=pytz.timezone('America/New_York')
now=datetime.now(NY)
import pytz as _p; utc=_p.UTC
dt=NY.localize(datetime(now.year,now.month,now.day,17,15,0)).astimezone(utc)
print(f'{dt.minute} {dt.hour}')
") * * 1-5 /bin/bash -lc 'source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate quantlab && cd ${PROJECT_DIR} && python scripts/check_daily_runs.py' >> ${LOG_FILE} 2>&1

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
