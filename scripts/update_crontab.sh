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
from quantlab.market_calendar import cron_schedule_for_date, to_utc
from datetime import date, time as _time

dt  = date.today()
s   = cron_schedule_for_date(dt)
off = abs(int(s['utc_offset']))   # 4 (EDT) or 5 (EST)

morning_check = to_utc(_time(8, 45), dt)   # 8:45 AM ET
evening_wday  = to_utc(_time(17, 0), dt)   # 5:00 PM ET Mon-Fri
evening_sun   = to_utc(_time(18, 0), dt)   # 6:00 PM ET Sunday
health        = to_utc(_time(18, 15), dt)  # 6:15 PM ET (after evening scan)

opt_start = 9  + off   # 9:00 AM ET in UTC (13 EDT / 14 EST)
opt_end   = 15 + off   # last run 3:30 PM ET in UTC (19 EDT / 20 EST)

print(s['eod_cron'])                   # 1  EOD tracker
print(s['tz_name'])                    # 2  timezone name
print(s['utc_offset'])                 # 3  UTC offset
print(s['eod_utc'])                    # 4  EOD UTC display
print(morning_check.cron_fields())     # 5  morning check cron fields
print(evening_wday.cron_fields())      # 6  evening scan Mon-Fri cron fields
print(evening_sun.cron_fields())       # 7  evening scan Sunday cron fields
print(health.cron_fields())            # 8  health check cron fields
print(str(morning_check))              # 9  morning UTC display
print(str(evening_wday))              # 10  evening UTC display
print(str(health))                    # 11  health UTC display
print(f"{opt_start}")                 # 12  options monitor start UTC hour
print(f"{opt_end}")                   # 13  options monitor end UTC hour
PYEOF
)"

EOD_CRON=$(echo         "$SCHED" | sed -n '1p')
TZ_NAME=$(echo          "$SCHED" | sed -n '2p')
UTC_OFFSET=$(echo       "$SCHED" | sed -n '3p')
EOD_UTC=$(echo          "$SCHED" | sed -n '4p')
MORNING_CRON=$(echo     "$SCHED" | sed -n '5p')
EVENING_CRON=$(echo     "$SCHED" | sed -n '6p')
EVENING_SUN_CRON=$(echo "$SCHED" | sed -n '7p')
HEALTH_CRON=$(echo      "$SCHED" | sed -n '8p')
MORNING_UTC=$(echo      "$SCHED" | sed -n '9p')
EVENING_UTC=$(echo      "$SCHED" | sed -n '10p')
HEALTH_UTC=$(echo       "$SCHED" | sed -n '11p')
OPT_START=$(echo        "$SCHED" | sed -n '12p')
OPT_END=$(echo          "$SCHED" | sed -n '13p')

echo "[$(ts)] update_crontab.sh"
echo "  Timezone     : ${TZ_NAME} (UTC${UTC_OFFSET})"
echo "  Morning check: 08:45 AM ET  =  ${MORNING_UTC}   (cron: ${MORNING_CRON} * * 1-5)"
echo "  Evening scan : 05:00 PM ET  =  ${EVENING_UTC}   (cron: ${EVENING_CRON} * * 1-5)"
echo "  Evening scan : 06:00 PM ET  =  Sunday only       (cron: ${EVENING_SUN_CRON} * * 0)"
echo "  EOD tracker  : 04:30 PM ET  =  ${EOD_UTC}   (cron: ${EOD_CRON} * * 1-5)"
echo "  Health check : 06:15 PM ET  =  ${HEALTH_UTC}   (cron: ${HEALTH_CRON} * * 1-5)"
echo "  Options      : 9:30 AM – 3:30 PM ET every 30 min  (cron: */30 ${OPT_START}-${OPT_END} * * 1-5)"

# ── Build new crontab content ──────────────────────────────────────────────────
NEW_CRONTAB="# QuantLab automated trading schedule
# Last updated : $(ts) by update_crontab.sh
# Timezone     : ${TZ_NAME} (UTC${UTC_OFFSET})
# Morning UTC  : ${MORNING_UTC}   Evening UTC : ${EVENING_UTC}   EOD UTC : ${EOD_UTC}
# Re-run this script manually after a DST change, or let the lines below do it.

# ── Morning check (08:45 AM ET, Mon–Fri) — lightweight, < 2 min ──────────────
${MORNING_CRON} * * 1-5 /bin/bash -lc 'source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate quantlab && cd ${PROJECT_DIR} && bash scripts/morning.sh' >> ${LOG_FILE} 2>&1

# ── Evening scan (05:00 PM ET, Mon–Fri) — full 2,325-symbol universe scan ─────
${EVENING_CRON} * * 1-5 /bin/bash -lc 'source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate quantlab && cd ${PROJECT_DIR} && bash scripts/evening_scan.sh' >> ${LOG_FILE} 2>&1

# ── Evening scan (06:00 PM ET, Sunday) — weekend data refresh ─────────────────
${EVENING_SUN_CRON} * * 0 /bin/bash -lc 'source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate quantlab && cd ${PROJECT_DIR} && bash scripts/evening_scan.sh' >> ${LOG_FILE} 2>&1

# ── End-of-day forward return tracker (04:30 PM ET, Mon–Fri) ─────────────────
${EOD_CRON} * * 1-5 /bin/bash -lc 'source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate quantlab && cd ${PROJECT_DIR} && python scripts/track_forward_returns.py' >> ${LOG_FILE} 2>&1

# ── Intraday options monitor (every 30 min, 9:30 AM – 4:00 PM ET, Mon–Fri) ────
# Script self-checks market hours (9:30 AM – 4:00 PM) and exits early if outside.
# Wide UTC window (${OPT_START}–${OPT_END}) handles EDT/EST drift automatically.
*/30 ${OPT_START}-${OPT_END} * * 1-5 /bin/bash -lc 'source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate quantlab && cd ${PROJECT_DIR} && [[ -f .env ]] && set -a && source .env && set +a; python scripts/monitor_options.py' >> ${LOG_FILE} 2>&1

# ── Daily health check (06:15 PM ET, Mon–Fri) ────────────────────────────────
# Runs after evening scan completes; exits 1 if any critical job is missing.
# EDT: 06:15 PM = 22:15 UTC  |  EST: 06:15 PM = 23:15 UTC
${HEALTH_CRON} * * 1-5 /bin/bash -lc 'source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate quantlab && cd ${PROJECT_DIR} && python scripts/check_daily_runs.py' >> ${LOG_FILE} 2>&1

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
