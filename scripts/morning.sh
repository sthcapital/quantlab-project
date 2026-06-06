#!/usr/bin/env bash
# =============================================================================
# scripts/morning.sh — Self-scheduling morning routine.
#
# A single launch covers the full trading day:
#
#   IMMEDIATE  Step 1: update_breadth.py --no-polygon  (load tape from DuckDB)
#   IMMEDIATE  Step 2: track_forward_returns.py        (catch-up closes)
#   IMMEDIATE  Step 3: daily_scan.sh --with-news       (morning scan)
#   12:30 ET   Step 4: scan_universe.py --no-news      (midday check)
#   16:30 ET   Step 5: track_forward_returns.py        (record closes)
#              + 5m    update_breadth.py               (EOD breadth)
#
# Background jobs (Steps 4–5) survive terminal close via nohup + disown.
# Each job is written to a self-deleting temp script and launched with:
#   nohup bash /tmp/ql_XXXXX.sh >> LOG 2>&1 &
#   disown $!
# This ensures: (a) SIGHUP immunity (terminal hang-up is ignored), and
# (b) all config variables are expanded at schedule time so the job runs
# with the same paths/ports regardless of environment drift.
#
# Breadth MUST run before the scan so the tape condition is consulted during
# conviction scoring and the min_conviction threshold is set correctly.
#
# Usage:
#   bash scripts/morning.sh             # with news (default)
#   bash scripts/morning.sh --no-news   # faster, price-only scan
# =============================================================================

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
CONDA_ENV="quantlab"
LOG_FILE="$HOME/quantlab-scan.log"
IBKR_HOST="172.23.208.1"
IBKR_PORT="7497"

WITH_NEWS="--with-news"
[[ "${1:-}" == "--no-news" ]] && WITH_NEWS=""

# ── Activate environment ───────────────────────────────────────────────────────
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$PROJECT_DIR"

sep() { printf '%.0s═' {1..62}; printf '\n'; }
ts()  { date '+%Y-%m-%d %H:%M:%S'; }
et()  {
    python3 -c "
import pytz; from datetime import datetime
print(datetime.now(pytz.timezone('America/New_York')).strftime('%H:%M'))
"
}

echo ""
sep
echo "  QuantLab Morning Routine  $(ts)"
echo "  news: ${WITH_NEWS:-disabled}"
sep

# ── Compute schedule ──────────────────────────────────────────────────────────
_SCHED=$(python3 - <<'PYEOF'
import pytz; from datetime import datetime
NY  = pytz.timezone("America/New_York")
now = datetime.now(NY)
def epoch(h, m):
    return int(NY.localize(datetime(now.year,now.month,now.day,h,m,0)).timestamp())
def fmt(s):
    if s<=0: return "PAST"
    h,r=divmod(s,3600); m=r//60
    return f"{h}h {m}m" if h else f"{m}m"
ne=int(now.timestamp()); me=epoch(12,30); ee=epoch(16,30)
sm=max(0,me-ne); se=max(0,ee-ne)
print(sm); print(se); print(fmt(sm)); print(fmt(se))
PYEOF
)
SLEEP_MIDDAY=$(echo "$_SCHED" | sed -n '1p')
SLEEP_EOD=$(echo    "$_SCHED" | sed -n '2p')
DUR_MIDDAY=$(echo   "$_SCHED" | sed -n '3p')
DUR_EOD=$(echo      "$_SCHED" | sed -n '4p')

# ── Step 1: Load breadth tape + fetch macro context ───────────────────────────
echo ""; echo "── [$(et)] Step 1: Breadth tape + macro context ─────────────"
python scripts/update_breadth.py --no-polygon 2>/dev/null || true
python scripts/fetch_macro_context.py 2>/dev/null || true

# ── Step 2: Forward return catch-up ───────────────────────────────────────────
echo ""; echo "── [$(et)] Step 2: Forward return catch-up ──────────────────"
python scripts/track_forward_returns.py --no-ibkr 2>/dev/null || true

# ── Step 3: Morning scan ───────────────────────────────────────────────────────
echo ""; echo "── [$(et)] Step 3: Morning scan ─────────────────────────────"
if [[ -n "$WITH_NEWS" ]]; then
    bash scripts/daily_scan.sh --with-news
else
    bash scripts/daily_scan.sh
fi
MORNING_ET="$(et)"

# ── Watchlist additions summary ────────────────────────────────────────────────
_WL_MSG=$(python3 - <<'PYEOF'
from quantlab.watchlist import get_active_watchlist
from datetime import date
today = date.today().isoformat()
entries = [e for e in get_active_watchlist() if str(e["date_added"]) == today]
if entries:
    parts = [f"{e['symbol']} {e['conviction_score']:.2f}" for e in entries]
    print(" + ".join(parts) + " added to watchlist")
else:
    print("no new entries (all below 0.70)")
PYEOF
)

# ── Schedule summary ───────────────────────────────────────────────────────────
echo ""; sep
echo "  [$MORNING_ET] Morning scan complete — $_WL_MSG"
if [[ "$SLEEP_MIDDAY" -gt 0 ]]; then
    echo "  [12:30] Midday check scheduled — runs in $DUR_MIDDAY"
else
    echo "  [12:30] Midday check SKIPPED — already past"
fi
if [[ "$SLEEP_EOD" -gt 0 ]]; then
    echo "  [16:30] EOD tracker scheduled  — runs in $DUR_EOD"
else
    echo "  [16:30] EOD tracker SKIPPED    — already past 16:30 ET"
fi
sep

# ── Step 4: Watchlist dashboard ───────────────────────────────────────────────
echo ""; echo "── [$(et)] Step 4: Watchlist dashboard ──────────────────────"
python scripts/watchlist_status.py --no-ibkr || true
echo ""

# ── Background helper: write a self-deleting temp script + launch via nohup ──
# nohup makes the process immune to SIGHUP (terminal hang-up).
# disown removes it from the shell job table so bash won't signal it on exit.
# Variables are expanded NOW so the job carries its own config snapshot.
_schedule() {
    local label="$1"
    local tmpscript
    tmpscript=$(mktemp /tmp/ql_XXXXXX.sh)
    # Body is passed on stdin
    cat > "$tmpscript"
    chmod +x "$tmpscript"
    nohup bash "$tmpscript" >> "$LOG_FILE" 2>&1 < /dev/null &
    local pid=$!
    disown "$pid"
    echo "  Scheduled $label (PID $pid) — temp script: $tmpscript"
}

# ── Background: Midday check at 12:30 PM ET ───────────────────────────────────
if [[ "$SLEEP_MIDDAY" -gt 0 ]]; then
    _schedule "[12:30] midday check" << EOJOB
#!/usr/bin/env bash
# QuantLab midday check — generated by morning.sh at $(ts)
sleep ${SLEEP_MIDDAY}
source "${CONDA_BASE}/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate "${CONDA_ENV}" 2>/dev/null || true
cd "${PROJECT_DIR}"
{
  echo ""
  echo "══ [12:30] Midday check — \$(date '+%Y-%m-%d %H:%M:%S') ══════════════"
} >> "${LOG_FILE}"
python scripts/scan_universe.py \\
    --universe       sp500_sample \\
    --signal         breakout \\
    --lookback       5 \\
    --min-conviction 0.3 \\
    --provider       ibkr \\
    --host           "${IBKR_HOST}" \\
    --port           "${IBKR_PORT}" \\
    --no-news \\
    --multi-lookback \\
    --secondary-lookback 20 \\
    --add-to-watchlist \\
    >> "${LOG_FILE}" 2>&1
echo "── [12:30] Midday check complete — \$(date '+%Y-%m-%d %H:%M:%S')" >> "${LOG_FILE}"
rm -f "\$0"
EOJOB
fi

# ── Background: EOD tracker + breadth update at 4:30 PM ET ───────────────────
if [[ "$SLEEP_EOD" -gt 0 ]]; then
    _schedule "[16:30] EOD tracker + breadth" << EOJOB
#!/usr/bin/env bash
# QuantLab EOD job — generated by morning.sh at $(ts)
sleep ${SLEEP_EOD}
source "${CONDA_BASE}/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate "${CONDA_ENV}" 2>/dev/null || true
cd "${PROJECT_DIR}"
{
  echo ""
  echo "══ [16:30] EOD tracker — \$(date '+%Y-%m-%d %H:%M:%S') ═══════════════"
} >> "${LOG_FILE}"
python scripts/track_forward_returns.py \\
    --host "${IBKR_HOST}" \\
    --port "${IBKR_PORT}" \\
    >> "${LOG_FILE}" 2>&1
echo "── [16:30] EOD tracker complete — \$(date '+%Y-%m-%d %H:%M:%S')" >> "${LOG_FILE}"
# Breadth update 5 min later (market data fully settled)
sleep 300
{
  echo ""
  echo "══ [16:35] Breadth update — \$(date '+%Y-%m-%d %H:%M:%S') ════════════"
} >> "${LOG_FILE}"
python scripts/update_breadth.py >> "${LOG_FILE}" 2>&1 || true
echo "── [16:35] Breadth update complete — \$(date '+%Y-%m-%d %H:%M:%S')" >> "${LOG_FILE}"
rm -f "\$0"
EOJOB
fi
