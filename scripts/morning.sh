#!/usr/bin/env bash
# =============================================================================
# scripts/morning.sh — Self-scheduling morning routine.
#
# A single launch covers the full trading day via background sleep jobs:
#
#   IMMEDIATE  Step 1: update_breadth.py --no-polygon      (load tape from DuckDB)
#   IMMEDIATE  Step 2: track_forward_returns.py --no-ibkr  (catch-up closes)
#   IMMEDIATE  Step 3: daily_scan.sh --with-news           (morning scan, breadth loaded)
#   12:30 ET   Step 4: scan_universe.py --no-news          (midday check)
#   16:30 ET   Step 5: track_forward_returns.py + update_breadth.py  (EOD close)
#
# Breadth MUST run before the scan so the tape condition is consulted during
# conviction scoring and the min_conviction threshold is set correctly.
#
# Steps 3 and 4 are fire-and-forget background processes (disowned so they
# survive terminal closure). If morning.sh is launched after 12:30 ET, the
# midday step is skipped automatically.
#
# Usage:
#   bash scripts/morning.sh             # with news (default)
#   bash scripts/morning.sh --no-news   # faster, price-only morning scan
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
# ET wall-clock time (HH:MM) — used for schedule labels
et()  {
    python3 -c "
import pytz
from datetime import datetime
print(datetime.now(pytz.timezone('America/New_York')).strftime('%H:%M'))
"
}

echo ""
sep
echo "  QuantLab Morning Routine  $(ts)"
echo "  news: ${WITH_NEWS:-disabled}"
sep

# ── Compute schedule: sleep durations and human labels ────────────────────────
_SCHED=$(python3 - <<'PYEOF'
import pytz
from datetime import datetime

NY  = pytz.timezone("America/New_York")
now = datetime.now(NY)

def to_epoch(h, m):
    return int(NY.localize(
        datetime(now.year, now.month, now.day, h, m, 0)
    ).timestamp())

def fmt(secs):
    if secs <= 0:
        return "PAST"
    h, r = divmod(secs, 3600)
    m = r // 60
    return f"{h}h {m}m" if h else f"{m}m"

now_e    = int(now.timestamp())
midday_e = to_epoch(12, 30)
eod_e    = to_epoch(16, 30)

sm = max(0, midday_e - now_e)
se = max(0, eod_e    - now_e)

# One value per line so bash read handles spaces in "Xh Ym" correctly
print(sm)
print(se)
print(fmt(sm))
print(fmt(se))
PYEOF
)

SLEEP_MIDDAY=$(echo "$_SCHED" | sed -n '1p')
SLEEP_EOD=$(echo    "$_SCHED" | sed -n '2p')
DUR_MIDDAY=$(echo   "$_SCHED" | sed -n '3p')
DUR_EOD=$(echo      "$_SCHED" | sed -n '4p')

# ── Step 1: Load breadth tape from DuckDB ─────────────────────────────────────
# Must run BEFORE the scan so scan_universe.py reads the current tape condition
# and raises min_conviction to 0.80 if tape=BEAR.  --no-polygon skips the API
# call and only recomputes rolling metrics from stored data (fast, offline).
echo ""
echo "── [$(et)] Step 1: Breadth tape load ────────────────────────"
python scripts/update_breadth.py --no-polygon 2>/dev/null || true

# ── Step 2: Forward return catch-up ───────────────────────────────────────────
echo ""
echo "── [$(et)] Step 2: Forward return catch-up ──────────────────"
python scripts/track_forward_returns.py --no-ibkr 2>/dev/null || true

# ── Step 3: Morning scan ───────────────────────────────────────────────────────
echo ""
echo "── [$(et)] Step 3: Morning scan ─────────────────────────────"
if [[ -n "$WITH_NEWS" ]]; then
    bash scripts/daily_scan.sh --with-news
else
    bash scripts/daily_scan.sh
fi

# Capture the ET time right when the morning scan finishes
MORNING_ET="$(et)"

# ── Collect watchlist additions from today ────────────────────────────────────
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
echo ""
sep
echo "  [$MORNING_ET] Morning scan complete — $_WL_MSG"

if [[ "$SLEEP_MIDDAY" -gt 0 ]]; then
    echo "  [12:30] Midday check scheduled — runs in $DUR_MIDDAY"
else
    echo "  [12:30] Midday check SKIPPED — already past"
fi

if [[ "$SLEEP_EOD" -gt 0 ]]; then
    echo "  [16:30] EOD tracker scheduled — runs in $DUR_EOD"
else
    echo "  [16:30] EOD tracker SKIPPED — already past 16:30 ET"
fi
sep

# ── Step 4: Watchlist dashboard ───────────────────────────────────────────────
echo ""
echo "── [$(et)] Step 4: Watchlist dashboard ──────────────────────"
python scripts/watchlist_status.py --no-ibkr || true

echo ""

# ── Background: Midday check at 12:30 PM ET ───────────────────────────────────
if [[ "$SLEEP_MIDDAY" -gt 0 ]]; then
    (
        sleep "$SLEEP_MIDDAY"
        source "$CONDA_BASE/etc/profile.d/conda.sh" 2>/dev/null
        conda activate "$CONDA_ENV" 2>/dev/null
        cd "$PROJECT_DIR"
        {
            echo ""
            echo "══ [12:30] Midday check — $(ts) ══════════════════════════"
        } >> "$LOG_FILE"
        python scripts/scan_universe.py \
            --universe       sp500_sample \
            --signal         breakout \
            --lookback       5 \
            --min-conviction 0.3 \
            --provider       ibkr \
            --host           "$IBKR_HOST" \
            --port           "$IBKR_PORT" \
            --no-news \
            --multi-lookback \
            --secondary-lookback 20 \
            --add-to-watchlist \
            >> "$LOG_FILE" 2>&1
        echo "── [12:30] Midday check complete — $(ts)" >> "$LOG_FILE"
    ) &
    disown $!
fi

# ── Background: EOD tracker at 4:30 PM ET ────────────────────────────────────
if [[ "$SLEEP_EOD" -gt 0 ]]; then
    (
        sleep "$SLEEP_EOD"
        source "$CONDA_BASE/etc/profile.d/conda.sh" 2>/dev/null
        conda activate "$CONDA_ENV" 2>/dev/null
        cd "$PROJECT_DIR"
        {
            echo ""
            echo "══ [16:30] EOD tracker — $(ts) ═══════════════════════════"
        } >> "$LOG_FILE"
        python scripts/track_forward_returns.py \
            --host "$IBKR_HOST" \
            --port "$IBKR_PORT" \
            >> "$LOG_FILE" 2>&1
        echo "── [16:30] EOD tracker complete — $(ts)" >> "$LOG_FILE"

        # Breadth update — runs after forward returns (market close data ready)
        echo "══ [16:35] Breadth update — $(ts) ════════════════════════" >> "$LOG_FILE"
        python scripts/update_breadth.py >> "$LOG_FILE" 2>&1 || true
        echo "── [16:35] Breadth update complete — $(ts)" >> "$LOG_FILE"
    ) &
    disown $!
fi
