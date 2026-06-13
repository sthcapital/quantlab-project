#!/usr/bin/env bash
# =============================================================================
# scripts/morning.sh — QuantLab lightweight morning check
#
# Runs Mon-Fri at 8:45 AM ET (12:45 UTC during EDT).
# Completes in under 2 minutes — does NOT run the full universe scan.
# The full scan runs as evening_scan.sh at 5:00 PM ET so results are
# ready before the next morning's open.
#
# Steps:
#   1. Update breadth tape (rolling recompute only, --no-polygon)
#   2. Update macro/VIX (fast path — no heavy FRED API calls)
#   3. Print watchlist pre-market status (stop levels, conviction, days held)
#   4. Log today's tape and top watchlist candidates
#
# Cron (EDT = UTC-4):
#   45 12 * * 1-5  (8:45 AM ET)
#
# Usage:
#   bash scripts/morning.sh
# =============================================================================

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
CONDA_ENV="quantlab"
LOG_FILE="$HOME/quantlab-scan.log"

# ── Environment secrets ────────────────────────────────────────────────────────
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

# ── Helpers ─────────────────────────────────────────────────────────────────────
ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG_FILE"; }
sep() { printf '%s\n' "══════════════════════════════════════════════════════════" \
          | tee -a "$LOG_FILE"; }

# ── Run lock — a duplicate invocation must never run a second check ────────────
# shellcheck source=lib/run_lock.sh
source "$PROJECT_DIR/scripts/lib/run_lock.sh"
acquire_run_lock "morning" "$LOG_FILE"

# ── Activate environment ────────────────────────────────────────────────────────
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$PROJECT_DIR"

# ── Header ──────────────────────────────────────────────────────────────────────
{
    echo ""
    sep
    echo "  QuantLab Morning Check"
    echo "  $(ts)"
    echo "  (Lightweight — full scan runs at 5:00 PM ET via evening_scan.sh)"
    sep
} | tee -a "$LOG_FILE"

# ── Step 0: Finalize prior options sessions ───────────────────────────────────
# Yesterday's EOD options flat file is reliably published by now, so finalize any
# session the evening scan left intraday/unfinalized.  This is the normal path to
# a 'final' session.  A credential failure (exit 2) is surfaced loudly; a session
# still stuck is also caught by the noon-deadline alert in the health check and
# report header.
{
    echo ""
    echo "══ [$(ts)] Options finalization sweep — $(date +%Y-%m-%d) ══════════"
} | tee -a "$LOG_FILE"

if python scripts/finalize_sessions.py --sweep 2>&1 | tee -a "$LOG_FILE"; then
    log "Options finalization sweep complete."
else
    log "ALERT: options finalization sweep reported a credential/permission "
    log "       failure — a prior session's EOD flat file is unreadable.  Check "
    log "       POLYGON_S3_ACCESS_KEY_ID / POLYGON_API_KEY."
fi

# ── Step 1: Breadth tape (rolling recompute, no Polygon fetch) ────────────────
{
    echo ""
    echo "══ [$(ts)] Breadth tape (cached) — $(date +%Y-%m-%d) ═══════════════"
} | tee -a "$LOG_FILE"

python scripts/update_breadth.py --no-polygon 2>&1 | tee -a "$LOG_FILE" \
    || log "WARNING: breadth rolling recompute failed — last tape reading in effect"

# ── Step 2: Macro/VIX update (fast path) ──────────────────────────────────────
{
    echo ""
    echo "══ [$(ts)] Macro/VIX update — $(date +%Y-%m-%d) ════════════════════"
} | tee -a "$LOG_FILE"

python scripts/update_macro.py 2>&1 | tee -a "$LOG_FILE" \
    || log "WARNING: macro update failed — last cached reading will be used"

# ── Step 3: Watchlist pre-market status ───────────────────────────────────────
{
    echo ""
    echo "══ [$(ts)] Watchlist pre-market check — $(date +%Y-%m-%d) ══════════"
} | tee -a "$LOG_FILE"

python scripts/watchlist_status.py --no-ibkr 2>&1 | tee -a "$LOG_FILE" \
    || log "WARNING: watchlist status check failed"

# ── Step 4: Morning summary — tape + top candidates ───────────────────────────
{
    echo ""
    echo "══ [$(ts)] Morning summary ═══════════════════════════════════════════"
} | tee -a "$LOG_FILE"

python -c "
import sys
sys.path.insert(0, 'src')
try:
    from quantlab.signals.breadth import get_latest_snapshot
    from quantlab.watchlist import get_active_watchlist
    from datetime import date

    snap = get_latest_snapshot()
    if snap:
        print(f'  Tape     : {snap.tape}')
        print(f'  {snap.summary_line()}')
    else:
        print('  Tape     : (no breadth data available)')

    active = get_active_watchlist()
    if active:
        top = sorted(active, key=lambda x: -(x.get('conviction_score') or 0))[:5]
        print(f'  Watchlist: {len(active)} active entries — top candidates:')
        for e in top:
            print(f'    {e[\"symbol\"]:<8}  conv={e.get(\"conviction_score\", 0):.2f}  '
                  f'stop={e.get(\"atr_stop\") or 0:.2f}  '
                  f'layers={e.get(\"signal_layers\") or \"\"}')
    else:
        print('  Watchlist: (empty)')
except Exception as exc:
    print(f'  Morning summary failed: {exc}')
" 2>&1 | tee -a "$LOG_FILE"

# ── Footer ───────────────────────────────────────────────────────────────────────
{
    echo ""
    echo "──────────────────────────────────────────────────────────"
    echo "  Morning Check complete: $(ts)"
    echo "  Evening scan at 5:00 PM ET — full results ready ~5:40 PM"
    sep
    echo ""
} | tee -a "$LOG_FILE"
