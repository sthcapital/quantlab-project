#!/usr/bin/env bash
# =============================================================================
# scripts/evening_scan.sh — QuantLab evening full-universe scanner
#
# Runs Mon-Fri at 5:00 PM ET (21:00 UTC during EDT) and Sunday at 6:00 PM ET.
# The full 2,325-symbol scan runs here so results are ready before market open
# the following morning.  The morning.sh script is a lightweight check only.
#
# Steps:
#   1. Build tradeable universe cache (Polygon grouped daily)
#   2. Update breadth (full S3 history load for SMA participation)
#   3. Update macro/FRED regime
#   4. Run EDGAR fundamentals refresh (Mondays only)
#   5. Full 2,325-symbol scan with --with-news --with-options
#      (degrades to --no-news if TWS is unreachable)
#   6. Backtest high-conviction symbols (>0.70)
#   7. IC monitor update
#   8. Generate daily report
#   9. Persistence summary to DuckDB
#
# Cron (EDT = UTC-4):
#   Mon-Fri : 0 21 * * 1-5  (5:00 PM ET)
#   Sunday  : 0 22 * * 0    (6:00 PM ET)
#
# Usage:
#   bash scripts/evening_scan.sh
# =============================================================================

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
CONDA_ENV="quantlab"
LOG_FILE="$HOME/quantlab-scan.log"

IBKR_HOST="172.23.208.1"
IBKR_PORT="7497"

UNIVERSE="tradeable"
SIGNAL="breakout"
LOOKBACK="5"
SECONDARY_LOOKBACK="20"
MIN_CONVICTION="0.4"
HIGH_CONV_THRESHOLD="0.70"

# Two-year backtest window (GNU date / BSD date fallback for macOS)
BACKTEST_START="$(date -d '2 years ago' +%Y-%m-%d 2>/dev/null \
                  || date -v-2y +%Y-%m-%d)"
BACKTEST_END="$(date +%Y-%m-%d)"

# ── Environment secrets ────────────────────────────────────────────────────────
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

# ── Helpers ─────────────────────────────────────────────────────────────────────
ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$LOG_FILE"; }
sep() { printf '%s\n' "══════════════════════════════════════════════════════════" \
          | tee -a "$LOG_FILE"; }

# ── Activate environment ────────────────────────────────────────────────────────
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$PROJECT_DIR"

# ── Header ──────────────────────────────────────────────────────────────────────
{
    echo ""
    sep
    echo "  QuantLab Evening Scan"
    echo "  $(ts)"
    echo "  Universe  : $UNIVERSE"
    echo "  Signal    : $SIGNAL  lookback=$LOOKBACK+$SECONDARY_LOOKBACK  min_conviction=$MIN_CONVICTION"
    echo "  High-conv : >${HIGH_CONV_THRESHOLD} conviction → full backtest triggered"
    echo "  IBKR      : $IBKR_HOST:$IBKR_PORT (news only — bars from flatfile)"
    echo "  Backtest  : $BACKTEST_START → $BACKTEST_END  (2-year window)"
    sep
} | tee -a "$LOG_FILE"

# ── Tradeable universe build ────────────────────────────────────────────────────
{
    echo ""
    echo "══ [$(ts)] Universe build — $(date +%Y-%m-%d) ════════════════════════"
} | tee -a "$LOG_FILE"

python -c "
from datetime import date
from quantlab.universe import UniverseManager, load_universe_cache
from quantlab.providers.polygon import PolygonProvider
import os, sys
today = date.today()
if load_universe_cache(today):
    print(f'  Universe cache hit for {today} — skipping rebuild')
    sys.exit(0)
api_key = os.environ.get('POLYGON_API_KEY', '')
if not api_key:
    print('  WARNING: POLYGON_API_KEY not set — will use most recent cached universe')
    sys.exit(0)
polygon = PolygonProvider(api_key=api_key)
mgr = UniverseManager()
syms, stats = mgr.build_tradeable_universe(today, polygon, ib=None, optionable_only=False)
print(f'  {stats.summary()}')
" 2>&1 | tee -a "$LOG_FILE" || log "WARNING: universe build failed — scan will use cached data"

# ── Breadth update (full S3 history load) ──────────────────────────────────────
{
    echo ""
    echo "══ [$(ts)] Breadth update — $(date +%Y-%m-%d) ════════════════════════"
} | tee -a "$LOG_FILE"

if python scripts/update_breadth.py 2>&1 | tee -a "$LOG_FILE"; then
    log "Breadth update complete."
else
    log "WARNING: breadth update failed — scan will use last cached reading."
fi

# ── Macro regime update (FRED + CBOE VIX) ──────────────────────────────────────
{
    echo ""
    echo "══ [$(ts)] Macro regime update — $(date +%Y-%m-%d) ══════════════════"
} | tee -a "$LOG_FILE"

python scripts/update_macro.py 2>&1 | tee -a "$LOG_FILE" \
    || log "WARNING: macro update failed — scan will use last cached reading"

# ── Weekly EDGAR fundamentals refresh (Mondays only) ───────────────────────────
if [[ "$(date +%u)" == "1" ]]; then
    {
        echo ""
        echo "══ [$(ts)] EDGAR fundamentals fetch (weekly, Monday) ═══════════"
    } | tee -a "$LOG_FILE"

    python scripts/fetch_edgar_universe.py 2>&1 | tee -a "$LOG_FILE" \
        || log "WARNING: EDGAR fundamentals fetch failed — scan will use cached data"
fi

# ── Pre-flight: TWS reachability (advisory — degrades to --no-news if down) ────
log "Pre-flight: checking TWS at $IBKR_HOST:$IBKR_PORT ..."

TWS_UP=""
if python3 -c "
import sys
from quantlab.providers.ibkr import ping_tws
ok = ping_tws('$IBKR_HOST', $IBKR_PORT, timeout=5.0)
sys.exit(0 if ok else 1)
" 2>/dev/null; then
    log "TWS reachable — news enabled."
    TWS_UP="true"
else
    log "WARNING: TWS not reachable at $IBKR_HOST:$IBKR_PORT"
    log "         Proceeding without news (options and bars unaffected)."
fi

# ── Universe scan ───────────────────────────────────────────────────────────────
log "Starting universe scan ..."

SCAN_ARGS=(
    --universe          "$UNIVERSE"
    --signal            "$SIGNAL"
    --lookback          "$LOOKBACK"
    --min-conviction    "$MIN_CONVICTION"
    --provider          flatfile
    --host              "$IBKR_HOST"
    --port              "$IBKR_PORT"
    --multi-lookback
    --secondary-lookback "$SECONDARY_LOOKBACK"
    --save-db
    --add-to-watchlist
)
if [[ -n "${POLYGON_API_KEY:-}" ]]; then
    SCAN_ARGS+=(--with-options)
else
    log "WARNING: POLYGON_API_KEY not set — skipping --with-options"
fi
if [[ -n "$TWS_UP" ]]; then
    # news is enabled by default — no flag needed
else
    SCAN_ARGS+=(--no-news)
fi

log "Running: python scripts/scan_universe.py ${SCAN_ARGS[*]}"
SCAN_OUTPUT="$(python scripts/scan_universe.py "${SCAN_ARGS[@]}" 2>&1 \
               | tee -a "$LOG_FILE")"
if [[ -z "$SCAN_OUTPUT" ]]; then
    log "ERROR: scan produced no output — check $LOG_FILE for details"
fi

# ── Extract high-conviction symbols ─────────────────────────────────────────────
_PARSE_SCRIPT='
import sys, re
threshold = float(sys.argv[1])
for line in sys.stdin:
    m = re.search(r"^\s+\d+\.\s+(.+?)\s{2,}conviction=([0-9.]+)", line)
    if m:
        sym  = m.group(1).strip()
        conv = float(m.group(2))
        if conv > threshold:
            print(sym)
'

mapfile -t HIGH_CONV_SYMS < <(
    echo "$SCAN_OUTPUT" | python3 -c "$_PARSE_SCRIPT" "$HIGH_CONV_THRESHOLD"
)

# ── Backtest high-conviction symbols ────────────────────────────────────────────
if [[ ${#HIGH_CONV_SYMS[@]} -eq 0 ]]; then
    log "No symbols above ${HIGH_CONV_THRESHOLD} conviction today — no backtests queued."
else
    log "High-conviction symbols (>${HIGH_CONV_THRESHOLD}): ${HIGH_CONV_SYMS[*]}"
    log "Running full backtest on each ..."

    for SYM in "${HIGH_CONV_SYMS[@]}"; do
        log "─── Backtest: $SYM ($BACKTEST_START → $BACKTEST_END) ───"

        python scripts/run_backtest.py \
            --provider ibkr \
            --symbol   "$SYM" \
            --start    "$BACKTEST_START" \
            --end      "$BACKTEST_END" \
            --signal   "$SIGNAL" \
            --lookback "$LOOKBACK" \
            --save-db  \
            --no-news  \
            --host     "$IBKR_HOST" \
            --port     "$IBKR_PORT" \
            2>&1 | tee -a "$LOG_FILE" || log "WARNING: backtest failed for $SYM"

        log "─── Backtest complete: $SYM"
    done
fi

# ── IC Monitor (signal information coefficient) ────────────────────────────────
{
    echo ""
    echo "══ [$(ts)] IC Monitor — $(date +%Y-%m-%d) ════════════════════════════"
} | tee -a "$LOG_FILE"

python -m quantlab.research.ic_monitor 2>&1 | tee -a "$LOG_FILE" \
    || log "WARNING: IC monitor failed — IC history not updated"

# ── Daily report ─────────────────────────────────────────────────────────────────
log "Generating daily institutional watchlist report ..."
python scripts/generate_report.py 2>&1 | tee -a "$LOG_FILE" \
    || log "WARNING: report generation failed"

# ── Post-scan persistence summary ───────────────────────────────────────────────
{
    echo ""
    echo "══ [$(ts)] Persistence summary ══════════════════════════════════════"
} | tee -a "$LOG_FILE"

python -c "
import sys
sys.path.insert(0, 'src')
try:
    import duckdb
    from quantlab.storage import DB_PATH
    con = duckdb.connect(str(DB_PATH))
    today = __import__('datetime').date.today().isoformat()
    tables = [
        ('scan_results',           'scan_date'),
        ('earnings_results',       'report_date'),
        ('macro_snapshots',        'as_of'),
        ('options_snapshots',      'snap_date'),
        ('edgar_fundamentals',     'fetch_date'),
        ('breadth_history',        'date'),
        ('institutional_watchlist','last_seen'),
        ('daily_reports',          'date'),
    ]
    print()
    for tbl, date_col in tables:
        try:
            total = con.execute(f'SELECT COUNT(*) FROM {tbl}').fetchone()[0]
            today_n = con.execute(
                f\"SELECT COUNT(*) FROM {tbl} WHERE CAST({date_col} AS VARCHAR) = ?\",
                [today]
            ).fetchone()[0]
            flag = '  ← NEW' if today_n > 0 else ''
            print(f'  {tbl:<30} total={total:>6}  today={today_n:>4}{flag}')
        except Exception as e:
            print(f'  {tbl:<30} (unavailable: {e})')
    con.close()
    print()
except Exception as e:
    print(f'  DuckDB summary failed: {e}')
" 2>&1 | tee -a "$LOG_FILE"

# ── Footer ───────────────────────────────────────────────────────────────────────
{
    echo "──────────────────────────────────────────────────────────"
    echo "  Finished: $(ts)"
    echo "  Full log: $LOG_FILE"
    sep
    echo ""
} | tee -a "$LOG_FILE"
