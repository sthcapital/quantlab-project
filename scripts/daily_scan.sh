#!/usr/bin/env bash
# =============================================================================
# scripts/daily_scan.sh — QuantLab daily pre-market scanner
#
# Run every trading-day morning before market open (e.g. 08:30 ET).
# 1. Activates the quantlab conda environment
# 2. Builds tradeable universe (Polygon grouped daily + Massive options check)
# 3. Updates breadth metrics (Polygon + Massive S3 flat files)
# 4. Runs scan_universe.py — bars from Massive S3 flatfile, options from
#    MassiveOptionsProvider (Polygon key).  TWS is NOT used for bars or options.
# 5. Runs run_backtest.py on every symbol scoring above 0.70 conviction
# 6. Appends all output to ~/quantlab-scan.log with timestamps
#
# TWS (IBKR Gateway) is used for:
#   - News headlines   (--with-news flag)
#   - Forward return price updates  (track_forward_returns.py)
#   - Execution / order placement   (not in this script)
#   TWS is NOT required for bars, options scoring, or the main scan.
#
# Cron (08:00 ET = 12:00 UTC Mon–Fri):
#   0 12 * * 1-5 /home/quantlab/projects/quantlab-project/scripts/daily_scan.sh
#
# Usage:
#   bash scripts/daily_scan.sh                # price-only scan (<5 min, flatfile bars)
#   bash scripts/daily_scan.sh --with-news    # include news tagging via IBKR
# =============================================================================

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
CONDA_ENV="quantlab"
LOG_FILE="$HOME/quantlab-scan.log"

IBKR_HOST="172.23.208.1"
IBKR_PORT="7497"

UNIVERSE="tradeable"               # sp500_sample (50) | tradeable (~2300) | tradeable_no_options

# ── Environment secrets ────────────────────────────────────────────────────────
if [[ -f "$PROJECT_DIR/.env" ]]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi
SIGNAL="breakout"
LOOKBACK="5"
SECONDARY_LOOKBACK="20"         # secondary lookback for multi-confirmation (lb=5 + lb=20)
MIN_CONVICTION="0.4"            # floor; auto-raised to 0.80 when breadth tape=BEAR
HIGH_CONV_THRESHOLD="0.70"      # symbols above this trigger a backtest run

# Two-year backtest window (GNU date / BSD date fallback for macOS)
BACKTEST_START="$(date -d '2 years ago' +%Y-%m-%d 2>/dev/null \
                  || date -v-2y +%Y-%m-%d)"
BACKTEST_END="$(date +%Y-%m-%d)"

WITH_NEWS=""
WITH_OPTIONS=""
for _arg in "$@"; do
    [[ "$_arg" == "--with-news"    ]] && WITH_NEWS="true"
    [[ "$_arg" == "--with-options" ]] && WITH_OPTIONS="true"
done

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
    echo "  QuantLab Daily Pre-Market Scan"
    echo "  $(ts)"
    echo "  Universe  : $UNIVERSE"
    echo "  Signal    : $SIGNAL  lookback=$LOOKBACK+$SECONDARY_LOOKBACK (multi-confirmed)  min_conviction=$MIN_CONVICTION"
    echo "  High-conv : >${HIGH_CONV_THRESHOLD} conviction → full backtest triggered  (✓ = multi-confirmed)"
    echo "  IBKR      : $IBKR_HOST:$IBKR_PORT"
    if [[ -n "$WITH_NEWS" ]]; then
        echo "  News      : enabled"
    else
        echo "  News      : disabled  (pass --with-news to enable)"
    fi
    if [[ -n "$WITH_OPTIONS" ]]; then
        echo "  Options   : enabled"
    else
        echo "  Options   : disabled  (pass --with-options to enable)"
    fi
    echo "  Backtest  : $BACKTEST_START → $BACKTEST_END  (2-year window)"
    sep
} | tee -a "$LOG_FILE"

# ── Tradeable universe build ────────────────────────────────────────────────────
# Ensures today's universe parquet cache exists before the scan.
# If the cache is already present (same-day re-run), this completes instantly.
# Non-fatal: scan falls back to the most recent cached universe on failure.
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

# ── Breadth update ──────────────────────────────────────────────────────────────
# Runs before scan_universe.py so the tape condition (BEAR/BULL) and the
# McClellan-based breadth_override flag are current when conviction is scored.
# POLYGON_API_KEY is available via the .env loaded above.
# Non-fatal: if Polygon is unreachable the scan falls back to the last cached
# breadth reading stored in DuckDB.
{
    echo ""
    echo "══ [$(ts)] Breadth update — $(date +%Y-%m-%d) ════════════════════════"
} | tee -a "$LOG_FILE"

if python scripts/update_breadth.py 2>&1 | tee -a "$LOG_FILE"; then
    log "Breadth update complete."
else
    log "WARNING: breadth update failed — scan will use last cached reading."
fi

# ── Pre-flight: TWS reachability ────────────────────────────────────────────────
log "Pre-flight: checking TWS at $IBKR_HOST:$IBKR_PORT ..."

if ! python3 -c "
import sys
from quantlab.providers.ibkr import ping_tws
ok = ping_tws('$IBKR_HOST', $IBKR_PORT, timeout=5.0)
sys.exit(0 if ok else 1)
" 2>/dev/null; then
    log "ABORT: TWS not reachable at $IBKR_HOST:$IBKR_PORT"
    log "       Start TWS or IB Gateway, enable socket API, and retry."
    sep
    exit 1
fi

log "TWS reachable — proceeding."

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
[[ -z "$WITH_NEWS"    ]] && SCAN_ARGS+=(--no-news)
[[ -n "$WITH_OPTIONS" ]] && SCAN_ARGS+=(--with-options)

# Capture and tee simultaneously; || true prevents set -e from aborting on
# a non-zero exit (e.g. if the scanner finds no setups and exits non-zero).
SCAN_OUTPUT="$(python scripts/scan_universe.py "${SCAN_ARGS[@]}" 2>&1 \
               | tee -a "$LOG_FILE")" || true

# ── Extract high-conviction symbols ─────────────────────────────────────────────
# Output lines look like:  "   1. XOM      conviction=0.75  close=..."
# "BRK B" contains a space; Python handles this — don't use word-splitting.
#
# The Python code is stored in a variable and passed via -c so it doesn't
# conflict with the stdin pipe carrying SCAN_OUTPUT.
_PARSE_SCRIPT='
import sys, re
threshold = float(sys.argv[1])
for line in sys.stdin:
    # conviction=X.XX optionally followed by " ✓" for multi-confirmed symbols
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
    log "Running full backtest on each to refresh IS metrics and Wyckoff analysis ..."

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

# ── Forward return tracking (run after close — records 1D/3D/5D returns) ──────
# This updates watchlist entries that hit their return horizon today.
# Safe to run in the morning scan too — it simply finds nothing to update yet.
log "Updating forward returns for watchlist entries ..."
python scripts/track_forward_returns.py \
    --host "$IBKR_HOST" \
    --port "$IBKR_PORT" \
    2>&1 | tee -a "$LOG_FILE" || log "WARNING: forward return tracking failed"

# ── Daily report ─────────────────────────────────────────────────────────────────
# Generates data/reports/YYYY-MM-DD_watchlist.html and updates daily_reports DuckDB.
# Non-fatal — scan results are already stored even if report generation fails.
log "Generating daily institutional watchlist report ..."
python scripts/generate_report.py 2>&1 | tee -a "$LOG_FILE" \
    || log "WARNING: report generation failed"

# ── Footer ───────────────────────────────────────────────────────────────────────
{
    echo "──────────────────────────────────────────────────────────"
    echo "  Finished: $(ts)"
    echo "  Full log: $LOG_FILE"
    sep
    echo ""
} | tee -a "$LOG_FILE"
