#!/usr/bin/env bash
# =============================================================================
# scripts/morning.sh — Manual morning routine / catch-up run
#
# Runs the full pre-market cycle in the correct order:
#   1. track_forward_returns.py  — records any return horizons reached since
#                                  the last run (catches up missed closes)
#   2. daily_scan.sh --with-news — full pre-market scan, watchlist update,
#                                  backtest on any 0.70+ signal
#   3. Watchlist status summary  — shows open positions, unrealised returns,
#                                  and cumulative hit rates
#
# Usage:
#   bash scripts/morning.sh             # standard run
#   bash scripts/morning.sh --no-news   # faster, price-only scan
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
CONDA_ENV="quantlab"

WITH_NEWS="--with-news"
[[ "${1:-}" == "--no-news" ]] && WITH_NEWS=""

# ── Activate environment ───────────────────────────────────────────────────────
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$PROJECT_DIR"

sep() { printf '%.0s═' {1..56}; printf '\n'; }
ts()  { date '+%Y-%m-%d %H:%M:%S'; }

echo ""
sep
echo "  QuantLab Morning Routine  $(ts)"
echo "  ${WITH_NEWS:-price-only (--no-news)}"
sep

# ── Step 1: Catch-up forward returns ──────────────────────────────────────────
echo ""
echo "── Step 1: Forward return catch-up ────────────────────"
python scripts/track_forward_returns.py \
    --no-ibkr 2>/dev/null || true
# --no-ibkr for instant offline inspection; the EOD cron job does the live run

# ── Step 2: Morning scan ───────────────────────────────────────────────────────
echo ""
echo "── Step 2: Morning scan ────────────────────────────────"
if [[ -n "$WITH_NEWS" ]]; then
    bash scripts/daily_scan.sh --with-news
else
    bash scripts/daily_scan.sh
fi

# ── Step 3: Watchlist status ───────────────────────────────────────────────────
echo ""
echo "── Step 3: Watchlist status ────────────────────────────"
python3 - <<'PYEOF'
from datetime import date
from quantlab.watchlist import (
    get_active_watchlist, get_watchlist_summary, _trading_days_elapsed
)

active  = get_active_watchlist()
summary = get_watchlist_summary()

by_status = summary.get("by_status", {})
watching  = by_status.get("watching", 0)
stopped   = by_status.get("stopped_out", 0)
expired   = by_status.get("expired", 0)
total     = summary.get("total", 0)

print(f"\n  Watchlist  total={total}  "
      f"watching={watching}  stopped_out={stopped}  expired={expired}")

if active:
    print(f"\n  {'symbol':<8}  {'date_added':>12}  {'entry':>8}  {'stop':>8}  "
          f"{'conv':>5}  {'days':>5}  {'unreal':>8}")
    print(f"  {'─'*72}")
    for e in active:
        da = date.fromisoformat(str(e["date_added"]))
        days = _trading_days_elapsed(da)
        unreal = e.get("unrealized_ret")
        unreal_str = f"{unreal*100:+.2f}%" if unreal is not None else "    --"
        flag = "  ⚠ NEAR STOP" if (
            unreal is not None and e["atr_stop"] and
            e["current_price"] and
            e["current_price"] < (e["atr_stop"] or 0) * 1.02
        ) else ""
        print(
            f"  {e['symbol']:<8}  {str(e['date_added']):>12}  "
            f"{e['entry_price']:>8.2f}  {e['atr_stop'] or 0:>8.2f}  "
            f"{e['conviction_score']:>5.2f}  {days:>5}  {unreal_str:>8}{flag}"
        )

# Cumulative realized return stats
has_data = False
for label, key in [("1D", "ret_1d"), ("3D", "ret_3d"), ("5D", "ret_5d")]:
    agg = summary.get(key, {})
    avg = agg.get("avg")
    hit = agg.get("hit_rate")
    if avg is not None:
        if not has_data:
            print("\n  Realized returns (cumulative):")
            has_data = True
        stars = "★" * (1 + int(hit >= 0.6) + int(hit >= 0.75))
        print(f"    {label}  avg={avg*100:+.2f}%  "
              f"hit_rate={hit*100:.0f}%  {stars}")

if not has_data:
    print("\n  No realized returns yet — forward data accumulates after market close")

PYEOF

echo ""
sep
echo "  Run complete: $(ts)"
sep
echo ""
