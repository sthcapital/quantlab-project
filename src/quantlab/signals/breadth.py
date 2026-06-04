"""
quantlab.signals.breadth — Institutional market breadth computation.

Uses Polygon.io grouped-daily data (all ~12,299 US stocks in one call) to
compute the full suite of breadth metrics used by institutional traders:

Primary signals (Stockbee / Gil Morales / Chris Kacher approach):
    up_4pct_count / down_4pct_count  — "power" moves ≥ 4% in a session
    10-day ratio                      — rolling up_4pct / down_4pct
    up_25pct_quarter                  — stocks up ≥ 25% in 63 days
    down_25pct_quarter                — stocks down ≥ 25% in 63 days

Classic market-internals signals:
    advances / declines
    new_highs_52w / new_lows_52w
    pct_above_20sma / 50sma / 200sma
    advance_decline_ratio
    new_high_low_ratio

Rolling / momentum signals:
    McClellan Oscillator  = EMA₁₉(A-D) − EMA₃₉(A-D)
    McClellan Summation   = cumulative McClellan Oscillator
    AD line               = cumulative (A - D)

Tape classification:
    BULL    — 10d-ratio > 2.0 and McClellan > -100
    NEUTRAL — 10d-ratio 1.0-2.0
    BEAR    — 10d-ratio < 0.5 or McClellan < -100 or up_25pct_quarter < 200
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Sequence

logger = logging.getLogger(__name__)

# ── BreadthSnapshot dataclass ──────────────────────────────────────────────────

@dataclass
class BreadthSnapshot:
    """Complete market breadth reading for a single trading day."""

    date: str                           # YYYY-MM-DD

    # Raw counts
    advances: int           = 0         # stocks up on day (vol filter applied)
    declines: int           = 0         # stocks down on day
    unchanged: int          = 0         # flat on day
    total_stocks: int       = 0         # total stocks processed

    # Power-move counts (Stockbee primary)
    up_4pct_count: int      = 0         # stocks up ≥ 4%
    down_4pct_count: int    = 0         # stocks down ≥ 4%
    up_25pct_quarter: int   = 0         # stocks up ≥ 25% over 63 days
    down_25pct_quarter: int = 0         # stocks down ≥ 25% over 63 days

    # 52-week extremes
    new_highs_52w: int      = 0
    new_lows_52w: int       = 0

    # SMA participation
    pct_above_20sma: float  = 0.0       # 0.0–100.0
    pct_above_50sma: float  = 0.0
    pct_above_200sma: float = 0.0

    # Derived ratios
    advance_decline_ratio: float = 0.0  # advances / declines
    new_high_low_ratio: float    = 0.0  # new_highs / new_lows

    # Rolling metrics (filled by rolling_breadth())
    ratio_10d: float | None              = None   # 10d up_4pct / down_4pct
    mcclellan_oscillator: float | None   = None
    mcclellan_summation: float | None    = None
    ad_line: int | None                  = None

    # Tape classification
    tape: str = "NEUTRAL"   # BULL / NEUTRAL / BEAR

    def summary_line(self) -> str:
        """One-line breadth summary for scan output and logs."""
        r10  = f"{self.ratio_10d:.2f}" if self.ratio_10d is not None else "--"
        mc   = f"{self.mcclellan_oscillator:+.0f}" if self.mcclellan_oscillator is not None else "--"
        return (
            f"Breadth {self.date}: "
            f"A={self.advances} D={self.declines} | "
            f"up4%={self.up_4pct_count} dn4%={self.down_4pct_count} | "
            f"10d-ratio={r10} | "
            f"NH={self.new_highs_52w} NL={self.new_lows_52w} | "
            f"McClellan={mc} | "
            f"tape={self.tape}"
        )


# ── EMA helper ─────────────────────────────────────────────────────────────────

def _ema(values: list[float], period: int) -> list[float]:
    """
    Compute exponential moving average.

    Uses standard multiplier = 2 / (period + 1).  The first value is used
    as the seed — no warm-up period is required.
    """
    if not values:
        return []
    k   = 2.0 / (period + 1)
    out = [values[0]]
    for v in values[1:]:
        out.append(v * k + out[-1] * (1.0 - k))
    return out


# ── Core breadth computation ───────────────────────────────────────────────────

def compute_market_breadth(
    trade_date: date | str,
    today_data: dict,                  # {symbol: Bar} from get_grouped_daily
    prev_data: dict | None = None,     # {symbol: Bar} from previous day
    history_data: dict | None = None,  # {symbol: list[Bar]} 252 days for NH/NL/SMA
    min_volume: float = 10_000,
) -> BreadthSnapshot:
    """
    Compute a complete BreadthSnapshot from grouped daily data.

    Args:
        trade_date:   The trading date being analysed.
        today_data:   {symbol: Bar} from PolygonProvider.get_grouped_daily().
        prev_data:    Previous session grouped daily for close-to-close returns.
                      Falls back to intraday (close/open) when None.
        history_data: {symbol: [Bar]} — 252-day history per symbol for SMA and
                      52-week high/low computation.  When None, those fields are 0.
        min_volume:   Minimum volume to include a stock (filters penny/inactive).

    Returns:
        BreadthSnapshot with all available fields populated.
    """
    if isinstance(trade_date, str):
        trade_date = date.fromisoformat(trade_date)

    snapshot = BreadthSnapshot(date=trade_date.isoformat())

    advances = declines = unchanged = 0
    up_4pct = down_4pct = 0
    new_highs = new_lows = 0
    above_20 = above_50 = above_200 = sma_total = 0
    up_25q = down_25q = 0

    for symbol, bar in today_data.items():
        if bar.volume < min_volume:
            continue
        if bar.close <= 0:
            continue

        # ── Percent change: close-to-close if prev available, else intraday ──
        if prev_data and symbol in prev_data and prev_data[symbol].close > 0:
            pct = (bar.close / prev_data[symbol].close) - 1.0
        elif bar.open > 0:
            pct = (bar.close / bar.open) - 1.0
        else:
            continue

        if pct > 0:
            advances += 1
        elif pct < 0:
            declines += 1
        else:
            unchanged += 1

        if pct >= 0.04:
            up_4pct += 1
        elif pct <= -0.04:
            down_4pct += 1

        # ── Historical metrics (require prior bar data) ────────────────────────
        if history_data and symbol in history_data:
            hist = history_data[symbol]
            n    = len(hist)

            # 52-week high/low (need 252 bars)
            if n >= 252:
                closes_252 = [b.close for b in hist[-252:]]
                hi52 = max(closes_252)
                lo52 = min(closes_252)
                if bar.close >= hi52 * 0.99:
                    new_highs += 1
                elif bar.close <= lo52 * 1.01:
                    new_lows += 1

            # SMA participation
            if n >= 20:
                sma20 = sum(b.close for b in hist[-20:]) / 20
                if n >= 50:
                    sma50 = sum(b.close for b in hist[-50:]) / 50
                else:
                    sma50 = None
                if n >= 200:
                    sma200 = sum(b.close for b in hist[-200:]) / 200
                else:
                    sma200 = None

                sma_total += 1
                if bar.close > sma20:   above_20  += 1
                if sma50  and bar.close > sma50:  above_50  += 1
                if sma200 and bar.close > sma200: above_200 += 1

            # 25% quarter move (63 bars)
            if n >= 63:
                ret_63 = (bar.close / hist[-63].close) - 1.0
                if ret_63 >= 0.25:
                    up_25q += 1
                elif ret_63 <= -0.25:
                    down_25q += 1

    total = advances + declines + unchanged
    snapshot.advances           = advances
    snapshot.declines           = declines
    snapshot.unchanged          = unchanged
    snapshot.total_stocks       = total
    snapshot.up_4pct_count      = up_4pct
    snapshot.down_4pct_count    = down_4pct
    snapshot.new_highs_52w      = new_highs
    snapshot.new_lows_52w       = new_lows
    snapshot.up_25pct_quarter   = up_25q
    snapshot.down_25pct_quarter = down_25q

    if sma_total > 0:
        snapshot.pct_above_20sma  = round(above_20  / sma_total * 100, 2)
        snapshot.pct_above_50sma  = round(above_50  / sma_total * 100, 2)
        snapshot.pct_above_200sma = round(above_200 / sma_total * 100, 2)

    snapshot.advance_decline_ratio = (
        round(advances / declines, 4) if declines > 0 else float(advances)
    )
    snapshot.new_high_low_ratio = (
        round(new_highs / new_lows, 4) if new_lows > 0 else float(new_highs)
    )

    return snapshot


# ── Rolling metrics ────────────────────────────────────────────────────────────

def rolling_breadth(
    snapshots: list[BreadthSnapshot],
    window: int = 10,
) -> list[BreadthSnapshot]:
    """
    Compute rolling breadth metrics and classify the tape for each snapshot.

    Adds to each snapshot (in-place):
        ratio_10d             — rolling window up_4pct / down_4pct
        mcclellan_oscillator  — EMA₁₉(A-D) − EMA₃₉(A-D)
        mcclellan_summation   — cumulative McClellan Oscillator
        ad_line               — cumulative A-D
        tape                  — BULL / NEUTRAL / BEAR

    Args:
        snapshots: List of BreadthSnapshot ordered oldest → newest.
        window:    Rolling window for up_4pct / down_4pct ratio (default 10).

    Returns:
        The same list with rolling fields populated.
    """
    if not snapshots:
        return snapshots

    n        = len(snapshots)
    ad_vals  = [float(s.advances - s.declines) for s in snapshots]
    ema19    = _ema(ad_vals, 19)
    ema39    = _ema(ad_vals, 39)

    cumulative_mc = 0.0
    cumulative_ad = 0

    for i, s in enumerate(snapshots):
        # 10-day ratio
        start = max(0, i - window + 1)
        w_up  = sum(snapshots[j].up_4pct_count  for j in range(start, i + 1))
        w_dn  = sum(snapshots[j].down_4pct_count for j in range(start, i + 1))
        s.ratio_10d = round(w_up / w_dn, 4) if w_dn > 0 else float(w_up)

        # McClellan
        mc = round(ema19[i] - ema39[i], 2)
        s.mcclellan_oscillator = mc
        cumulative_mc += mc
        s.mcclellan_summation  = round(cumulative_mc, 2)

        # AD line
        cumulative_ad += (s.advances - s.declines)
        s.ad_line = cumulative_ad

        # Tape classification
        s.tape = _classify_tape(s)

    return snapshots


def _classify_tape(s: BreadthSnapshot) -> str:
    """Classify the market tape from a BreadthSnapshot's rolling metrics."""
    mc  = s.mcclellan_oscillator or 0.0
    r10 = s.ratio_10d
    u25 = s.up_25pct_quarter

    # Hard bear signals (override individual conviction)
    if mc < -100:
        return "BEAR"
    if u25 > 0 and u25 < 200:
        return "BEAR"

    if r10 is None:
        return "NEUTRAL"

    if r10 > 2.0:
        return "BULL"
    if r10 > 1.0:
        return "NEUTRAL"
    if r10 > 0.5:
        return "WEAK"
    return "BEAR"


# ── Breadth regime adjustment for conviction scorer ────────────────────────────

def breadth_regime_adjustment(
    snapshot: BreadthSnapshot | None,
) -> tuple[float, bool]:
    """
    Compute the conviction score adjustment and override flag from breadth.

    Returns:
        (adjustment, override) where:
            adjustment — negative delta to apply to conviction score (-0.12 to 0.0)
            override   — True when market is in hard-bear mode (no new longs)

    Scoring matrix:
        10d-ratio > 2.0         : 0.00  (full bull — no penalty)
        10d-ratio 1.0–2.0       : -0.03
        10d-ratio 0.5–1.0       : -0.07
        10d-ratio < 0.5         : -0.12
        new_high_low_ratio < 0.5: -0.05 additional
        McClellan < -100        : override → no new longs
        up_25pct_quarter < 200  : override → no new longs
    """
    if snapshot is None or snapshot.ratio_10d is None:
        return 0.0, False   # neutral when no breadth data

    mc  = snapshot.mcclellan_oscillator or 0.0
    r10 = snapshot.ratio_10d
    u25 = snapshot.up_25pct_quarter

    # Hard overrides
    if mc < -100:
        return 0.0, True
    if u25 > 0 and u25 < 200:
        return 0.0, True

    # Ratio-based penalty
    if r10 > 2.0:
        adj = 0.00
    elif r10 > 1.0:
        adj = -0.03
    elif r10 > 0.5:
        adj = -0.07
    else:
        adj = -0.12

    # Additional warning for NH/NL deterioration
    nhl = snapshot.new_high_low_ratio
    if nhl > 0 and nhl < 0.5:
        adj = round(adj - 0.05, 4)

    return adj, False


# ── DuckDB persistence ────────────────────────────────────────────────────────

def save_breadth_snapshot(snapshot: BreadthSnapshot) -> None:
    """Insert or replace a BreadthSnapshot in the DuckDB breadth_history table."""
    try:
        from quantlab.storage import get_db
        con = get_db()
        con.execute("""
            INSERT OR REPLACE INTO breadth_history (
                date, advances, declines, unchanged, total_stocks,
                up_4pct_count, down_4pct_count, up_25pct_quarter, down_25pct_quarter,
                new_highs_52w, new_lows_52w,
                pct_above_20sma, pct_above_50sma, pct_above_200sma,
                advance_decline_ratio, new_high_low_ratio,
                ratio_10d, mcclellan_oscillator, mcclellan_summation, ad_line, tape
            ) VALUES (
                ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
            )
        """, [
            snapshot.date,
            snapshot.advances, snapshot.declines, snapshot.unchanged, snapshot.total_stocks,
            snapshot.up_4pct_count, snapshot.down_4pct_count,
            snapshot.up_25pct_quarter, snapshot.down_25pct_quarter,
            snapshot.new_highs_52w, snapshot.new_lows_52w,
            snapshot.pct_above_20sma, snapshot.pct_above_50sma, snapshot.pct_above_200sma,
            snapshot.advance_decline_ratio, snapshot.new_high_low_ratio,
            snapshot.ratio_10d, snapshot.mcclellan_oscillator,
            snapshot.mcclellan_summation, snapshot.ad_line, snapshot.tape,
        ])
        con.close()
    except Exception as e:
        logger.warning("breadth_history insert failed: %s", e)


def load_recent_snapshots(n: int = 60) -> list[BreadthSnapshot]:
    """Load the most recent n BreadthSnapshots from DuckDB, oldest first."""
    try:
        from quantlab.storage import get_db
        con = get_db()
        rows = con.execute(f"""
            SELECT date, advances, declines, unchanged, total_stocks,
                   up_4pct_count, down_4pct_count, up_25pct_quarter, down_25pct_quarter,
                   new_highs_52w, new_lows_52w,
                   pct_above_20sma, pct_above_50sma, pct_above_200sma,
                   advance_decline_ratio, new_high_low_ratio,
                   ratio_10d, mcclellan_oscillator, mcclellan_summation, ad_line, tape
            FROM breadth_history
            ORDER BY date DESC LIMIT {n}
        """).fetchall()
        con.close()
        snapshots = []
        for r in reversed(rows):  # oldest first
            s = BreadthSnapshot(
                date=str(r[0]),
                advances=r[1], declines=r[2], unchanged=r[3], total_stocks=r[4],
                up_4pct_count=r[5], down_4pct_count=r[6],
                up_25pct_quarter=r[7], down_25pct_quarter=r[8],
                new_highs_52w=r[9], new_lows_52w=r[10],
                pct_above_20sma=r[11] or 0.0, pct_above_50sma=r[12] or 0.0,
                pct_above_200sma=r[13] or 0.0,
                advance_decline_ratio=r[14] or 0.0,
                new_high_low_ratio=r[15] or 0.0,
                ratio_10d=r[16], mcclellan_oscillator=r[17],
                mcclellan_summation=r[18], ad_line=r[19],
                tape=r[20] or "NEUTRAL",
            )
            snapshots.append(s)
        return snapshots
    except Exception as e:
        logger.debug("load_recent_snapshots failed: %s", e)
        return []


def get_latest_snapshot() -> BreadthSnapshot | None:
    """Return the most recent stored BreadthSnapshot, or None if table is empty."""
    rows = load_recent_snapshots(1)
    return rows[0] if rows else None
