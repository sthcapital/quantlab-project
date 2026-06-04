"""
quantlab.signals.volume_profile — Institutional volume signature detection.

Three functions that characterise the *quality* of volume during and leading
up to a breakout.  All operate on OHLCV Bar sequences; no tick data required.

The Wyckoff accumulation fingerprint in volume:
    - Advancing sessions carry above-average volume  (institutions buying)
    - Declining sessions carry below-average volume  (weak sellers, no supply)
    - The breakout bar is the highest-volume session in the base window

Functions:
    accumulation_days_ratio(bars, window=60)
        Ratio of above-avg-volume up-days to all above-avg-volume days.
        >0.6 = buying pressure dominates; <0.4 = selling pressure dominates.

    volume_trend_score(bars, window=20)
        Fraction of bars matching the ideal accumulation signature
        (up day + above-avg vol OR down day + below-avg vol).
        0.5 = random/neutral; >0.65 = clean accumulation character.

    climactic_volume_score(bars, lookback=20)
        Scores how far the most recent bar's volume exceeds the prior
        lookback maximum.  A true breakout bar is the largest-volume
        session in the base period.  Returns 0.0 when the last bar is
        not even the prior maximum.
"""

from __future__ import annotations

from typing import Sequence

from quantlab.providers.base import Bar


# ── Rolling volume helper (inline to avoid cross-module coupling) ──────────────

def _rolling_avg_volume(bars: list[Bar], period: int = 20) -> list[float]:
    result: list[float] = []
    for i in range(len(bars)):
        window = [b.volume for b in bars[max(0, i - period):i]]
        result.append(sum(window) / len(window) if window else 0.0)
    return result


# ── 1. Accumulation days ratio ─────────────────────────────────────────────────

def accumulation_days_ratio(
    bars: Sequence[Bar],
    window: int = 60,
    vol_period: int = 20,
) -> float:
    """
    Ratio of above-average-volume up-days to all above-average-volume days.

    Counts only bars where volume exceeds the rolling `vol_period` average,
    then classifies each as an accumulation bar (close > open) or a
    distribution bar (close < open).

        score = accumulation_bars / (accumulation_bars + distribution_bars)

    Args:
        bars:       OHLCV sequence, oldest first.
        window:     Number of recent bars to evaluate (default 60).
        vol_period: Rolling average volume period (default 20).

    Returns:
        Float in [0.0, 1.0].  0.5 = neutral (no heavy-volume bars or balanced).
        > 0.6 = accumulation dominant.  < 0.4 = distribution dominant.
    """
    bars = list(bars)
    if len(bars) < vol_period + 2:
        return 0.5

    window_bars = bars[-window:] if len(bars) >= window else bars
    avg_vols    = _rolling_avg_volume(bars, vol_period)
    offset      = len(bars) - len(window_bars)

    accum = 0
    distrib = 0

    for i, bar in enumerate(window_bars):
        avg_vol = avg_vols[offset + i]
        if avg_vol <= 0 or bar.volume < avg_vol:
            continue  # only count above-average-volume bars

        if bar.close > bar.open:
            accum += 1
        elif bar.close < bar.open:
            distrib += 1

    total = accum + distrib
    if total == 0:
        return 0.5

    return round(accum / total, 4)


# ── 2. Volume trend score ──────────────────────────────────────────────────────

def volume_trend_score(
    bars: Sequence[Bar],
    window: int = 20,
    vol_period: int = 20,
) -> float:
    """
    Score the ideal accumulation volume signature: expanding on advances,
    contracting on declines.

    For each bar in the window, checks two conditions:
        - Up day  (close > open) AND volume > rolling average  → ideal up-bar
        - Down day (close < open) AND volume < rolling average → ideal down-bar

    Both conditions are instances of the Wyckoff accumulation pattern.
    The score is the fraction of bars that satisfy either condition.

    Args:
        bars:       OHLCV sequence, oldest first.
        window:     Number of recent bars to evaluate (default 20).
        vol_period: Rolling average volume period (default 20).

    Returns:
        Float in [0.0, 1.0].  ~0.50 = random / no signal.
        > 0.65 = clean accumulation character.
    """
    bars = list(bars)
    if len(bars) < vol_period + 2:
        return 0.5

    window_bars = bars[-window:] if len(bars) >= window else bars
    avg_vols    = _rolling_avg_volume(bars, vol_period)
    offset      = len(bars) - len(window_bars)

    ideal = 0

    for i, bar in enumerate(window_bars):
        avg_vol = avg_vols[offset + i]
        if avg_vol <= 0:
            continue

        is_up   = bar.close > bar.open
        is_down = bar.close < bar.open
        above   = bar.volume > avg_vol
        below   = bar.volume < avg_vol

        if (is_up and above) or (is_down and below):
            ideal += 1

    total = len(window_bars)
    return round(ideal / total, 4) if total > 0 else 0.5


# ── 3. Climactic volume score ──────────────────────────────────────────────────

def climactic_volume_score(
    bars: Sequence[Bar],
    lookback: int = 20,
) -> float:
    """
    Score how far the most recent bar's volume exceeds the prior lookback max.

    A genuine breakout bar should be the highest-volume session in the base
    period — institutions committing capital to drive price above resistance.

    Scoring:
        last_bar.volume < prior_max             → 0.0 (not climactic)
        last_bar.volume == prior_max            → 0.0 (just tied)
        last_bar.volume == 1.5 × prior_max      → 0.33
        last_bar.volume == 2.0 × prior_max      → 0.67
        last_bar.volume >= 2.5 × prior_max      → 1.0

    Args:
        bars:     OHLCV sequence, oldest first. Needs ≥ lookback + 2 bars.
        lookback: Number of prior bars to use as the volume baseline.

    Returns:
        Float in [0.0, 1.0].  Threshold for conviction boost: ≥ 0.70.
    """
    bars = list(bars)
    if len(bars) < lookback + 2:
        return 0.0

    last_vol   = bars[-1].volume
    prior_vols = [b.volume for b in bars[-lookback - 1: -1]]
    prior_max  = max(prior_vols) if prior_vols else 0.0

    if prior_max <= 0 or last_vol <= prior_max:
        return 0.0

    ratio = last_vol / prior_max
    # Linear from 0 (ratio=1.0) to 1.0 (ratio=2.5)
    score = min(1.0, (ratio - 1.0) / 1.5)
    return round(score, 4)
