"""
quantlab.signals.wyckoff — Wyckoff accumulation / distribution detection.

Four pure functions operating on sequences of Bar objects.  No I/O, no side
effects, no external dependencies beyond the standard library.

Theory:
    Richard Wyckoff described how the "composite operator" (large institutions)
    accumulate or distribute positions in phases before a markup or markdown.
    These functions identify structural signatures of accumulation so the
    scanner can distinguish genuine base breakouts from distribution traps.

Functions:
    absorption_score(bars)         — high volume without new lows
    base_quality_score(bars)       — tightness and duration of consolidation
    volume_character_score(bars)   — up-day vs down-day volume ratio in base
    is_wyckoff_spring(bars)        — brief undercut of support then recovery
"""

from __future__ import annotations

import math
from typing import Sequence

from quantlab.providers.base import Bar


# ── Helpers ────────────────────────────────────────────────────────────────────

def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _rolling_avg_volume(bars: Sequence[Bar], period: int = 20) -> list[float]:
    """Return the N-bar simple average volume for each bar (backward-looking)."""
    bars = list(bars)
    result: list[float] = []
    for i, _ in enumerate(bars):
        window = [b.volume for b in bars[max(0, i - period):i]]
        result.append(_avg(window) if window else 0.0)
    return result


def _atr(bars: Sequence[Bar], period: int = 14) -> float | None:
    """Average True Range over the last `period` bars."""
    bars = list(bars)
    if len(bars) < period + 1:
        return None
    trs = [
        max(
            bars[i].high - bars[i].low,
            abs(bars[i].high - bars[i - 1].close),
            abs(bars[i].low  - bars[i - 1].close),
        )
        for i in range(1, len(bars))
    ]
    return _avg(trs[-period:])


# ── 1. Absorption score ────────────────────────────────────────────────────────

def absorption_score(
    bars: Sequence[Bar],
    volume_threshold: float = 1.3,
    lookback: int = 60,
    volume_period: int = 20,
) -> float:
    """
    Detect high volume without new lows — the Wyckoff absorption signature.

    Institutions absorbing supply will generate above-average volume while
    price refuses to make new lows.  A high score means most of the heavy-
    volume activity occurred on bars where price held up (absorption), not
    on bars where price broke to new lows (genuine distribution selling).

    Algorithm:
        For each bar in the lookback window:
          - "Heavy volume" = volume > avg_volume_20d * volume_threshold
          - "Absorbed"     = bar is heavy-volume AND low >= N-bar rolling low
          - "Distributed"  = bar is heavy-volume AND low  < N-bar rolling low

        score = absorbed / (absorbed + distributed)  if any heavy-volume bars
              = 0.5  (neutral) if no heavy-volume bars in the window

    Args:
        bars:             OHLCV bar sequence, oldest first. Needs ≥ lookback bars.
        volume_threshold: Multiplier above 20-day avg to qualify as heavy volume.
        lookback:         Number of recent bars to evaluate.
        volume_period:    Rolling average volume period.

    Returns:
        Float in [0.0, 1.0]. >0.65 = absorption dominant. <0.35 = distribution.
    """
    bars = list(bars)
    if len(bars) < volume_period + 2:
        return 0.5

    window = bars[-lookback:] if len(bars) >= lookback else bars
    avg_vols = _rolling_avg_volume(bars, volume_period)
    # align avg_vols to the window
    offset = len(bars) - len(window)
    window_avg_vols = avg_vols[offset:]

    # Establish the support level from the base-formation period (first 2/3 of
    # window).  Using a rolling minimum would follow the price down and make
    # a gradual decline look like absorption — the fixed support reference
    # correctly anchors the comparison to the established base.
    base_end = max(1, int(len(window) * 2 / 3))
    base_support = min(b.low for b in window[:base_end])

    absorbed = 0
    distributed = 0

    for i, bar in enumerate(window):
        avg_vol = window_avg_vols[i]
        if avg_vol <= 0 or bar.volume < avg_vol * volume_threshold:
            continue  # not a heavy-volume bar

        if bar.low >= base_support * (1.0 - 0.015):
            absorbed += 1      # heavy volume, price holds near support — absorption
        else:
            distributed += 1   # heavy volume, price breaks well below support

    total = absorbed + distributed
    if total == 0:
        return 0.5  # no heavy-volume bars → neutral

    return round(absorbed / total, 4)


# ── 2. Base quality score ──────────────────────────────────────────────────────

def base_quality_score(
    bars: Sequence[Bar],
    min_weeks: int = 12,
    bars_per_week: int = 5,
    max_range_pct: float = 0.15,
    atr_contraction_ratio: float = 0.75,
    proximity_floor: float = 0.85,
) -> float:
    """
    Score the tightness, duration, and high-proximity of a consolidation base.

    A high score indicates a long, tight base where price is coiling near
    its highs — the "cause being built" phase in Wyckoff terminology.

    Four sub-scores combined with equal weight:

        duration_score:   weeks_in_range / target_weeks  (capped at 1.0)
        tightness_score:  1 - (actual_range_pct / max_range_pct)
        atr_score:        1 - (current_atr / early_atr)  — how much ATR contracted
        proximity_score:  close / period_high  — price near the top of the base

    Args:
        bars:                OHLCV sequence, oldest first.
        min_weeks:           Target base duration in weeks (default 12 ≈ 3 months).
        bars_per_week:       Trading days per week (default 5).
        max_range_pct:       Maximum (high-low)/low to qualify as "tight" (default 15%).
        atr_contraction_ratio: Target ratio of current ATR to early-period ATR.
        proximity_floor:     Minimum close/period_high to score positively.

    Returns:
        Float in [0.0, 1.0].  >0.65 = well-formed base.  <0.35 = wide or short.
    """
    bars = list(bars)
    min_bars = min_weeks * bars_per_week
    if len(bars) < max(min_bars // 2, 20):
        return 0.0

    # Duration: how many bars has price been in range?
    period_high = max(b.high for b in bars)
    period_low  = min(b.low  for b in bars)
    range_pct = (period_high - period_low) / period_low if period_low > 0 else 1.0

    # Count bars where price stayed within the base range (within max_range_pct of its high)
    base_threshold = period_high * (1 - max_range_pct)
    bars_in_range = sum(1 for b in bars if b.low >= base_threshold)
    duration_score = min(1.0, bars_in_range / min_bars)

    # Tightness: how narrow is the range?
    tightness_score = max(0.0, 1.0 - (range_pct / max_range_pct))

    # ATR contraction: current ATR vs early-period ATR
    atr_now = _atr(bars, period=14)
    early_bars = bars[:min(30, len(bars) // 2)]
    atr_early = _atr(early_bars + [bars[len(early_bars)]], period=min(14, len(early_bars)))
    if atr_now is not None and atr_early is not None and atr_early > 0:
        contraction = 1.0 - (atr_now / atr_early)
        # Positive contraction = ATR shrinking; target >= (1 - atr_contraction_ratio)
        atr_score = max(0.0, min(1.0, contraction / (1.0 - atr_contraction_ratio)))
    else:
        atr_score = 0.5

    # Proximity: close relative to period high
    latest_close = bars[-1].close
    proximity_raw = latest_close / period_high if period_high > 0 else 0.0
    if proximity_raw >= proximity_floor:
        proximity_score = (proximity_raw - proximity_floor) / (1.0 - proximity_floor)
    else:
        proximity_score = 0.0

    composite = (duration_score + tightness_score + atr_score + proximity_score) / 4.0
    return round(max(0.0, min(1.0, composite)), 4)


# ── 3. Volume character score ──────────────────────────────────────────────────

def volume_character_score(
    bars: Sequence[Bar],
    lookback: int = 60,
    volume_period: int = 20,
) -> float:
    """
    Score the volume character of a base: up-days on heavy volume, down-days on light.

    Wyckoff accumulation fingerprint: institutions buy on weakness (up the price)
    and step back on down-days (letting price drift on low volume).  The result is
    above-average volume on advances and below-average volume on declines.

    Distribution is the mirror: down-days on heavy volume, up-days on thin volume.

    Algorithm:
        For each bar in the lookback window:
          accumulation_bar: close > open AND volume > avg_volume
          distribution_bar: close < open AND volume > avg_volume

        raw_score = (accumulation_bars - distribution_bars) / total_bars
        Normalised from [-1, 1] → [0.0, 1.0] via (raw + 1) / 2.

    Args:
        bars:          OHLCV sequence, oldest first.
        lookback:      Number of recent bars to evaluate.
        volume_period: Rolling period for average volume baseline.

    Returns:
        Float in [0.0, 1.0]. >0.55 = net accumulation character.
                              <0.45 = net distribution character.
                               0.50 = neutral / mixed.
    """
    bars = list(bars)
    if len(bars) < volume_period + 2:
        return 0.5

    window = bars[-lookback:] if len(bars) >= lookback else bars
    avg_vols = _rolling_avg_volume(bars, volume_period)
    offset = len(bars) - len(window)
    window_avg_vols = avg_vols[offset:]

    accumulation = 0
    distribution = 0

    for i, bar in enumerate(window):
        avg_vol = window_avg_vols[i]
        if avg_vol <= 0:
            continue
        is_up_day   = bar.close > bar.open
        is_down_day = bar.close < bar.open
        above_avg   = bar.volume > avg_vol

        if is_up_day and above_avg:
            accumulation += 1
        elif is_down_day and above_avg:
            distribution += 1

    total = len(window)
    if total == 0:
        return 0.5

    raw = (accumulation - distribution) / total   # in [-1.0, 1.0]
    return round((raw + 1.0) / 2.0, 4)            # normalised to [0.0, 1.0]


# ── 4. Wyckoff spring detector ─────────────────────────────────────────────────

def is_wyckoff_spring(
    bars: Sequence[Bar],
    lookback: int = 60,
    undercut_pct: float = 0.015,
    recovery_bars: int = 3,
    volume_confirmation: bool = True,
    volume_threshold: float = 1.2,
    volume_period: int = 20,
) -> bool:
    """
    Detect a Wyckoff spring: a brief undercut of base support followed by
    recovery back above the base lows within a few bars.

    The spring is engineered by operators to trigger stop-loss orders below
    support, absorb the resulting supply at lower prices, then reverse.
    Identifying it after the fact confirms accumulation is complete and
    markup is imminent.

    Detection criteria:
        1. Establish the base support level: min(low) over the lookback window
           excluding the most recent `recovery_bars` bars.
        2. Find any bar in the recent `recovery_bars` window whose low dips
           below support by at least `undercut_pct` (default 1.5%).
        3. That bar (or one within `recovery_bars`) must close back above
           the support level (recovery).
        4. Optional: the spring bar's volume is above the N-day average
           (confirms shakeout, not a genuine breakdown).

    Args:
        bars:                OHLCV sequence, oldest first.  Needs ≥ lookback bars.
        lookback:            Bars to use to establish the base support level.
        undercut_pct:        Minimum fraction below support to qualify (default 1.5%).
        recovery_bars:       Window in which undercut and recovery must occur.
        volume_confirmation: Require above-average volume on the spring bar.
        volume_threshold:    Volume multiple above average to qualify (default 1.2×).
        volume_period:       Rolling period for average volume baseline.

    Returns:
        True if a spring pattern is detected in the recent bars, False otherwise.
    """
    bars = list(bars)
    if len(bars) < lookback + recovery_bars:
        return False

    # Base support: minimum low over the lookback window (excluding recent window)
    base_window = bars[-(lookback + recovery_bars): -recovery_bars]
    if not base_window:
        return False
    support = min(b.low for b in base_window)

    # Recent window: where the spring should appear
    recent = bars[-recovery_bars:]
    avg_vols = _rolling_avg_volume(bars, volume_period)

    for i, bar in enumerate(recent):
        undercut = bar.low < support * (1.0 - undercut_pct)
        if not undercut:
            continue

        # Volume check on the spring bar
        bar_global_idx = len(bars) - recovery_bars + i
        avg_vol = avg_vols[bar_global_idx] if bar_global_idx < len(avg_vols) else 0.0
        if volume_confirmation and avg_vol > 0 and bar.volume < avg_vol * volume_threshold:
            continue  # undercut happened but on too little volume — not a spring

        # Recovery: this bar or a subsequent bar must close above support
        for recovery_bar in recent[i:]:
            if recovery_bar.close >= support:
                return True

    return False
