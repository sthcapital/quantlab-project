"""
quantlab.signals.earnings — Earnings acceleration detection from OHLCV bars.

No fundamental data is required.  The module identifies likely earnings dates
from overnight price gaps accompanied by volume spikes, then characterises
the pattern of post-earnings moves over time.

The core insight: stocks with *accelerating* post-earnings moves are under
active institutional re-rating — the kind of momentum that precedes large
breakout moves.  A rising EPS surprise magnitude, measured entirely from
price data, is a useful conviction filter even without fundamental data.

Functions:
    detect_earnings_dates(bars)       — gap + volume anomaly detection
    compute_earnings_profile(sym, bars) — full EarningsProfile from bars
    earnings_acceleration_score(profile) — 0.0–1.0 conviction input
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from quantlab.providers.base import Bar


# ── Rolling volume helper (local, avoids import coupling with wyckoff.py) ──────

def _rolling_avg_volume(bars: list[Bar], period: int = 20) -> list[float]:
    result: list[float] = []
    for i in range(len(bars)):
        window = [b.volume for b in bars[max(0, i - period):i]]
        result.append(sum(window) / len(window) if window else 0.0)
    return result


# ── EarningsProfile ────────────────────────────────────────────────────────────

@dataclass
class EarningsProfile:
    """Characterisation of a symbol's historical earnings-event pattern."""

    symbol: str
    earnings_dates: list[str]          # YYYY-MM-DD of each detected event

    # Frequency
    earnings_count: int                # number of detected events
    earnings_frequency: float          # events per year (ideal: ~4)

    # Magnitude
    avg_post_earnings_return: float    # mean |gap| across events (unsigned)
    avg_signed_return: float           # mean signed gap (positive = bullish bias)

    # Surprise direction
    positive_surprise_rate: float      # fraction of events with gap > +1%

    # Acceleration
    acceleration_trend: float          # (recent_half_avg - early_half_avg) / early
                                       # clipped to [-1, +1]; >0 = accelerating
    last_4_avg: float                  # mean |gap| of last 4 events
    prior_4_avg: float                 # mean |gap| of 4 events before those

    def is_accelerating(self, min_trend: float = 0.10) -> bool:
        """True when recent post-earnings moves are materially larger than prior ones."""
        return self.earnings_count >= 4 and self.acceleration_trend > min_trend

    def summary(self) -> str:
        return (
            f"{self.earnings_count} events  freq={self.earnings_frequency:.1f}/yr  "
            f"avg_gap={self.avg_post_earnings_return * 100:.1f}%  "
            f"+surprise={self.positive_surprise_rate * 100:.0f}%  "
            f"accel={self.acceleration_trend:+.2f}"
        )


# ── Earnings date detection ────────────────────────────────────────────────────

def detect_earnings_dates(
    bars: Sequence[Bar],
    gap_threshold: float = 0.025,
    max_gap: float = 0.20,
    vol_threshold: float = 1.5,
    vol_period: int = 20,
    min_event_spacing: int = 30,
) -> list[str]:
    """
    Identify likely earnings announcement dates from price/volume anomalies.

    Heuristic: a quarterly earnings event typically causes an overnight gap
    (|open − prev_close| / prev_close) larger than `gap_threshold`, combined
    with above-average volume on the event day.

    Events within `min_event_spacing` bars of a prior event are skipped to
    prevent clustering (one earnings release = one event).  Gaps larger than
    `max_gap` (default 20%) are excluded as likely splits or M&A rather than
    earnings.

    Args:
        bars:               OHLCV sequence, oldest first.
        gap_threshold:      Minimum |gap| fraction to qualify (default 2.5%).
        max_gap:            Maximum |gap| fraction; above this is not earnings.
        vol_threshold:      Minimum volume / 20-day avg to qualify (default 1.5×).
        vol_period:         Rolling average volume window.
        min_event_spacing:  Minimum bars between consecutive events (default 30).

    Returns:
        List of YYYY-MM-DD strings, one per detected earnings event, in
        chronological order.
    """
    bars = list(bars)
    if len(bars) < vol_period + 2:
        return []

    avg_vols = _rolling_avg_volume(bars, vol_period)
    events: list[str] = []
    last_event_bar = -(min_event_spacing + 1)

    for i in range(1, len(bars)):
        prev, curr = bars[i - 1], bars[i]

        if prev.close <= 0:
            continue

        gap = (curr.open - prev.close) / prev.close
        if abs(gap) < gap_threshold or abs(gap) > max_gap:
            continue

        avg_vol = avg_vols[i]
        if avg_vol <= 0 or curr.volume < avg_vol * vol_threshold:
            continue

        if i - last_event_bar < min_event_spacing:
            continue  # too close to prior event — same reporting season

        events.append(curr.as_of.isoformat())
        last_event_bar = i

    return events


# ── Full profile computation ───────────────────────────────────────────────────

def compute_earnings_profile(
    symbol: str,
    bars: Sequence[Bar],
    gap_threshold: float = 0.025,
    max_gap: float = 0.20,
    vol_threshold: float = 1.5,
) -> EarningsProfile:
    """
    Build a complete EarningsProfile from bar history.

    The gap on the first bar of an earnings event (open vs prior close) is
    used as the post-earnings return proxy.  A positive gap indicates a
    positive surprise; a negative gap indicates disappointment or a miss.

    Acceleration is measured by comparing the mean absolute gap in the
    second half of all detected events against the first half.

    Args:
        symbol:          Ticker symbol label.
        bars:            Full OHLCV history (oldest first). More bars = better
                         acceleration signal; minimum ~1 year for 4 events.
        gap_threshold:   Passed through to detect_earnings_dates().
        max_gap:         Passed through to detect_earnings_dates().
        vol_threshold:   Passed through to detect_earnings_dates().

    Returns:
        EarningsProfile with all metrics populated.  If fewer than 2 events
        are detected, most metrics default to 0.0.
    """
    bars = list(bars)
    _empty = EarningsProfile(
        symbol=symbol, earnings_dates=[],
        earnings_count=0, earnings_frequency=0.0,
        avg_post_earnings_return=0.0, avg_signed_return=0.0,
        positive_surprise_rate=0.0, acceleration_trend=0.0,
        last_4_avg=0.0, prior_4_avg=0.0,
    )

    dates = detect_earnings_dates(bars, gap_threshold, max_gap, vol_threshold)
    if not dates:
        return _empty

    date_to_idx = {b.as_of.isoformat(): i for i, b in enumerate(bars)}

    # Per-event gap (signed)
    gaps: list[float] = []
    for d in dates:
        idx = date_to_idx.get(d)
        if idx is None or idx == 0:
            continue
        gap = (bars[idx].open - bars[idx - 1].close) / bars[idx - 1].close
        gaps.append(gap)

    n = len(gaps)
    if n == 0:
        return _empty

    # Frequency
    if n >= 2:
        first_idx = date_to_idx.get(dates[0], 0)
        last_idx  = date_to_idx.get(dates[-1], len(bars) - 1)
        calendar_years = max(
            (bars[last_idx].as_of - bars[first_idx].as_of).days / 365.25,
            0.01,
        )
        frequency = n / calendar_years
    else:
        frequency = 0.0

    avg_abs    = sum(abs(g) for g in gaps) / n
    avg_signed = sum(gaps) / n
    pos_rate   = sum(1 for g in gaps if g > 0.01) / n

    # Acceleration: second half absolute magnitudes vs first half
    if n >= 4:
        half = n // 2
        first_abs = [abs(g) for g in gaps[:half]]
        second_abs = [abs(g) for g in gaps[half:]]
        early_avg  = sum(first_abs) / len(first_abs)
        recent_avg = sum(second_abs) / len(second_abs)

        raw = (recent_avg - early_avg) / early_avg if early_avg > 0 else 0.0
        trend = max(-1.0, min(1.0, raw))

        last_4  = [abs(g) for g in gaps[-4:]]
        prior_4 = [abs(g) for g in gaps[-8:-4]] if n >= 8 else first_abs[:4]
        last_4_avg  = sum(last_4)  / len(last_4)
        prior_4_avg = sum(prior_4) / len(prior_4) if prior_4 else 0.0
    else:
        trend = 0.0
        last_4_avg  = avg_abs
        prior_4_avg = 0.0

    return EarningsProfile(
        symbol=symbol,
        earnings_dates=dates,
        earnings_count=n,
        earnings_frequency=round(frequency, 2),
        avg_post_earnings_return=round(avg_abs, 6),
        avg_signed_return=round(avg_signed, 6),
        positive_surprise_rate=round(pos_rate, 4),
        acceleration_trend=round(trend, 4),
        last_4_avg=round(last_4_avg, 6),
        prior_4_avg=round(prior_4_avg, 6),
    )


# ── Conviction score ───────────────────────────────────────────────────────────

def earnings_acceleration_score(profile: EarningsProfile) -> float:
    """
    Return a 0.0–1.0 score for earnings acceleration quality.

    Requires at least 4 detected events (~1 year of quarterly data).
    Three independent sub-scores are summed and clamped to 1.0:

        Positive surprise rate > 60%     : +0.35  (majority of events bullish)
        Positive surprise rate 50–60%    : +0.20
        Acceleration trend > +10%        : +0.35  (recent magnitudes growing)
        Acceleration trend > 0           : +0.20
        Avg gap magnitude > 5%           : +0.30  (large movers = high attention)
        Avg gap magnitude 3–5%           : +0.15

    A score ≥ 0.5 triggers the +0.10 conviction boost in score_conviction().

    Args:
        profile: EarningsProfile returned by compute_earnings_profile().

    Returns:
        Float in [0.0, 1.0].
    """
    if profile.earnings_count < 4:
        return 0.0

    score = 0.0

    if profile.positive_surprise_rate > 0.60:
        score += 0.35
    elif profile.positive_surprise_rate > 0.50:
        score += 0.20

    if profile.acceleration_trend > 0.10:
        score += 0.35
    elif profile.acceleration_trend > 0.0:
        score += 0.20

    if profile.avg_post_earnings_return > 0.05:
        score += 0.30
    elif profile.avg_post_earnings_return > 0.03:
        score += 0.15

    return round(min(1.0, score), 4)
