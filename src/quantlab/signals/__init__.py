"""
Layer 4: Signal generation.

All signals operate on sequences of Bar objects and return SignalResult
dataclasses — no side effects, no I/O, pure computation.

Current signals:
    SmaSignal      — price above N-day simple moving average
    BreakoutSignal — close above N-day prior high (with optional volume filter)
    RegimeFilter   — SPY SMA trend filter (long only when market above its SMA)

Coming in Phase 4:
    AtrStop        — Average True Range based stop price calculation
    RelVolumeFilter — relative volume vs N-day average
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from quantlab.providers.base import Bar


# ── Signal result ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SignalResult:
    """Output of a signal check for a single bar."""

    date: str
    symbol: str
    signal: bool            # True = setup triggered
    signal_type: str        # e.g. "sma", "breakout"
    entry_close: float
    indicator_value: float | None   # SMA level, prior high, etc.
    lookback: int


# ── Indicator helpers ─────────────────────────────────────────────────────────

def sma(values: Sequence[float], window: int) -> float | None:
    if len(values) < window:
        return None
    return sum(values[-window:]) / window


def atr(bars: Sequence[Bar], period: int = 14) -> float | None:
    """Average True Range over `period` bars."""
    if len(bars) < period + 1:
        return None
    true_ranges = [bars[i].true_range(bars[i - 1]) for i in range(1, len(bars))]
    recent = true_ranges[-period:]
    return sum(recent) / len(recent)


def relative_volume(bars: Sequence[Bar], period: int = 20) -> float | None:
    """Today's volume divided by N-day average volume."""
    if len(bars) < period + 1:
        return None
    avg_vol = sum(b.volume for b in bars[-period - 1:-1]) / period
    if avg_vol == 0:
        return None
    return bars[-1].volume / avg_vol


# ── Signal generators ─────────────────────────────────────────────────────────

def sma_signal(bars: Sequence[Bar], symbol: str, lookback: int = 20) -> SignalResult | None:
    """
    SMA crossover signal: fire when close > N-day SMA.

    Requires at least lookback + 1 bars. Uses bars[:-1] for the SMA
    (prior day) and bars[-1].close as today's close to avoid lookahead.
    """
    if len(bars) <= lookback:
        return None

    closes = [b.close for b in bars]
    avg = sma(closes[:-1], lookback)

    if avg is None:
        return None

    latest = bars[-1]
    return SignalResult(
        date=latest.as_of.isoformat(),
        symbol=symbol,
        signal=latest.close > avg,
        signal_type="sma",
        entry_close=latest.close,
        indicator_value=avg,
        lookback=lookback,
    )


def breakout_signal(
    bars: Sequence[Bar],
    symbol: str,
    lookback: int = 20,
    min_rel_volume: float | None = None,
) -> SignalResult | None:
    """
    N-day high breakout signal: fire when close > max(high) over prior N bars.

    Args:
        bars:           Sequence of Bar objects, at least lookback + 1 bars.
        symbol:         Ticker symbol label.
        lookback:       Number of prior bars to compute the high over.
        min_rel_volume: Optional minimum relative volume (e.g. 1.5 = 150% of avg).
                        If set and today's rel vol is below threshold, signal is False.
    """
    if len(bars) <= lookback:
        return None

    prior_bars = bars[-lookback - 1:-1]
    prior_high = max(b.high for b in prior_bars)
    latest = bars[-1]

    triggered = latest.close > prior_high

    # Optional relative volume filter
    if triggered and min_rel_volume is not None:
        rv = relative_volume(bars, period=20)
        if rv is not None and rv < min_rel_volume:
            triggered = False

    return SignalResult(
        date=latest.as_of.isoformat(),
        symbol=symbol,
        signal=triggered,
        signal_type="breakout",
        entry_close=latest.close,
        indicator_value=prior_high,
        lookback=lookback,
    )


def regime_is_bullish(market_bars: Sequence[Bar], sma_period: int = 200) -> bool:
    """
    Simple market regime filter: returns True when the market index
    (SPY) is above its N-day SMA. Only take long signals in bull regime.
    """
    if len(market_bars) <= sma_period:
        return True  # default to bullish if not enough data
    closes = [b.close for b in market_bars]
    avg = sma(closes[:-1], sma_period)
    return avg is not None and market_bars[-1].close > avg


# ── ATR stop calculator ───────────────────────────────────────────────────────

def atr_stop_price(
    bars: Sequence[Bar],
    entry_price: float,
    atr_period: int = 14,
    atr_multiplier: float = 2.0,
) -> float | None:
    """
    Calculate an ATR-based stop loss price.

    stop = entry_price - (ATR * multiplier)

    A multiplier of 2.0 means the trade is allowed to move against entry
    by 2x the recent average daily range before being stopped out.
    """
    avg_true_range = atr(bars, atr_period)
    if avg_true_range is None:
        return None
    return entry_price - (avg_true_range * atr_multiplier)
