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


# ── Stage classification (Weinstein / Minervini) ─────────────────────────────

def stage_classification(bars: Sequence[Bar], ma_period: int = 150) -> int:
    """
    Classify a stock into one of four Weinstein/Minervini market stages.

    Requires at least ma_period + 40 bars for reliable classification.
    Returns 0 (undetermined) when insufficient data is available.

    Stages:
        1 — Basing:    price near 52W low, flat MA, volume contracting.
                       These go on the weekend watchlist as base candidates.
        2 — Advancing: above rising MA, higher highs and higher lows.
                       ONLY Stage 2 stocks are valid long-entry candidates.
        3 — Topping:   extended above MA, lower highs forming, or MA flattening.
                       Distribution phase — reduce or avoid new entries.
        4 — Declining: below declining MA, lower highs and lower lows.
                       Mark-down phase — do not enter long.

    Args:
        bars:      Daily bar sequence, oldest first.
        ma_period: Moving average period in trading days (150 ≈ 30 weeks).

    Returns:
        int 1–4, or 0 when data is insufficient.
    """
    bars = list(bars)
    if len(bars) < ma_period + 40:
        return 0

    closes  = [b.close for b in bars]
    highs   = [b.high  for b in bars]
    lows    = [b.low   for b in bars]
    volumes = [b.volume for b in bars]

    close_now = closes[-1]

    # 30-week MA and its direction (compared to 20 bars ago)
    ma_now  = sum(closes[-ma_period:]) / ma_period
    ma_prev = sum(closes[-ma_period - 20:-20]) / ma_period

    # 52-week reference points
    lookback = min(252, len(bars))
    high_52w = max(closes[-lookback:])
    low_52w  = min(closes[-lookback:])

    # Recent vs prior swing highs/lows (last 20 bars vs 20-40 bars ago)
    recent_high = max(highs[-20:])
    prior_high  = max(highs[-40:-20])
    recent_low  = min(lows[-20:])
    prior_low   = min(lows[-40:-20])

    # Volume trend: recent 20-bar avg vs prior 20-bar avg
    recent_vol = sum(volumes[-20:]) / 20
    prior_vol  = sum(volumes[-40:-20]) / 20
    vol_declining = (prior_vol > 0) and (recent_vol < prior_vol * 0.90)

    above_ma   = close_now > ma_now
    ma_rising  = ma_now > ma_prev

    # ── Stage 2: Advancing — the only valid long-entry stage ─────────────────
    if above_ma and ma_rising and recent_high > prior_high and recent_low > prior_low:
        return 2

    # Also classify as Stage 2 when clearly above a rising MA even without
    # confirmed HH/HL yet (early Stage 2 — price just broke above MA)
    if above_ma and ma_rising and close_now > ma_now * 1.05:
        return 2

    # ── Stage 1: Basing near lows — weekend watchlist candidates ─────────────
    if (close_now <= low_52w * 1.15          # within 15% of 52W low
            and abs(close_now - ma_now) / ma_now < 0.10   # within 10% of MA
            and vol_declining):               # volume contracting = sellers done
        return 1

    # ── Stage 4: Declining — below declining MA, lower highs and lows ────────
    if (not above_ma and not ma_rising
            and recent_high < prior_high and recent_low < prior_low):
        return 4

    # Partial Stage 4: below a declining MA without confirmed lower lows yet
    if not above_ma and not ma_rising:
        return 4

    # ── Stage 3: Topping — extended above MA or lower highs forming ──────────
    if above_ma:
        extended = close_now > ma_now * 1.15          # >15% above 30W MA
        lower_highs = not ma_rising and recent_high < prior_high
        if extended or lower_highs:
            return 3
        return 2   # above rising MA, not extended → still Stage 2

    return 0   # undetermined


# ── Breakout volume quality (Weinstein) ───────────────────────────────────────

def volume_on_breakout_score(bars: Sequence[Bar], period: int = 20) -> float:
    """
    Score breakout volume quality per Weinstein's 2× minimum rule.

    Weinstein's rule: volume must be at least 2× the 4-week (20-day) average
    on the breakout week/bar for a technically valid breakout.

    Args:
        bars:   Daily bar sequence; the last bar is the potential breakout bar.
        period: Comparison period (default 20 = 4 trading weeks).

    Returns:
        0.0 — volume < 1× avg   (false breakout likely — avoid)
        0.3 — volume 1–2× avg   (weak breakout — below Weinstein minimum)
        0.7 — volume 2–3× avg   (valid breakout — Weinstein minimum met)
        1.0 — volume > 3× avg   (institutional conviction — best breakouts)
    """
    if len(bars) < period + 1:
        return 0.0
    avg_vol = sum(b.volume for b in bars[-period - 1:-1]) / period
    if avg_vol == 0:
        return 0.0
    ratio = bars[-1].volume / avg_vol
    if ratio < 1.0:
        return 0.0
    elif ratio < 2.0:
        return 0.3
    elif ratio < 3.0:
        return 0.7
    else:
        return 1.0


# ── Volume dry-up score (Kell / Darvas) ──────────────────────────────────────

def volume_dry_up_score(bars: Sequence[Bar], window: int = 10) -> float:
    """
    Score volume contraction during a base formation (Kell/Darvas dry-up).

    Compares average volume over the last `window` bars to the prior `window`
    bars.  Declining volume inside a base indicates seller exhaustion — the
    "volume dry-up" that precedes the best Darvas-box breakouts.

    Args:
        bars:   Daily bar sequence.
        window: Comparison window in bars (default 10 ≈ 2 trading weeks).

    Returns:
        1.0 — volume declining 30%+  (sellers exhausted — ideal Darvas setup)
        0.6 — volume declining 15–30%
        0.3 — volume flat (within ±15%)
        0.0 — volume increasing in base (distribution — not accumulation)
    """
    if len(bars) < window * 2:
        return 0.0
    recent_avg = sum(b.volume for b in bars[-window:]) / window
    prior_avg  = sum(b.volume for b in bars[-window * 2:-window]) / window
    if prior_avg == 0:
        return 0.0
    decline_pct = 1.0 - (recent_avg / prior_avg)   # positive = declining
    if decline_pct >= 0.30:
        return 1.0
    elif decline_pct >= 0.15:
        return 0.6
    elif decline_pct >= -0.15:
        return 0.3
    else:
        return 0.0   # volume increasing → distribution risk


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
