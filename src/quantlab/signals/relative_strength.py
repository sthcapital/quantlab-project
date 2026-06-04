"""
quantlab.signals.relative_strength — Price-based relative strength vs market.

Measures whether a stock is leading or lagging the broader market over
multiple timeframes.  True market leadership requires consistent outperformance
on both intermediate (63-day / 3-month) and longer (126-day / 6-month) windows.

A stock that is breaking out from a base while the market is going sideways
or declining has earned its move.  A stock breaking out while everything is
rising may simply be moving with the tide.

Functions:
    rs_score(symbol_bars, market_bars, periods=[63, 126])
        Per-symbol relative strength score 0.0–1.0.
        0.5 = matched market exactly.  >0.6 = outperforming.  >0.8 = leadership.

    rs_rank(symbol_bars_dict, market_bars)
        Rank a universe of symbols by RS score.
        Returns percentile rankings 0–100 (100 = strongest relative strength).
"""

from __future__ import annotations

import math
from typing import Sequence

from quantlab.providers.base import Bar


# ── Core RS calculation ────────────────────────────────────────────────────────

def rs_score(
    symbol_bars: Sequence[Bar],
    market_bars: Sequence[Bar],
    periods: list[int] | None = None,
    normalization: float = 0.15,
) -> float:
    """
    Compute a symbol's relative strength vs a market benchmark.

    For each lookback period, calculates:

        excess_return = symbol_return(period) − market_return(period)

    Maps each excess return to [0, 1] via tanh normalisation and averages
    across all periods.

        score = 0.5 + 0.5 × tanh(excess / normalization)

    Reference points with default normalization=0.15:
        excess =   0%   →  score 0.50  (matched market)
        excess =  +5%   →  score 0.66  (moderate outperformance)
        excess = +10%   →  score 0.78  (strong outperformance)
        excess = +15%   →  score 0.88  (leadership)
        excess = +20%   →  score 0.93  (dominant leader)
        excess =  −5%   →  score 0.34  (moderate underperformance)
        excess = −15%   →  score 0.12  (consistent laggard)

    Args:
        symbol_bars:   OHLCV bar sequence for the symbol, oldest first.
        market_bars:   OHLCV bar sequence for the benchmark (e.g. SPY),
                       oldest first.  Should cover at least max(periods) bars.
        periods:       Lookback windows in bars.  Default [63, 126]
                       (≈ 3 months and 6 months).
        normalization: Excess return that maps to score ≈ 0.67 (tanh scaling).
                       Default 0.15 = 15%.

    Returns:
        Float in [0.0, 1.0].  Returns 0.5 (neutral) when either bar sequence
        is too short for any requested period.
    """
    if periods is None:
        periods = [63, 126]

    symbol_bars = list(symbol_bars)
    market_bars = list(market_bars)

    period_scores: list[float] = []

    for period in periods:
        # Need at least period + 1 bars so bars[-period] and bars[-1] exist
        if len(symbol_bars) < period + 1 or len(market_bars) < period + 1:
            period_scores.append(0.5)   # neutral when insufficient history
            continue

        sym_ret = (symbol_bars[-1].close / symbol_bars[-period].close) - 1.0
        mkt_ret = (market_bars[-1].close / market_bars[-period].close) - 1.0

        excess = sym_ret - mkt_ret
        score  = 0.5 + 0.5 * math.tanh(excess / normalization)
        period_scores.append(round(max(0.0, min(1.0, score)), 6))

    return round(sum(period_scores) / len(period_scores), 4) if period_scores else 0.5


# ── Universe ranking ───────────────────────────────────────────────────────────

def rs_rank(
    symbol_bars_dict: dict[str, Sequence[Bar]],
    market_bars: Sequence[Bar],
    periods: list[int] | None = None,
) -> dict[str, float]:
    """
    Rank a universe of symbols by relative strength and return percentile scores.

    Computes rs_score() for every symbol, then assigns percentile ranks:
    100 = strongest RS in universe, 0 = weakest, 50 = median.

    The top quartile (RS rank > 75) is the target zone for breakout setups —
    stocks that are already proving themselves relative to the market.

    Args:
        symbol_bars_dict: Dict mapping symbol → bar sequence.
        market_bars:      Benchmark bars (e.g. SPY).
        periods:          Passed through to rs_score().

    Returns:
        Dict {symbol: percentile_rank}.  Empty dict when input is empty.

    Example::

        ranks = rs_rank({"AAPL": aapl_bars, "XOM": xom_bars, ...}, spy_bars)
        leaders = {sym: rank for sym, rank in ranks.items() if rank > 75}
    """
    if periods is None:
        periods = [63, 126]

    if not symbol_bars_dict:
        return {}

    # Compute raw scores
    scores: dict[str, float] = {
        sym: rs_score(bars, market_bars, periods)
        for sym, bars in symbol_bars_dict.items()
    }

    # Assign percentile ranks (0 = lowest, 100 = highest)
    sorted_syms = sorted(scores, key=lambda s: scores[s])
    n = len(sorted_syms)

    ranks: dict[str, float] = {}
    for i, sym in enumerate(sorted_syms):
        pct = (i / (n - 1) * 100.0) if n > 1 else 50.0
        ranks[sym] = round(pct, 1)

    return ranks
