"""
Layer 7: Execution infrastructure — market scanner and conviction scoring.

The scanner is the missing piece identified in the architecture review.
It loops over a symbol universe, runs signal checks on each, layers
confirmation signals (news, regime), scores conviction, and returns
a ranked list of setups ready for the risk gate.

ZeroMQ pub/sub bus is stubbed here for Phase 5+ — the scanner publishes
signals, the execution subscriber receives and validates before order submission.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Sequence

from quantlab.providers.base import Bar, MarketDataProvider
from quantlab.signals import (
    SignalResult,
    breakout_signal,
    sma_signal,
    regime_is_bullish,
    atr_stop_price,
)

logger = logging.getLogger(__name__)


# ── Conviction scoring ─────────────────────────────────────────────────────────

@dataclass
class ScanResult:
    """
    Output of the scanner for a single symbol.
    Carries everything needed for the risk gate decision.
    """

    symbol: str
    scan_date: str
    signal_type: str
    signal: bool
    entry_close: float
    indicator_value: float | None
    lookback: int

    # Conviction layers
    regime_bullish: bool = True
    news_count: int = 0
    news_category: str = "none"
    news_k_score: float | None = None
    news_c_score: float | None = None
    rel_volume: float | None = None
    atr_stop: float | None = None

    # Computed conviction score (0.0 – 1.0)
    conviction_score: float = 0.0

    def is_actionable(self, min_conviction: float = 0.4) -> bool:
        return self.signal and self.conviction_score >= min_conviction


def score_conviction(result: ScanResult) -> float:
    """
    Score 0.0–1.0 based on how many confirmation layers align.

    Scoring weights (total = 1.0):
        Signal fired             : 0.30  (mandatory — 0 if no signal)
        Regime bullish           : 0.20
        Recent news present      : 0.15
        News is earnings/upgrade : 0.15  (bonus for high-quality catalyst)
        Rel volume > 1.5x        : 0.10
        Strong news confidence   : 0.10  (IBKR C: score > 0.7)
    """
    if not result.signal:
        return 0.0

    score = 0.30  # signal fired

    if result.regime_bullish:
        score += 0.20

    if result.news_count > 0:
        score += 0.15
        if result.news_category in {"upgrade", "earnings"}:
            score += 0.15

    if result.rel_volume is not None and result.rel_volume >= 1.5:
        score += 0.10

    if result.news_c_score is not None and result.news_c_score >= 0.7:
        score += 0.10

    return min(score, 1.0)


# ── Universe management ────────────────────────────────────────────────────────

# Default watchlists — will grow as the system matures
SP500_SAMPLE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK.B",
    "UNH", "LLY", "JPM", "V", "XOM", "MA", "AVGO", "PG", "HD", "CVX",
    "MRK", "COST", "ABBV", "KO", "PEP", "BAC", "ADBE", "TMO", "WMT",
    "ACN", "MCD", "CSCO", "ABT", "CRM", "NFLX", "DHR", "ORCL", "NKE",
    "LIN", "TXN", "NEE", "RTX", "BMY", "AMGN", "UPS", "HON", "PM",
    "INTC", "QCOM", "IBM", "CAT", "GS",
]

WATCHLIST_SMALL = ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "GOOGL", "META"]


def load_universe(name: str = "small") -> list[str]:
    """
    Return a symbol list by name.

    Args:
        name: "small" (7 symbols for testing), "sp500_sample" (50 symbols),
              or pass a comma-separated string like "AAPL,MSFT,NVDA"
    """
    if name == "small":
        return WATCHLIST_SMALL
    if name == "sp500_sample":
        return SP500_SAMPLE
    # custom comma-separated
    return [s.strip().upper() for s in name.split(",") if s.strip()]


# ── Scanner ───────────────────────────────────────────────────────────────────

def scan_symbol(
    symbol: str,
    bars: Sequence[Bar],
    signal_type: str = "breakout",
    lookback: int = 20,
    regime_bars: Sequence[Bar] | None = None,
    news_features=None,
    min_rel_volume: float | None = 1.5,
) -> ScanResult | None:
    """
    Run all signal and confirmation checks for a single symbol.

    Args:
        symbol:         Ticker symbol.
        bars:           Recent bar history (at least lookback + 1 bars).
        signal_type:    "breakout" or "sma".
        lookback:       Signal lookback period.
        regime_bars:    SPY bars for regime filter (optional).
        news_features:  NewsFeatures dataclass from quantlab.news (optional).
        min_rel_volume: Minimum relative volume threshold.

    Returns:
        ScanResult with conviction_score, or None if not enough bars.
    """
    from quantlab.signals import relative_volume as _rel_vol

    if len(bars) <= lookback:
        logger.debug(f"{symbol}: not enough bars ({len(bars)} <= {lookback})")
        return None

    # Run primary signal
    if signal_type == "breakout":
        signal_result = breakout_signal(bars, symbol, lookback, min_rel_volume=None)
    elif signal_type == "sma":
        signal_result = sma_signal(bars, symbol, lookback)
    else:
        raise ValueError(f"Unknown signal_type: {signal_type}")

    if signal_result is None:
        return None

    today = bars[-1].as_of.isoformat()

    # Regime filter
    bullish = True
    if regime_bars and len(regime_bars) > 200:
        bullish = regime_is_bullish(regime_bars, sma_period=200)

    # Relative volume
    rv = _rel_vol(bars, period=20)

    # ATR stop
    stop = atr_stop_price(bars, signal_result.entry_close)

    # News features
    n_count = 0
    n_cat = "none"
    k_score = None
    c_score = None
    if news_features is not None:
        n_count = news_features.total_count
        n_cat = news_features.dominant_category
        k_score = news_features.avg_k_score
        c_score = news_features.avg_c_score

    result = ScanResult(
        symbol=symbol,
        scan_date=today,
        signal_type=signal_type,
        signal=signal_result.signal,
        entry_close=signal_result.entry_close,
        indicator_value=signal_result.indicator_value,
        lookback=lookback,
        regime_bullish=bullish,
        news_count=n_count,
        news_category=n_cat,
        news_k_score=k_score,
        news_c_score=c_score,
        rel_volume=rv,
        atr_stop=stop,
    )

    result.conviction_score = score_conviction(result)
    return result


def run_universe_scan(
    provider: MarketDataProvider,
    symbols: list[str],
    start_date: date,
    end_date: date,
    signal_type: str = "breakout",
    lookback: int = 20,
    min_conviction: float = 0.4,
    cost_bps: float = 10.0,
    ibkr_connection=None,  # live IB() instance for news; None = skip news
) -> list[ScanResult]:
    """
    Scan a universe of symbols and return ranked actionable setups.

    This is the top-level entry point for the daily pre-market scan.
    Results are sorted by conviction_score descending.

    Args:
        provider:       MarketDataProvider instance.
        symbols:        List of ticker symbols to scan.
        start_date:     Bar history start date.
        end_date:       Bar history end date (today for live scans).
        signal_type:    "breakout" or "sma".
        lookback:       Signal lookback period in bars.
        min_conviction: Minimum score to include in results.
        cost_bps:       Transaction cost (for display/logging).
        ibkr_connection: Live IB() for news fetching (None = price-only scan).

    Returns:
        List of ScanResult sorted by conviction_score descending.
    """
    from quantlab.news import fetch_news, compute_news_features
    from quantlab.storage import append_trades_to_db
    from datetime import datetime

    results = []
    total = len(symbols)

    for i, symbol in enumerate(symbols, 1):
        try:
            logger.info(f"[{i}/{total}] Scanning {symbol}...")

            bars = provider.get_daily_bars(symbol, start_date, end_date)
            if not bars:
                logger.warning(f"{symbol}: no bars returned")
                continue

            # Fetch news if IBKR connection is provided
            news_feat = None
            if ibkr_connection is not None:
                try:
                    from ib_insync import Stock as _Stock
                    contract = _Stock(symbol, "SMART", "USD")
                    qualified = ibkr_connection.qualifyContracts(contract)
                    if qualified:
                        news_items = fetch_news(ibkr_connection, qualified[0], days=30, limit=50)
                        news_feat = compute_news_features(
                            news_items,
                            end_date.isoformat(),
                            lookback_days=7,
                        )
                except Exception as e:
                    logger.debug(f"{symbol} news fetch failed: {e}")

            result = scan_symbol(
                symbol=symbol,
                bars=bars,
                signal_type=signal_type,
                lookback=lookback,
                news_features=news_feat,
            )

            if result is not None:
                results.append(result)

        except Exception as e:
            logger.error(f"{symbol}: scan error — {e}")
            continue

    # Sort by conviction score, highest first
    results.sort(key=lambda r: r.conviction_score, reverse=True)

    # Filter to actionable
    actionable = [r for r in results if r.is_actionable(min_conviction)]

    logger.info(
        f"Scan complete: {total} symbols → {len(results)} processed → "
        f"{len(actionable)} actionable (min_conviction={min_conviction})"
    )

    return actionable


# ── ZeroMQ stub (Phase 5+) ─────────────────────────────────────────────────────

class SignalPublisher:
    """
    ZeroMQ publisher stub for Phase 5+ live execution.

    The scanner publishes ScanResult objects on a ZMQ PUB socket.
    The execution engine subscribes on a SUB socket and validates
    before submitting orders.

    This is a stub — implement when moving to paper trading.
    """

    def __init__(self, address: str = "tcp://127.0.0.1:5555") -> None:
        self.address = address
        self._socket = None

    def connect(self) -> None:
        try:
            import zmq
            ctx = zmq.Context()
            self._socket = ctx.socket(zmq.PUB)
            self._socket.bind(self.address)
            logger.info(f"Signal publisher bound to {self.address}")
        except ImportError:
            logger.warning("pyzmq not installed — SignalPublisher is a no-op")

    def publish(self, result: ScanResult) -> None:
        if self._socket is None:
            return
        import json
        payload = json.dumps(result.__dict__, default=str)
        self._socket.send_string(f"SIGNAL {payload}")

    def close(self) -> None:
        if self._socket:
            self._socket.close()
