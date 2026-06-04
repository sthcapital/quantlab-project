"""
Layer 4: Backtesting and research engine.

BacktestResult contains the full institutional-grade metric set discussed
in the Perplexity chat:
    - Total return, max drawdown
    - Sharpe, Sortino, Calmar
    - Profit factor, expectancy per trade
    - Win rate, avg win, avg loss (win/loss size ratio)
    - Exposure (% of time in market)
    - Trade log with forward returns, MFE, MAE

All computation is pure Python — no pandas required at this layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from quantlab.providers.base import Bar


# ── Trade record ──────────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    """Full record of a single completed or open trade."""

    symbol: str
    signal_date: str
    entry_date: str
    entry_price: float
    exit_date: str | None
    exit_price: float | None
    trade_return: float | None      # (exit / entry) - 1
    ret_1d: float | None            # 1-day forward return
    ret_3d: float | None
    ret_5d: float | None
    mfe_5d: float | None            # max favorable excursion over 5 bars
    mae_5d: float | None            # max adverse excursion over 5 bars
    atr_stop: float | None          # ATR-based stop price at entry
    news_category: str = "none"
    news_count: int = 0
    news_k_score: float | None = None
    news_c_score: float | None = None
    cost_bps: float = 0.0           # round-trip transaction cost in basis points


# ── Performance metrics ───────────────────────────────────────────────────────

@dataclass
class PerformanceMetrics:
    """Full institutional-grade metric set."""

    symbol: str
    signal_type: str
    lookback: int
    bar_count: int
    trade_count: int

    # Return metrics
    total_return: float
    annualised_return: float
    max_drawdown: float
    calmar_ratio: float | None      # annualised_return / abs(max_drawdown)

    # Risk-adjusted return
    sharpe_ratio: float | None      # annualised (rf=0)
    sortino_ratio: float | None     # uses downside deviation

    # Trade quality
    win_rate: float
    avg_win: float | None
    avg_loss: float | None
    win_loss_ratio: float | None    # avg_win / abs(avg_loss)
    profit_factor: float | None     # gross profit / gross loss
    expectancy: float | None        # (win_rate * avg_win) - (loss_rate * abs(avg_loss))
    avg_trade_return: float
    exposure_pct: float             # % of bars in market

    # Sample quality
    sufficient_sample: bool         # True when trade_count >= MIN_TRADES


MIN_TRADES = 30  # flagged in chat as minimum for statistical significance
ANNUALISE_FACTOR = 252


def _safe_div(a: float, b: float) -> float | None:
    if b == 0:
        return None
    return a / b


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def compute_metrics(
    symbol: str,
    signal_type: str,
    lookback: int,
    bars: Sequence[Bar],
    trades: list[TradeRecord],
    equity_curve: list[float],
    strategy_returns: list[float],
    positions: list[int],
) -> PerformanceMetrics:
    """Compute the full institutional metric set from backtest outputs."""

    n_bars = len(bars)
    n_trades = len(trades)

    # Total and annualised return
    initial = equity_curve[0]
    final = equity_curve[-1]
    total_ret = (final / initial) - 1.0
    years = n_bars / ANNUALISE_FACTOR
    ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0.0

    # Max drawdown
    peak = equity_curve[0]
    max_dd = 0.0
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        dd = (equity / peak) - 1.0
        if dd < max_dd:
            max_dd = dd

    # Calmar
    calmar = _safe_div(ann_ret, abs(max_dd)) if max_dd != 0 else None

    # Sharpe (annualised, rf=0)
    active_returns = [r for r in strategy_returns if r != 0.0]
    if len(active_returns) >= 2:
        mean_r = sum(active_returns) / len(active_returns)
        std_r = _std(active_returns)
        sharpe = _safe_div(mean_r, std_r)
        sharpe = sharpe * math.sqrt(ANNUALISE_FACTOR) if sharpe is not None else None
    else:
        sharpe = None

    # Sortino (downside deviation only)
    downside = [r for r in active_returns if r < 0]
    if len(downside) >= 2 and len(active_returns) >= 2:
        mean_r = sum(active_returns) / len(active_returns)
        down_std = _std(downside)
        sortino = _safe_div(mean_r, down_std)
        sortino = sortino * math.sqrt(ANNUALISE_FACTOR) if sortino is not None else None
    else:
        sortino = None

    # Trade-level metrics
    trade_returns = [t.trade_return for t in trades if t.trade_return is not None]
    wins = [r for r in trade_returns if r > 0]
    losses = [r for r in trade_returns if r <= 0]

    win_rate = _safe_div(len(wins), len(trade_returns)) or 0.0
    avg_win = (sum(wins) / len(wins)) if wins else None
    avg_loss = (sum(losses) / len(losses)) if losses else None
    win_loss_ratio = _safe_div(avg_win, abs(avg_loss)) if avg_win and avg_loss else None
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = _safe_div(gross_profit, gross_loss) if gross_loss > 0 else None

    loss_rate = 1.0 - win_rate
    expectancy = None
    if avg_win is not None and avg_loss is not None:
        expectancy = (win_rate * avg_win) - (loss_rate * abs(avg_loss))

    avg_trade_ret = sum(trade_returns) / len(trade_returns) if trade_returns else 0.0

    # Exposure
    bars_in_market = sum(1 for p in positions if p != 0)
    exposure = _safe_div(bars_in_market, n_bars) or 0.0

    return PerformanceMetrics(
        symbol=symbol,
        signal_type=signal_type,
        lookback=lookback,
        bar_count=n_bars,
        trade_count=n_trades,
        total_return=total_ret,
        annualised_return=ann_ret,
        max_drawdown=max_dd,
        calmar_ratio=calmar,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        win_loss_ratio=win_loss_ratio,
        profit_factor=profit_factor,
        expectancy=expectancy,
        avg_trade_return=avg_trade_ret,
        exposure_pct=exposure,
        sufficient_sample=n_trades >= MIN_TRADES,
    )


# ── Forward return + MFE/MAE extractor ───────────────────────────────────────

def forward_returns(
    bars: Sequence[Bar],
    entry_index: int,
    entry_price: float,
    horizons: tuple[int, ...] = (1, 3, 5),
    window: int = 5,
) -> dict[str, float | None]:
    """
    Compute forward returns and excursion metrics from the bar after entry.

    Returns dict with keys: ret_1d, ret_3d, ret_5d, mfe_5d, mae_5d
    """
    n = len(bars)
    result: dict[str, float | None] = {}

    for h in horizons:
        idx = entry_index + h
        if idx < n:
            result[f"ret_{h}d"] = (bars[idx].close / entry_price) - 1.0
        else:
            result[f"ret_{h}d"] = None

    fwd_window = list(bars[entry_index + 1 : entry_index + window + 1])
    if fwd_window:
        result["mfe_5d"] = max((b.high / entry_price) - 1.0 for b in fwd_window)
        result["mae_5d"] = min((b.low / entry_price) - 1.0 for b in fwd_window)
    else:
        result["mfe_5d"] = None
        result["mae_5d"] = None

    return result
