"""
Phase 3 backtest engine — callable module, not a script.

Wraps the simulation loop into pure functions that return dataclasses.
Transaction costs (default 10 bps round-trip) are applied at every exit.

Main entry points:
    run_backtest()      — single backtest run
    sensitivity_sweep() — compare metrics across lookback values
    walk_forward()      — rolling in-sample / out-of-sample validation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from quantlab.providers.base import Bar
from quantlab.research import (
    TradeRecord,
    PerformanceMetrics,
    compute_metrics,
    forward_returns,
)
from quantlab.risk import apply_transaction_cost, DEFAULT_COST_BPS
from quantlab.signals import breakout_signal, sma_signal, atr_stop_price


# ── Single run ─────────────────────────────────────────────────────────────────

@dataclass
class BacktestOutput:
    """Full result of one backtest run."""

    symbol: str
    signal_type: str
    lookback: int
    cost_bps: float
    trades: list[TradeRecord]
    equity_curve: list[float]
    strategy_returns: list[float]
    positions: list[int]
    metrics: PerformanceMetrics


def run_backtest(
    bars: Sequence[Bar],
    symbol: str,
    signal_type: str = "breakout",
    lookback: int = 20,
    initial_capital: float = 10_000.0,
    cost_bps: float = DEFAULT_COST_BPS,
) -> BacktestOutput:
    """
    Simulate a signal-driven long-only strategy over a bar sequence.

    Execution model: today's signal sets tomorrow's position (next-bar fill).
    Transaction costs are deducted from every completed trade's return.

    Args:
        bars:            OHLCV bar sequence, oldest first.
        symbol:          Ticker symbol label.
        signal_type:     "breakout" or "sma".
        lookback:        Signal lookback period in bars.
        initial_capital: Starting portfolio value.
        cost_bps:        Round-trip cost in basis points (default 10 = 0.10%).

    Returns:
        BacktestOutput with trades, equity curve, positions, and metrics.
    """
    bars = list(bars)

    trade_records: list[TradeRecord] = []
    positions = [0]
    equity_curve = [initial_capital]
    strategy_returns = [0.0]

    for i in range(1, len(bars)):
        bar_slice = bars[: i + 1]

        if signal_type == "breakout":
            sig_result = breakout_signal(bar_slice, symbol, lookback)
        elif signal_type == "sma":
            sig_result = sma_signal(bar_slice, symbol, lookback)
        else:
            raise ValueError(f"Unknown signal_type: {signal_type!r}")

        sig = sig_result.signal if sig_result else False
        prev_pos = positions[-1]
        positions.append(1 if sig else 0)

        daily_ret = (bars[i].close / bars[i - 1].close) - 1.0
        strat_ret = prev_pos * daily_ret
        equity_curve.append(equity_curve[-1] * (1.0 + strat_ret))
        strategy_returns.append(strat_ret)

        # Entry
        if positions[-2] == 0 and positions[-1] == 1:
            fwd = forward_returns(bars, i, bars[i].close)
            stop = atr_stop_price(bars[: i + 1], bars[i].close)
            trade_records.append(TradeRecord(
                symbol=symbol,
                signal_date=bars[i].as_of.isoformat(),
                entry_date=bars[i].as_of.isoformat(),
                entry_price=bars[i].close,
                exit_date=None,
                exit_price=None,
                trade_return=None,
                ret_1d=fwd.get("ret_1d"),
                ret_3d=fwd.get("ret_3d"),
                ret_5d=fwd.get("ret_5d"),
                mfe_5d=fwd.get("mfe_5d"),
                mae_5d=fwd.get("mae_5d"),
                atr_stop=stop,
                cost_bps=cost_bps,
            ))

        # Exit — apply round-trip cost to actual hold-period return
        elif positions[-2] == 1 and positions[-1] == 0 and trade_records:
            last = trade_records[-1]
            raw_ret = (bars[i].close / last.entry_price) - 1.0
            trade_records[-1].exit_date = bars[i].as_of.isoformat()
            trade_records[-1].exit_price = bars[i].close
            trade_records[-1].trade_return = apply_transaction_cost(raw_ret, cost_bps)

    metrics = compute_metrics(
        symbol=symbol,
        signal_type=signal_type,
        lookback=lookback,
        bars=bars,
        trades=trade_records,
        equity_curve=equity_curve,
        strategy_returns=strategy_returns,
        positions=positions,
    )

    return BacktestOutput(
        symbol=symbol,
        signal_type=signal_type,
        lookback=lookback,
        cost_bps=cost_bps,
        trades=trade_records,
        equity_curve=equity_curve,
        strategy_returns=strategy_returns,
        positions=positions,
        metrics=metrics,
    )


# ── Parameter sensitivity sweep ────────────────────────────────────────────────

DEFAULT_LOOKBACKS: list[int] = [5, 10, 20, 50]


def sensitivity_sweep(
    bars: Sequence[Bar],
    symbol: str,
    signal_type: str = "breakout",
    lookbacks: list[int] | None = None,
    initial_capital: float = 10_000.0,
    cost_bps: float = DEFAULT_COST_BPS,
) -> dict[int, PerformanceMetrics]:
    """
    Run one backtest per lookback value and return a dict of results.

    Lookback values where len(bars) <= lookback are silently skipped.

    Args:
        bars:        Full bar history (oldest first).
        symbol:      Ticker symbol.
        signal_type: "breakout" or "sma".
        lookbacks:   Lookback periods to test. Default [5, 10, 20, 50].
        initial_capital: Starting capital (reset for each run).
        cost_bps:    Round-trip cost applied uniformly across all runs.

    Returns:
        Dict mapping lookback → PerformanceMetrics.
    """
    lookbacks = lookbacks or DEFAULT_LOOKBACKS
    bars = list(bars)
    results: dict[int, PerformanceMetrics] = {}

    for lb in lookbacks:
        if len(bars) <= lb:
            continue
        out = run_backtest(bars, symbol, signal_type, lb, initial_capital, cost_bps)
        results[lb] = out.metrics

    return results


def print_sensitivity_table(results: dict[int, PerformanceMetrics]) -> None:
    """Print a compact side-by-side comparison of sweep results."""
    from quantlab.risk import fmt_pct, fmt_float

    hdr = (
        f"{'lookback':>10}  {'trades':>7}  {'total_ret':>10}  "
        f"{'sharpe':>8}  {'calmar':>8}  {'win_rate':>9}  {'sample':>7}"
    )
    sep = "=" * len(hdr)
    print(f"\n{sep}")
    print("  Parameter Sensitivity Sweep")
    print(sep)
    print(hdr)
    print("-" * len(hdr))

    for lb in sorted(results):
        m = results[lb]
        flag = "  *" if not m.sufficient_sample else ""
        print(
            f"{lb:>10}  {m.trade_count:>7}  {fmt_pct(m.total_return):>10}  "
            f"{fmt_float(m.sharpe_ratio):>8}  {fmt_float(m.calmar_ratio):>8}  "
            f"{fmt_pct(m.win_rate):>9}  {'OK' if m.sufficient_sample else 'LOW':>7}{flag}"
        )

    print(sep)
    if any(not m.sufficient_sample for m in results.values()):
        print("  * fewer than 30 trades — treat as directional only")


# ── Walk-forward validation ────────────────────────────────────────────────────

@dataclass
class WalkForwardWindow:
    """One IS/OOS pair in a rolling walk-forward test."""

    window_index: int
    is_start_bar: int       # inclusive index into the full bar list
    is_end_bar: int         # exclusive
    oos_start_bar: int
    oos_end_bar: int        # exclusive
    in_sample: PerformanceMetrics
    out_of_sample: PerformanceMetrics | None   # None when OOS slice too short


def walk_forward(
    bars: Sequence[Bar],
    symbol: str,
    signal_type: str = "breakout",
    lookback: int = 20,
    is_bars: int = 252,
    oos_bars: int = 63,
    initial_capital: float = 10_000.0,
    cost_bps: float = DEFAULT_COST_BPS,
) -> list[WalkForwardWindow]:
    """
    Rolling walk-forward validation.

    Slides a fixed in-sample window forward by `oos_bars` steps, runs a
    backtest on both the IS and OOS slices, and returns paired metrics.

    Args:
        bars:            Full bar history (oldest first).
        symbol:          Ticker symbol.
        signal_type:     "breakout" or "sma".
        lookback:        Signal lookback period.
        is_bars:         In-sample window length in bars (default 252 ≈ 1 yr).
        oos_bars:        Out-of-sample window length in bars (default 63 ≈ 1 qtr).
        initial_capital: Starting capital reset for each window.
        cost_bps:        Round-trip cost applied in every window.

    Returns:
        List of WalkForwardWindow, one per complete IS window found.
    """
    bars = list(bars)
    n = len(bars)
    windows: list[WalkForwardWindow] = []
    idx = 0

    while True:
        is_end = idx + is_bars
        if is_end > n:
            break

        is_slice = bars[idx:is_end]
        is_out = run_backtest(is_slice, symbol, signal_type, lookback, initial_capital, cost_bps)

        oos_start = is_end
        oos_end = min(oos_start + oos_bars, n)
        oos_metrics: PerformanceMetrics | None = None

        if oos_start < n:
            oos_slice = bars[oos_start:oos_end]
            if len(oos_slice) > lookback:
                oos_out = run_backtest(
                    oos_slice, symbol, signal_type, lookback, initial_capital, cost_bps
                )
                oos_metrics = oos_out.metrics

        windows.append(WalkForwardWindow(
            window_index=len(windows),
            is_start_bar=idx,
            is_end_bar=is_end,
            oos_start_bar=oos_start,
            oos_end_bar=oos_end,
            in_sample=is_out.metrics,
            out_of_sample=oos_metrics,
        ))

        idx += oos_bars

    return windows


def print_walk_forward_summary(windows: list[WalkForwardWindow]) -> None:
    """Print IS vs OOS Sharpe and return side-by-side for all windows."""
    from quantlab.risk import fmt_float, fmt_pct

    sep = "=" * 74
    print(f"\n{sep}")
    print("  Walk-Forward Validation")
    print(sep)
    print(
        f"{'win':>4}  {'IS bars':>8}  {'IS sharpe':>10}  {'IS ret':>8}  "
        f"{'OOS sharpe':>11}  {'OOS ret':>9}"
    )
    print("-" * 74)

    for w in windows:
        is_len = w.is_end_bar - w.is_start_bar
        oos_sh = fmt_float(w.out_of_sample.sharpe_ratio) if w.out_of_sample else "  --"
        oos_ret = fmt_pct(w.out_of_sample.total_return) if w.out_of_sample else "    --"
        print(
            f"{w.window_index:>4}  {is_len:>8}  "
            f"{fmt_float(w.in_sample.sharpe_ratio):>10}  {fmt_pct(w.in_sample.total_return):>8}  "
            f"{oos_sh:>11}  {oos_ret:>9}"
        )

    print(sep)
