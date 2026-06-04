"""
Layer 6: Risk management and performance reporting.

Transaction cost model:
    Applies a round-trip cost in basis points (bps) to each trade return.
    Default 10 bps (0.10%) covers typical retail commission + spread for liquid stocks.
    1 bps = 0.01% = 0.0001

Summary printing and grouped analysis (no_news vs with_news, by category).
"""

from __future__ import annotations

from typing import Sequence

from quantlab.research import PerformanceMetrics


# ── Transaction cost model ────────────────────────────────────────────────────

DEFAULT_COST_BPS = 10.0  # 10 basis points round-trip (~$0.005/share + spread)


def apply_transaction_cost(raw_return: float, cost_bps: float = DEFAULT_COST_BPS) -> float:
    """
    Deduct round-trip transaction cost from a raw trade return.

    Args:
        raw_return: Gross return, e.g. 0.05 = 5%.
        cost_bps:   Round-trip cost in basis points. Default 10 bps.

    Returns:
        Net return after cost deduction.
    """
    return raw_return - (cost_bps / 10_000)


def apply_costs_to_trades(trades: list, cost_bps: float = DEFAULT_COST_BPS) -> list:
    """Apply transaction costs to a list of TradeRecord objects in-place."""
    for trade in trades:
        if trade.trade_return is not None:
            trade.trade_return = apply_transaction_cost(trade.trade_return, cost_bps)
            trade.cost_bps = cost_bps
    return trades


# ── Formatting helpers ─────────────────────────────────────────────────────────

def fmt_pct(value: float | None, decimals: int = 2) -> str:
    if value is None:
        return "NA"
    return f"{value * 100:.{decimals}f}%"


def fmt_float(value: float | None, decimals: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{value:.{decimals}f}"


# ── Grouped summary analysis ───────────────────────────────────────────────────

def _summarize_returns(values: list[float | None]) -> dict:
    clean = [v for v in values if v is not None]
    if not clean:
        return {"n": 0, "avg": None, "med": None, "hit": None, "std": None}

    n = len(clean)
    clean_sorted = sorted(clean)
    avg = sum(clean) / n
    med = clean_sorted[n // 2] if n % 2 == 1 else (clean_sorted[n // 2 - 1] + clean_sorted[n // 2]) / 2

    import math
    variance = sum((v - avg) ** 2 for v in clean) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(variance)

    return {
        "n": n,
        "avg": avg,
        "med": med,
        "hit": sum(1 for v in clean if v > 0) / n,
        "std": std,
    }


def print_trade_summary(label: str, trades: list) -> None:
    """Print a grouped summary table for a list of trade records."""
    if not trades:
        print(f"\n== {label} ==  (no trades)")
        return

    r1 = _summarize_returns([t.ret_1d for t in trades])
    r3 = _summarize_returns([t.ret_3d for t in trades])
    r5 = _summarize_returns([t.ret_5d for t in trades])
    mfe = _summarize_returns([t.mfe_5d for t in trades])
    mae = _summarize_returns([t.mae_5d for t in trades])

    print(f"\n== {label} ==")
    print(f"  signals  = {len(trades)}", "  *** BELOW MIN SAMPLE ***" if len(trades) < 30 else "")
    print(f"  1D  avg={fmt_pct(r1['avg'])}  med={fmt_pct(r1['med'])}  hit={fmt_pct(r1['hit'])}  std={fmt_pct(r1['std'])}")
    print(f"  3D  avg={fmt_pct(r3['avg'])}  med={fmt_pct(r3['med'])}  hit={fmt_pct(r3['hit'])}  std={fmt_pct(r3['std'])}")
    print(f"  5D  avg={fmt_pct(r5['avg'])}  med={fmt_pct(r5['med'])}  hit={fmt_pct(r5['hit'])}  std={fmt_pct(r5['std'])}")
    print(f"  MFE avg={fmt_pct(mfe['avg'])}  med={fmt_pct(mfe['med'])}")
    print(f"  MAE avg={fmt_pct(mae['avg'])}  med={fmt_pct(mae['med'])}")


def print_metrics(m: PerformanceMetrics) -> None:
    """Print the full PerformanceMetrics block."""
    flag = "  *** BELOW MIN SAMPLE — treat as directional only ***" if not m.sufficient_sample else ""

    print(f"\n{'='*60}")
    print(f"  {m.symbol} | {m.signal_type} | lookback={m.lookback}")
    print(f"{'='*60}")
    print(f"  bars            = {m.bar_count}")
    print(f"  trades          = {m.trade_count}{flag}")
    print(f"  total return    = {fmt_pct(m.total_return)}")
    print(f"  ann. return     = {fmt_pct(m.annualised_return)}")
    print(f"  max drawdown    = {fmt_pct(m.max_drawdown)}")
    print(f"  calmar          = {fmt_float(m.calmar_ratio)}")
    print(f"  sharpe          = {fmt_float(m.sharpe_ratio)}")
    print(f"  sortino         = {fmt_float(m.sortino_ratio)}")
    print(f"  win rate        = {fmt_pct(m.win_rate)}")
    print(f"  avg win         = {fmt_pct(m.avg_win)}")
    print(f"  avg loss        = {fmt_pct(m.avg_loss)}")
    print(f"  win/loss ratio  = {fmt_float(m.win_loss_ratio)}")
    print(f"  profit factor   = {fmt_float(m.profit_factor)}")
    print(f"  expectancy/trade= {fmt_pct(m.expectancy)}")
    print(f"  avg trade ret   = {fmt_pct(m.avg_trade_return)}")
    print(f"  exposure        = {fmt_pct(m.exposure_pct)}")
    print(f"{'='*60}")


def print_grouped_summaries(trades: list) -> None:
    """Print no_news / with_news / by_category breakdowns."""
    print_trade_summary("all signals", trades)
    print_trade_summary("no news", [t for t in trades if t.news_count == 0])
    print_trade_summary("with news", [t for t in trades if t.news_count > 0])

    categories = sorted(set(t.news_category for t in trades))
    for cat in categories:
        subset = [t for t in trades if t.news_category == cat]
        if len(subset) >= 2:
            print_trade_summary(f"category={cat}", subset)
