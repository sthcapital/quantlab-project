from argparse import ArgumentParser
from datetime import datetime

from quantlab.providers import create_market_data_provider


def parse_date(value: str):
    return datetime.strptime(value, "%Y-%m-%d").date()


def max_drawdown(equity_curve: list[float]) -> float:
    peak = equity_curve[0]
    max_dd = 0.0

    for equity in equity_curve:
        if equity > peak:
            peak = equity
        drawdown = (equity / peak) - 1.0
        if drawdown < max_dd:
            max_dd = drawdown

    return max_dd


def main() -> None:
    parser = ArgumentParser(description="Run a simple daily-bar backtest.")
    parser.add_argument("--provider", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", required=True, type=parse_date)
    parser.add_argument("--end", required=True, type=parse_date)
    parser.add_argument("--lookback", type=int, default=5)
    parser.add_argument("--initial-capital", type=float, default=10000.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7497)
    parser.add_argument("--client-id", type=int, default=1)
    parser.add_argument("--base-url", default="https://www.alphavantage.co")
    parser.add_argument("--api-key", default=None)

    args = parser.parse_args()

    provider_kwargs = {}
    if args.provider.lower() == "ibkr":
        provider_kwargs = {
            "host": args.host,
            "port": args.port,
            "client_id": args.client_id,
        }
    else:
        provider_kwargs = {
            "base_url": args.base_url,
            "api_key": args.api_key,
        }

    provider = create_market_data_provider(args.provider, **provider_kwargs)
    bars = provider.get_daily_bars(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
    )

    if len(bars) <= args.lookback:
        raise SystemExit(
            f"Need more than {args.lookback} bars, received {len(bars)}."
        )

    dates = [bar.as_of for bar in bars]
    closes = [bar.close for bar in bars]

    sma_values: list[float | None] = []
    for i in range(len(closes)):
        if i + 1 < args.lookback:
            sma_values.append(None)
        else:
            sma = sum(closes[i - args.lookback + 1:i + 1]) / args.lookback
            sma_values.append(sma)

    raw_signals: list[int] = []
    for close, sma in zip(closes, sma_values):
        if sma is None:
            raw_signals.append(0)
        else:
            raw_signals.append(1 if close > sma else 0)

    positions = [0]
    for i in range(1, len(raw_signals)):
        positions.append(raw_signals[i - 1])

    daily_returns = [0.0]
    strategy_returns = [0.0]
    equity_curve = [args.initial_capital]

    for i in range(1, len(closes)):
        daily_return = (closes[i] / closes[i - 1]) - 1.0
        strategy_return = positions[i] * daily_return
        next_equity = equity_curve[-1] * (1.0 + strategy_return)

        daily_returns.append(daily_return)
        strategy_returns.append(strategy_return)
        equity_curve.append(next_equity)

    total_return = (equity_curve[-1] / args.initial_capital) - 1.0
    max_dd = max_drawdown(equity_curve)

    print(f"provider={args.provider}")
    print(f"symbol={args.symbol}")
    print(f"bars={len(bars)}")
    print(f"lookback={args.lookback}")
    print(f"start={dates[0]}")
    print(f"end={dates[-1]}")
    print(f"final_position={positions[-1]}")
    print(f"initial_capital={args.initial_capital:.2f}")
    print(f"ending_equity={equity_curve[-1]:.2f}")
    print(f"total_return={total_return:.4%}")
    print(f"max_drawdown={max_dd:.4%}")

    print("last_rows=")
    for i in range(max(0, len(dates) - 5), len(dates)):
        sma_text = "None" if sma_values[i] is None else f"{sma_values[i]:.2f}"
        print(
            f"{dates[i]} close={closes[i]:.2f} sma={sma_text} "
            f"signal={raw_signals[i]} position={positions[i]} "
            f"equity={equity_curve[i]:.2f}"
        )


if __name__ == "__main__":
    main()
