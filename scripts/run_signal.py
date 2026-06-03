from argparse import ArgumentParser
from datetime import datetime

from quantlab.providers import create_market_data_provider


def parse_date(value: str):
    return datetime.strptime(value, "%Y-%m-%d").date()


def main() -> None:
    parser = ArgumentParser(description="Run a simple signal on daily bars.")
    parser.add_argument("--provider", required=True, help="Provider name, e.g. ibkr")
    parser.add_argument("--symbol", required=True, help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--start", required=True, type=parse_date, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end", required=True, type=parse_date, help="End date in YYYY-MM-DD format")
    parser.add_argument("--lookback", type=int, default=5, help="Moving average lookback")
    parser.add_argument("--host", default="127.0.0.1", help="IBKR host")
    parser.add_argument("--port", type=int, default=7497, help="IBKR port")
    parser.add_argument("--client-id", type=int, default=1, help="IBKR client id")
    parser.add_argument("--base-url", default="https://www.alphavantage.co", help="HTTP provider base URL")
    parser.add_argument("--api-key", default=None, help="HTTP provider API key")

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

    if len(bars) < args.lookback:
        raise SystemExit(
            f"Not enough bars for lookback={args.lookback}. Received {len(bars)} bars."
        )

    closes = [bar.close for bar in bars]
    latest_close = closes[-1]
    moving_average = sum(closes[-args.lookback:]) / args.lookback
    signal = "long" if latest_close > moving_average else "flat"

    print(f"provider={args.provider}")
    print(f"symbol={args.symbol}")
    print(f"bars={len(bars)}")
    print(f"lookback={args.lookback}")
    print(f"latest_date={bars[-1].as_of}")
    print(f"latest_close={latest_close:.2f}")
    print(f"moving_average={moving_average:.2f}")
    print(f"signal={signal}")


if __name__ == "__main__":
    main()
