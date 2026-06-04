from argparse import ArgumentParser
from datetime import datetime

from quantlab.providers import create_market_data_provider


def parse_date(value: str):
    return datetime.strptime(value, "%Y-%m-%d").date()


def main() -> None:
    parser = ArgumentParser(description="Fetch daily bars from a configured market data provider.")
    parser.add_argument("--provider", required=True, help="Provider name, e.g. alpha_vantage or ibkr")
    parser.add_argument("--symbol", required=True, help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--start", required=True, type=parse_date, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end", required=True, type=parse_date, help="End date in YYYY-MM-DD format")
    parser.add_argument("--host", default="127.0.0.1", help="IBKR host")
    parser.add_argument("--port", type=int, default=7497, help="IBKR port")
    parser.add_argument("--client-id", type=int, default=1, help="IBKR client id")
    parser.add_argument("--base-url", default="https://www.alphavantage.co", help="HTTP provider base URL")
    parser.add_argument("--api-key", default=None, help="HTTP provider API key")

    args = parser.parse_args()

    provider_kwargs = {}
    if args.provider.lower() == "ibkr":
        from quantlab.providers.ibkr import ping_tws
        if not ping_tws(args.host, args.port):
            raise SystemExit(
                f"\nTWS / IB Gateway is not reachable at {args.host}:{args.port}.\n"
                "Start TWS or IB Gateway, enable API access, and try again."
            )
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

    print(f"provider={args.provider}")
    print(f"symbol={args.symbol}")
    print(f"bar_count={len(bars)}")

    for bar in bars[-5:]:
        print(
            f"{bar.as_of} "
            f"open={bar.open} high={bar.high} low={bar.low} "
            f"close={bar.close} volume={bar.volume}"
        )


if __name__ == "__main__":
    main()
