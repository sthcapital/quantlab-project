from argparse import ArgumentParser
from datetime import datetime, timedelta

from ib_insync import IB, Stock


def main() -> None:
    parser = ArgumentParser(description="Fetch IBKR historical news headlines for a symbol.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7497)
    parser.add_argument("--client-id", type=int, default=11)
    args = parser.parse_args()

    ib = IB()

    try:
        ib.connect(args.host, args.port, clientId=args.client_id, timeout=10)

        providers = ib.reqNewsProviders()
        print("providers=")
        for provider in providers:
            print(provider)

        contract = Stock(args.symbol, "SMART", "USD")
        qualified = ib.qualifyContracts(contract)
        if not qualified:
            raise SystemExit(f"Could not qualify contract for {args.symbol}")

        con_id = qualified[0].conId
        end_dt = datetime.utcnow()
        start_dt = end_dt - timedelta(days=args.days)

        headlines = ib.reqHistoricalNews(
            conId=con_id,
            providerCodes="BRFG+BRFUPDN+DJNL",
            startDateTime=start_dt.strftime("%Y%m%d %H:%M:%S"),
            endDateTime=end_dt.strftime("%Y%m%d %H:%M:%S"),
            totalResults=20,
            historicalNewsOptions=[],
        )

        print(f"symbol={args.symbol}")
        print(f"headline_count={len(headlines)}")
        for item in headlines:
            print(
                f"{item.time} provider={item.providerCode} "
                f"article_id={item.articleId} headline={item.headline}"
            )

    finally:
        if ib.isConnected():
            ib.disconnect()


if __name__ == "__main__":
    main()
