from argparse import ArgumentParser
from math import fabs

from ib_insync import IB, Stock, Option


def main() -> None:
    parser = ArgumentParser(description="Fetch a sample IBKR option chain for a symbol.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--expiry", default=None, help="Optional expiry in YYYYMMDD format")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7497)
    parser.add_argument("--client-id", type=int, default=21)
    parser.add_argument("--count", type=int, default=5, help="Number of strikes above/below spot to keep")
    args = parser.parse_args()

    ib = IB()

    try:
        ib.connect(args.host, args.port, clientId=args.client_id, timeout=10)

        stock = Stock(args.symbol, "SMART", "USD")
        qualified = ib.qualifyContracts(stock)
        if not qualified:
            raise SystemExit(f"Could not qualify stock contract for {args.symbol}")

        stock = qualified[0]
        ticker = ib.reqTickers(stock)[0]

        spot = ticker.marketPrice()
        if not spot or spot != spot:
            spot = ticker.close

        print(f"symbol={args.symbol}")
        print(f"conId={stock.conId}")
        print(f"spot={spot}")

        chains = ib.reqSecDefOptParams(
            underlyingSymbol=args.symbol,
            futFopExchange="",
            underlyingSecType=stock.secType,
            underlyingConId=stock.conId,
        )

        if not chains:
            raise SystemExit(f"No option chains returned for {args.symbol}")

        chain = next((c for c in chains if c.exchange == "SMART"), chains[0])

        expirations = sorted(chain.expirations)
        strikes = sorted(float(s) for s in chain.strikes if s > 0)

        expiry = args.expiry or expirations[0]
        print(f"selected_expiry={expiry}")

        nearest = sorted(strikes, key=lambda x: fabs(x - spot))[: args.count * 2]
        nearest = sorted(nearest)

        print("sample_contracts=")
        for strike in nearest:
            for right in ("C", "P"):
                contract = Option(args.symbol, expiry, strike, right, "SMART", tradingClass=chain.tradingClass)
                qualified_option = ib.qualifyContracts(contract)
                if qualified_option:
                    c = qualified_option[0]
                    print(
                        f"localSymbol={c.localSymbol} "
                        f"expiry={expiry} strike={strike} right={right} "
                        f"conId={c.conId}"
                    )

    finally:
        if ib.isConnected():
            ib.disconnect()


if __name__ == "__main__":
    main()
