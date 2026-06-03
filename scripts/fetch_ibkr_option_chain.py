from argparse import ArgumentParser
from datetime import datetime
from math import fabs

from ib_insync import IB, Stock, Option


def get_spot_price(ib: IB, stock: Stock) -> float:
    ib.reqMarketDataType(3)

    ticker = ib.reqTickers(stock)[0]
    candidates = [
        ticker.marketPrice(),
        ticker.last,
        ticker.close,
    ]

    for value in candidates:
        if value is not None and value == value and value > 0:
            return float(value)

    bars = ib.reqHistoricalData(
        stock,
        endDateTime="",
        durationStr="5 D",
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=True,
        formatDate=1,
    )

    if not bars:
        raise SystemExit("Could not determine spot price from market data or recent history.")

    return float(bars[-1].close)


def main() -> None:
    parser = ArgumentParser(description="Fetch a sample IBKR option chain for a symbol.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--expiry", default=None, help="Optional expiry in YYYYMMDD format")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7497)
    parser.add_argument("--client-id", type=int, default=21)
    parser.add_argument("--count", type=int, default=4, help="Number of strikes above/below spot to keep")
    args = parser.parse_args()

    ib = IB()

    try:
        ib.connect(args.host, args.port, clientId=args.client_id, timeout=10)

        stock = Stock(args.symbol, "SMART", "USD")
        qualified = ib.qualifyContracts(stock)
        if not qualified:
            raise SystemExit(f"Could not qualify stock contract for {args.symbol}")

        stock = qualified[0]
        spot = get_spot_price(ib, stock)

        print(f"symbol={args.symbol}")
        print(f"conId={stock.conId}")
        print(f"spot={spot:.2f}")

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

        today = datetime.utcnow().strftime("%Y%m%d")
        valid_expiries = [exp for exp in expirations if exp >= today]
        if not valid_expiries:
            raise SystemExit("No valid future expiries returned.")

        expiry = args.expiry or valid_expiries[min(2, len(valid_expiries) - 1)]

        filtered_strikes = [s for s in strikes if 0.5 * spot <= s <= 1.5 * spot]
        nearest = sorted(filtered_strikes, key=lambda x: fabs(x - spot))[: args.count * 2]
        nearest = sorted(nearest)

        print(f"selected_expiry={expiry}")
        print("sample_contracts=")

        found = 0
        for strike in nearest:
            for right in ("C", "P"):
                contract = Option(
                    args.symbol,
                    expiry,
                    strike,
                    right,
                    "SMART",
                    tradingClass=chain.tradingClass,
                )
                qualified_option = ib.qualifyContracts(contract)
                if qualified_option:
                    c = qualified_option[0]
                    found += 1
                    print(
                        f"localSymbol={c.localSymbol} "
                        f"expiry={expiry} strike={strike} right={right} "
                        f"conId={c.conId}"
                    )

        if found == 0:
            print("No option contracts qualified for the selected expiry/strike sample.")

    finally:
        if ib.isConnected():
            ib.disconnect()


if __name__ == "__main__":
    main()
