from argparse import ArgumentParser
from datetime import datetime
from math import fabs

from ib_insync import IB, Stock, Option


def get_spot_price(ib: IB, stock: Stock) -> float:
    ib.reqMarketDataType(3)
    ticker = ib.reqTickers(stock)[0]

    candidates = [ticker.marketPrice(), ticker.last, ticker.close]
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
        raise SystemExit("Could not determine spot price.")

    return float(bars[-1].close)


def safe_mid(bid, ask):
    if bid is not None and ask is not None and bid == bid and ask == ask and bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    return None


def fmt(value):
    if value is None or value != value:
        return "nan"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    parser = ArgumentParser(description="Fetch delayed IBKR option quotes and Greeks.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--expiry", default=None, help="Optional expiry in YYYYMMDD format")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7497)
    parser.add_argument("--client-id", type=int, default=22)
    parser.add_argument("--count", type=int, default=3, help="Number of nearest strikes to sample")
    args = parser.parse_args()

    ib = IB()

    try:
        ib.connect(args.host, args.port, clientId=args.client_id, timeout=10)
        ib.reqMarketDataType(3)

        stock = Stock(args.symbol, "SMART", "USD")
        stock = ib.qualifyContracts(stock)[0]
        spot = get_spot_price(ib, stock)

        chains = ib.reqSecDefOptParams(
            underlyingSymbol=args.symbol,
            futFopExchange="",
            underlyingSecType=stock.secType,
            underlyingConId=stock.conId,
        )
        if not chains:
            raise SystemExit(f"No option chains returned for {args.symbol}")

        chain = next((c for c in chains if c.exchange == "SMART"), chains[0])

        today = datetime.utcnow().strftime("%Y%m%d")
        expirations = sorted(exp for exp in chain.expirations if exp >= today)
        if not expirations:
            raise SystemExit("No valid future expiries returned.")

        expiry = args.expiry or expirations[min(2, len(expirations) - 1)]
        strikes = sorted(float(s) for s in chain.strikes if 0.5 * spot <= float(s) <= 1.5 * spot)
        nearest = sorted(strikes, key=lambda x: fabs(x - spot))[: args.count]
        nearest = sorted(nearest)

        contracts = []
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
                qualified = ib.qualifyContracts(contract)
                if qualified:
                    contracts.append(qualified[0])

        if not contracts:
            raise SystemExit("No option contracts qualified.")

        tickers = ib.reqTickers(*contracts)

        print(f"symbol={args.symbol}")
        print(f"spot={spot:.2f}")
        print(f"selected_expiry={expiry}")
        print("quotes=")

        for contract, ticker in zip(contracts, tickers):
            bid = ticker.bid
            ask = ticker.ask
            last = ticker.last
            mid = safe_mid(bid, ask)

            greeks = ticker.modelGreeks
            iv = greeks.impliedVol if greeks else None
            delta = greeks.delta if greeks else None
            gamma = greeks.gamma if greeks else None
            theta = greeks.theta if greeks else None
            vega = greeks.vega if greeks else None

            print(
                f"localSymbol={contract.localSymbol} "
                f"strike={contract.strike} right={contract.right} "
                f"bid={fmt(bid)} ask={fmt(ask)} last={fmt(last)} mid={fmt(mid)} "
                f"iv={fmt(iv)} delta={fmt(delta)} gamma={fmt(gamma)} "
                f"theta={fmt(theta)} vega={fmt(vega)}"
            )

    finally:
        if ib.isConnected():
            ib.disconnect()


if __name__ == "__main__":
    main()
