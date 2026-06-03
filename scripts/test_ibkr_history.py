from ib_insync import IB, Stock


def main() -> None:
    ib = IB()

    try:
        ib.connect("127.0.0.1", 7497, clientId=2, timeout=10)

        contract = Stock("AAPL", "SMART", "USD")
        ib.qualifyContracts(contract)

        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr="10 D",
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )

        print(f"bar_count={len(bars)}")
        for bar in bars[-5:]:
            print(
                f"{bar.date} "
                f"open={bar.open} high={bar.high} low={bar.low} "
                f"close={bar.close} volume={bar.volume}"
            )
    finally:
        if ib.isConnected():
            ib.disconnect()


if __name__ == "__main__":
    main()
