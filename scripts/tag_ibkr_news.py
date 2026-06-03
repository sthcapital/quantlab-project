from argparse import ArgumentParser
import html
import re
from datetime import datetime, timedelta

from ib_insync import IB, Stock

NEWS_RE = re.compile(r"\{.*?\}!?")


def clean_headline(text: str) -> str:
    return html.unescape(NEWS_RE.sub("", text).strip())


def classify_headline(headline: str) -> str:
    low = headline.lower()

    if any(k in low for k in ["upgrade", "upgraded", "raises target", "raised target"]):
        return "upgrade"

    if any(
        k in low
        for k in [
            "downgrade",
            "downgraded",
            "reiterated",
            "underweight",
            "overweight",
            "neutral",
            "buy",
            "sell",
            "outperform",
            "market perform",
            "resumed",
        ]
    ):
        return "analyst_action"

    if any(k in low for k in ["earnings", "q1", "q2", "q3", "q4", "guide", "guidance", "revenue", "sales"]):
        return "earnings"

    if any(k in low for k in ["ceo", "cfo", "chief", "executive", "board"]):
        return "management"

    return "other"


def main() -> None:
    parser = ArgumentParser(description="Fetch and tag IBKR historical news headlines.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--limit", type=int, default=20)
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
            totalResults=args.limit,
            historicalNewsOptions=[],
        )

        rows = []
        print(f"symbol={args.symbol}")
        print(f"headline_count={len(headlines)}")

        for item in headlines:
            cleaned = clean_headline(item.headline)
            category = classify_headline(cleaned)
            rows.append((item.time, item.providerCode, item.articleId, category, cleaned))

        rows.sort(key=lambda row: row[0], reverse=True)

        for row in rows:
            print(
                f"{row[0]} provider={row[1]} article_id={row[2]} "
                f"category={row[3]} headline={row[4]}"
            )

        print("summary=")
        counts = {}
        for _, _, _, category, _ in rows:
            counts[category] = counts.get(category, 0) + 1

        for category in sorted(counts):
            print(f"{category}={counts[category]}")

    finally:
        if ib.isConnected():
            ib.disconnect()


if __name__ == "__main__":
    main()
