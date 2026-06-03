from argparse import ArgumentParser
import html
import re
from collections import Counter
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


def fetch_bars(ib: IB, symbol: str, start: str, end: str):
    contract = Stock(symbol, "SMART", "USD")
    qualified = ib.qualifyContracts(contract)
    if not qualified:
        raise SystemExit(f"Could not qualify contract for {symbol}")

    contract = qualified[0]
    bars = ib.reqHistoricalData(
        contract,
        endDateTime=end.replace("-", "") + " 23:59:59",
        durationStr="90 D",
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=True,
        formatDate=1,
    )

    results = []
    for bar in bars:
        bar_date = str(bar.date)[:10]
        if start <= bar_date <= end:
            results.append(
                {
                    "date": bar_date,
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume),
                }
            )
    return contract, results


def fetch_news(ib: IB, contract, days: int, limit: int):
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=days)

    headlines = ib.reqHistoricalNews(
        conId=contract.conId,
        providerCodes="BRFG+BRFUPDN+DJNL",
        startDateTime=start_dt.strftime("%Y%m%d %H:%M:%S"),
        endDateTime=end_dt.strftime("%Y%m%d %H:%M:%S"),
        totalResults=limit,
        historicalNewsOptions=[],
    )

    rows = []
    for item in headlines:
        cleaned = clean_headline(item.headline)
        rows.append(
            {
                "time": item.time,
                "date": str(item.time)[:10],
                "provider": item.providerCode,
                "article_id": item.articleId,
                "category": classify_headline(cleaned),
                "headline": cleaned,
            }
        )
    return rows


def rolling_news_counts(news_rows, trade_date: str, lookback_days: int):
    end_dt = datetime.strptime(trade_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=lookback_days)

    selected = []
    for row in news_rows:
        news_dt = datetime.strptime(row["date"], "%Y-%m-%d")
        if start_dt <= news_dt <= end_dt:
            selected.append(row)

    counts = Counter(row["category"] for row in selected)
    return {
        "recent_news_count": len(selected),
        "recent_upgrade_count": counts.get("upgrade", 0),
        "recent_analyst_action_count": counts.get("analyst_action", 0),
        "recent_earnings_count": counts.get("earnings", 0),
        "recent_management_count": counts.get("management", 0),
        "recent_other_count": counts.get("other", 0),
    }


def sma(values, window):
    if len(values) < window:
        return None
    return sum(values[-window:]) / window


def run_strategy(bars, mode: str, lookback: int, news_rows, news_lookback: int):
    closes = []
    trades = []

    for i, bar in enumerate(bars):
        closes.append(bar["close"])

        if i < lookback:
            continue

        signal = False

        if mode == "sma":
            avg = sma(closes[:-1], lookback)
            if avg is not None and bar["close"] > avg:
                signal = True

        elif mode == "breakout":
            prior_high = max(b["high"] for b in bars[i - lookback:i])
            if bar["close"] > prior_high:
                signal = True

        if signal:
            news_features = rolling_news_counts(news_rows, bar["date"], news_lookback)
            trades.append(
                {
                    "date": bar["date"],
                    "close": bar["close"],
                    **news_features,
                }
            )

    return trades


def main() -> None:
    parser = ArgumentParser(description="Run a simple backtest with IBKR news features.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--mode", choices=["sma", "breakout"], default="breakout")
    parser.add_argument("--lookback", type=int, default=5)
    parser.add_argument("--news-lookback", type=int, default=7)
    parser.add_argument("--news-days", type=int, default=120)
    parser.add_argument("--news-limit", type=int, default=100)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7497)
    parser.add_argument("--client-id", type=int, default=31)
    args = parser.parse_args()

    ib = IB()

    try:
        ib.connect(args.host, args.port, clientId=args.client_id, timeout=10)

        contract, bars = fetch_bars(ib, args.symbol, args.start, args.end)
        news_rows = fetch_news(ib, contract, args.news_days, args.news_limit)
        trades = run_strategy(
            bars=bars,
            mode=args.mode,
            lookback=args.lookback,
            news_rows=news_rows,
            news_lookback=args.news_lookback,
        )

        print(f"symbol={args.symbol}")
        print(f"bar_count={len(bars)}")
        print(f"news_count={len(news_rows)}")
        print(f"trade_count={len(trades)}")
        print("trades=")

        for trade in trades:
            print(
                f"{trade['date']} close={trade['close']:.2f} "
                f"recent_news_count={trade['recent_news_count']} "
                f"recent_upgrade_count={trade['recent_upgrade_count']} "
                f"recent_analyst_action_count={trade['recent_analyst_action_count']} "
                f"recent_earnings_count={trade['recent_earnings_count']} "
                f"recent_management_count={trade['recent_management_count']} "
                f"recent_other_count={trade['recent_other_count']}"
            )

    finally:
        if ib.isConnected():
            ib.disconnect()


if __name__ == "__main__":
    main()
