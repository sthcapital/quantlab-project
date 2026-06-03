from argparse import ArgumentParser
import csv
import html
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

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

    if any(
        k in low
        for k in ["earnings", "q1", "q2", "q3", "q4", "guide", "guidance", "revenue", "sales"]
    ):
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
        endDateTime="",
        durationStr="180 D",
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


def pct_return(exit_price, entry_price):
    if exit_price is None or entry_price == 0:
        return None
    return (exit_price / entry_price) - 1.0


def dominant_news_category(news_features):
    if news_features["recent_news_count"] == 0:
        return "none"

    category_counts = {
        "upgrade": news_features["recent_upgrade_count"],
        "analyst_action": news_features["recent_analyst_action_count"],
        "earnings": news_features["recent_earnings_count"],
        "management": news_features["recent_management_count"],
        "other": news_features["recent_other_count"],
    }
    return max(category_counts, key=category_counts.get)


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

        if not signal:
            continue

        entry_close = bar["close"]
        future_1 = bars[i + 1]["close"] if i + 1 < len(bars) else None
        future_3 = bars[i + 3]["close"] if i + 3 < len(bars) else None
        future_5 = bars[i + 5]["close"] if i + 5 < len(bars) else None

        forward_window = bars[i + 1 : i + 6]
        mfe_5d = None
        mae_5d = None
        if forward_window:
            mfe_5d = max((x["high"] / entry_close) - 1.0 for x in forward_window)
            mae_5d = min((x["low"] / entry_close) - 1.0 for x in forward_window)

        news_features = rolling_news_counts(news_rows, bar["date"], news_lookback)

        trades.append(
            {
                "date": bar["date"],
                "close": entry_close,
                "ret_1d": pct_return(future_1, entry_close),
                "ret_3d": pct_return(future_3, entry_close),
                "ret_5d": pct_return(future_5, entry_close),
                "mfe_5d": mfe_5d,
                "mae_5d": mae_5d,
                "dominant_news_category": dominant_news_category(news_features),
                **news_features,
            }
        )

    return trades


def fmt_pct(value):
    if value is None:
        return "NA"
    return f"{value * 100:.2f}%"


def summarize(values):
    clean = [v for v in values if v is not None]
    if not clean:
        return {"n": 0, "avg": None, "med": None, "hit": None}

    clean_sorted = sorted(clean)
    n = len(clean)
    med = clean_sorted[n // 2] if n % 2 == 1 else (clean_sorted[n // 2 - 1] + clean_sorted[n // 2]) / 2

    return {
        "n": n,
        "avg": sum(clean) / n,
        "med": med,
        "hit": sum(1 for v in clean if v > 0) / n,
    }


def print_summary(label, subset):
    r1 = summarize([x["ret_1d"] for x in subset])
    r3 = summarize([x["ret_3d"] for x in subset])
    r5 = summarize([x["ret_5d"] for x in subset])
    mfe = summarize([x["mfe_5d"] for x in subset])
    mae = summarize([x["mae_5d"] for x in subset])

    print(f"\n== {label} ==")
    print(f"signals={len(subset)}")
    print(f"1D  avg={fmt_pct(r1['avg'])} med={fmt_pct(r1['med'])} hit={fmt_pct(r1['hit'])}")
    print(f"3D  avg={fmt_pct(r3['avg'])} med={fmt_pct(r3['med'])} hit={fmt_pct(r3['hit'])}")
    print(f"5D  avg={fmt_pct(r5['avg'])} med={fmt_pct(r5['med'])} hit={fmt_pct(r5['hit'])}")
    print(f"MFE avg={fmt_pct(mfe['avg'])} med={fmt_pct(mfe['med'])}")
    print(f"MAE avg={fmt_pct(mae['avg'])} med={fmt_pct(mae['med'])}")


def export_trades_csv(symbol: str, mode: str, trades):
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = output_dir / f"{symbol}_{mode}_news_trades.csv"

    fieldnames = [
        "date",
        "close",
        "ret_1d",
        "ret_3d",
        "ret_5d",
        "mfe_5d",
        "mae_5d",
        "dominant_news_category",
        "recent_news_count",
        "recent_upgrade_count",
        "recent_analyst_action_count",
        "recent_earnings_count",
        "recent_management_count",
        "recent_other_count",
    ]

    with file_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for trade in trades:
            writer.writerow(trade)

    return file_path


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
                f"ret_1d={fmt_pct(trade['ret_1d'])} "
                f"ret_3d={fmt_pct(trade['ret_3d'])} "
                f"ret_5d={fmt_pct(trade['ret_5d'])} "
                f"mfe_5d={fmt_pct(trade['mfe_5d'])} "
                f"mae_5d={fmt_pct(trade['mae_5d'])} "
                f"recent_news_count={trade['recent_news_count']} "
                f"dominant_news_category={trade['dominant_news_category']} "
                f"recent_upgrade_count={trade['recent_upgrade_count']} "
                f"recent_analyst_action_count={trade['recent_analyst_action_count']} "
                f"recent_earnings_count={trade['recent_earnings_count']} "
                f"recent_management_count={trade['recent_management_count']} "
                f"recent_other_count={trade['recent_other_count']}"
            )

        print_summary("all_signals", trades)
        print_summary("no_news", [t for t in trades if t["recent_news_count"] == 0])
        print_summary("with_news", [t for t in trades if t["recent_news_count"] > 0])

        categories = sorted(set(t["dominant_news_category"] for t in trades))
        for category in categories:
            subset = [t for t in trades if t["dominant_news_category"] == category]
            if len(subset) >= 2:
                print_summary(f"category={category}", subset)

        csv_path = export_trades_csv(args.symbol, args.mode, trades)
        print(f"\ncsv_export={csv_path}")

    finally:
        if ib.isConnected():
            ib.disconnect()


if __name__ == "__main__":
    main()
