"""
scripts/run_universe_backtest.py — Universe walk-forward backtest with news tagging.

Runs walk-forward validation across a symbol universe, optionally tags every
trade with IBKR historical news, stores results in DuckDB, and prints:
  1. Top-N symbols ranked by average OOS Sharpe
  2. News vs no-news return lift table

Usage (mock, no IBKR required):
    python scripts/run_universe_backtest.py \\
        --provider mock --universe sp500_sample --signal breakout --lookback 5

Usage (IBKR with news, TWS must be running):
    python scripts/run_universe_backtest.py \\
        --provider ibkr --universe sp500_sample \\
        --signal breakout --lookback 5 \\
        --start 2024-01-02 --end 2025-12-31 \\
        --save-db --top-n 15
"""

from __future__ import annotations

import time
from argparse import ArgumentParser
from datetime import date, datetime

from quantlab.backtest import run_universe_backtest, print_universe_ranking
from quantlab.execution import load_universe
from quantlab.providers import create_market_data_provider
from quantlab.risk import fmt_pct, fmt_float
from quantlab.storage import (
    append_backtest_run,
    append_trades_to_db,
    append_walk_forward_windows,
    ensure_dirs,
)
from quantlab.utils import get_config, make_run_id, n_days_ago, parse_date, setup_logging


# ── News tagging ───────────────────────────────────────────────────────────────

def _tag_universe_with_news(results: list, args, ibkr_cfg: dict) -> None:
    """
    Fetch IBKR historical news for every symbol and tag its baseline trades.

    Opens a dedicated connection on news_client_id (separate from the bar-fetch
    client_id) so both can coexist in a single TWS session.
    """
    from ib_insync import IB, Stock
    from quantlab.news import fetch_news, tag_trades_with_news

    news_client_id = ibkr_cfg.get("news_client_id", args.client_id + 40)
    total = len(results)
    start_dt = datetime.combine(args.start, datetime.min.time())
    end_dt = datetime.combine(args.end, datetime.max.time().replace(microsecond=0))

    print(f"\nFetching news ({args.start} → {args.end}, client_id={news_client_id}) ...")
    ib = IB()
    try:
        ib.connect(args.host, args.port, clientId=news_client_id,
                   timeout=ibkr_cfg.get("timeout", 10))

        for i, r in enumerate(results, 1):
            try:
                contract = Stock(r.symbol, "SMART", "USD")
                qualified = ib.qualifyContracts(contract)
                if not qualified:
                    print(f"  [{i:>2}/{total}] {r.symbol:<8}  SKIP — could not qualify")
                    continue

                news_items = fetch_news(
                    ib, qualified[0],
                    start_dt=start_dt, end_dt=end_dt,
                    limit=200,
                )
                all_trades = r.baseline.trades
                tagged = tag_trades_with_news(all_trades, news_items, lookback_days=7)
                print(
                    f"  [{i:>2}/{total}] {r.symbol:<8}  "
                    f"{len(news_items):>4} headlines  "
                    f"{tagged:>3}/{len(all_trades)} trades tagged"
                )
                time.sleep(0.5)   # light pacing between news requests

            except Exception as exc:
                print(f"  [{i:>2}/{total}] {r.symbol:<8}  ERROR — {exc}")

    except Exception as exc:
        print(f"[news] Connection failed: {exc} — skipping news tagging")
    finally:
        if ib.isConnected():
            ib.disconnect()


# ── News vs no-news lift table ─────────────────────────────────────────────────

def print_news_lift_table(results: list, top_n: int = 15) -> None:
    """
    Print news vs no-news trade return lift for each symbol.

    Lift = avg_return(trades with news) − avg_return(trades without news).
    Symbols with no completed trades or no news-tagged trades are excluded.
    Sorted by lift descending to surface the strongest news edge.
    """
    rows = []
    for r in results:
        completed = [t for t in r.baseline.trades if t.trade_return is not None]
        with_news  = [t.trade_return for t in completed if t.news_count > 0]
        no_news    = [t.trade_return for t in completed if t.news_count == 0]

        if not with_news and not no_news:
            continue

        avg_news    = sum(with_news)  / len(with_news)  if with_news  else None
        avg_no_news = sum(no_news)    / len(no_news)    if no_news    else None
        lift = (avg_news - avg_no_news) if (avg_news is not None and avg_no_news is not None) else None

        # Most common news category among news-tagged trades
        cats = [t.news_category for t in completed if t.news_count > 0]
        dominant = max(set(cats), key=cats.count) if cats else "—"

        rows.append(dict(
            symbol=r.symbol,
            n_news=len(with_news),
            n_no_news=len(no_news),
            avg_news=avg_news,
            avg_no_news=avg_no_news,
            lift=lift,
            category=dominant,
        ))

    rows.sort(key=lambda x: (x["lift"] is None, -(x["lift"] or 0)))

    hdr = (
        f"{'#':>3}  {'sym':<8}  {'news n':>7}  {'avg news':>9}  "
        f"{'no-news n':>10}  {'avg no-news':>12}  {'lift':>8}  {'top cat':<14}"
    )
    sep = "=" * len(hdr)
    print(f"\n{sep}")
    print("  News vs No-News Return Lift")
    print(sep)
    print(hdr)
    print("-" * len(hdr))

    for rank, row in enumerate(rows[:top_n], 1):
        print(
            f"{rank:>3}.  {row['symbol']:<8}  "
            f"{row['n_news']:>7}  {fmt_pct(row['avg_news']):>9}  "
            f"{row['n_no_news']:>10}  {fmt_pct(row['avg_no_news']):>12}  "
            f"{fmt_pct(row['lift']):>8}  {row['category']:<14}"
        )

    print(sep)
    no_news_syms = sum(1 for r in rows if r["n_news"] == 0)
    if no_news_syms:
        print(f"  {no_news_syms} symbol(s) had no news-tagged trades (excluded)")


# ── Main ───────────────────────────────────────────────────────────────────────

def _ibkr_provider(args, ibkr_cfg: dict):
    from quantlab.providers.ibkr import IbkrProvider, ping_tws
    if not ping_tws(args.host, args.port):
        raise SystemExit(
            f"\nTWS / IB Gateway is not reachable at {args.host}:{args.port}.\n"
            "Start TWS or IB Gateway, enable API access, and try again."
        )
    return IbkrProvider(
        host=args.host,
        port=args.port,
        client_id=args.client_id,
        spot_client_id=ibkr_cfg["spot_client_id"],
    )


def main() -> None:
    setup_logging()
    ensure_dirs()
    cfg      = get_config("backtest")
    ibkr_cfg = get_config("ibkr")

    parser = ArgumentParser(
        description="Run walk-forward backtest across a symbol universe."
    )
    parser.add_argument("--provider", default="mock", choices=["ibkr", "mock", "http"])
    parser.add_argument(
        "--universe", default="small",
        help="small | sp500_sample | AAPL,MSFT,... (default: small)",
    )
    parser.add_argument("--signal", choices=["breakout", "sma"], default="breakout")
    parser.add_argument("--lookback", type=int, default=5)
    parser.add_argument("--start",  type=parse_date, default=n_days_ago(730))
    parser.add_argument("--end",    type=parse_date, default=date.today())
    parser.add_argument("--is-bars",  type=int, default=252)
    parser.add_argument("--oos-bars", type=int, default=63)
    parser.add_argument("--cost-bps", type=float, default=cfg["cost_bps"])
    parser.add_argument("--top-n",   type=int, default=10)
    parser.add_argument("--save-db", action="store_true")
    parser.add_argument("--no-news", action="store_true",
                        help="Skip IBKR news fetch (faster price-only run)")
    parser.add_argument("--host",      default=ibkr_cfg["host"])
    parser.add_argument("--port",      type=int, default=ibkr_cfg["port"])
    parser.add_argument("--client-id", type=int, default=ibkr_cfg["client_id"])
    args = parser.parse_args()

    symbols = load_universe(args.universe)
    run_id  = make_run_id(args.universe.upper(), args.signal)

    print(f"\n{'='*66}")
    print(f"  QuantLab Universe Backtest")
    print(f"  run_id   : {run_id}")
    print(f"  universe : {args.universe} ({len(symbols)} symbols)")
    print(f"  provider : {args.provider}")
    print(f"  signal   : {args.signal}  lookback={args.lookback}  cost={args.cost_bps} bps")
    print(f"  dates    : {args.start} → {args.end}")
    print(f"  windows  : IS={args.is_bars} bars  OOS={args.oos_bars} bars")
    print(f"  news     : {'disabled' if args.no_news or args.provider != 'ibkr' else 'enabled'}")
    print(f"  save-db  : {args.save_db}")
    print(f"{'='*66}\n")

    # ── Build provider ─────────────────────────────────────────────────────────
    if args.provider == "ibkr":
        provider = _ibkr_provider(args, ibkr_cfg)
    else:
        provider = create_market_data_provider(args.provider)

    # ── Run universe backtest (persistent IBKR connection for bars) ────────────
    if args.provider == "ibkr":
        with provider:
            results = run_universe_backtest(
                provider, symbols, args.start, args.end,
                signal_type=args.signal, lookback=args.lookback,
                is_bars=args.is_bars, oos_bars=args.oos_bars,
                cost_bps=args.cost_bps, verbose=True,
            )
    else:
        results = run_universe_backtest(
            provider, symbols, args.start, args.end,
            signal_type=args.signal, lookback=args.lookback,
            is_bars=args.is_bars, oos_bars=args.oos_bars,
            cost_bps=args.cost_bps, verbose=True,
        )

    if not results:
        print("No results produced.")
        return

    # ── Tag trades with news (separate connection, news_client_id) ─────────────
    if args.provider == "ibkr" and not args.no_news:
        _tag_universe_with_news(results, args, ibkr_cfg)

    # ── Persist to DuckDB ──────────────────────────────────────────────────────
    if args.save_db:
        print("\nPersisting to DuckDB...")
        for r in results:
            sym_run_id = f"{run_id}_{r.symbol}"
            append_backtest_run(
                sym_run_id, r.symbol, args.signal, args.lookback,
                args.start, args.end, r.baseline.metrics,
            )
            append_walk_forward_windows(
                sym_run_id, r.symbol, args.signal, args.lookback, r.windows
            )
            completed = [t for t in r.baseline.trades if t.trade_return is not None]
            if completed:
                append_trades_to_db(sym_run_id, args.signal, args.lookback, completed)
        print(f"  {len(results)} symbol runs stored (run_id prefix: {run_id})")

    # ── Print OOS Sharpe ranking ───────────────────────────────────────────────
    print_universe_ranking(results, top_n=args.top_n)

    # ── Print news vs no-news lift (IBKR runs only) ────────────────────────────
    if args.provider == "ibkr" and not args.no_news:
        print_news_lift_table(results, top_n=args.top_n)


if __name__ == "__main__":
    main()
