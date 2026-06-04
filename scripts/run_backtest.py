"""
scripts/run_backtest.py — Full backtest with all metrics and DuckDB storage.

Migrated from the flat script to call quantlab modules.
Adds: Sharpe, Sortino, Calmar, profit factor, expectancy, transaction costs,
      min-sample enforcement, DuckDB persistence, Parquet bar storage.

Usage:
    python scripts/run_backtest.py --provider ibkr --symbol AAPL \
        --start 2025-01-01 --end 2026-06-03 --signal breakout --lookback 20
"""

from argparse import ArgumentParser

from quantlab.providers import create_market_data_provider
from quantlab.signals import breakout_signal, sma_signal, atr_stop_price
from quantlab.research import forward_returns, TradeRecord, compute_metrics, MIN_TRADES
from quantlab.risk import (
    apply_transaction_cost, print_metrics, print_grouped_summaries, fmt_pct,
)
from quantlab.storage import export_trades_csv, save_bars_parquet, append_trades_to_db, ensure_dirs
from quantlab.utils import setup_logging, parse_date, make_run_id, get_config


def _tag_trades_with_news(
    trades: list,
    symbol: str,
    start_date,
    end_date,
    host: str,
    port: int,
    news_client_id: int,
    timeout: int,
) -> None:
    """
    Fetch historical IBKR news for the backtest period and tag each trade.

    For every TradeRecord, computes a 7-day pre-signal news window and writes
    news_count, news_category, news_k_score, news_c_score in place.
    Silently skips on any connection or fetch error so the backtest still
    prints even if news is unavailable.
    """
    from datetime import datetime
    from ib_insync import IB, Stock
    from quantlab.news import fetch_news, compute_news_features

    print(f"\nFetching news for {symbol} ({start_date} → {end_date}) ...", flush=True)
    ib = IB()
    try:
        ib.connect(host, port, clientId=news_client_id, timeout=timeout)
        contract = Stock(symbol, "SMART", "USD")
        qualified = ib.qualifyContracts(contract)
        if not qualified:
            print(f"  [news] Could not qualify {symbol} — skipping")
            return

        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.max.time().replace(microsecond=0))

        news_items = fetch_news(
            ib, qualified[0],
            start_dt=start_dt,
            end_dt=end_dt,
            limit=500,
        )
        print(f"  {len(news_items)} headline(s) fetched")

        tagged = 0
        for trade in trades:
            feat = compute_news_features(news_items, trade.signal_date, lookback_days=7)
            if feat.has_news():
                trade.news_count = feat.total_count
                trade.news_category = feat.dominant_category
                trade.news_k_score = feat.avg_k_score
                trade.news_c_score = feat.avg_c_score
                tagged += 1

        print(f"  {tagged}/{len(trades)} trade(s) tagged with news")

    except Exception as exc:
        print(f"  [news] Fetch failed: {exc} — continuing without news tagging")
    finally:
        if ib.isConnected():
            ib.disconnect()


def main() -> None:
    setup_logging()
    ensure_dirs()
    cfg = get_config("backtest")
    ibkr_cfg = get_config("ibkr")

    parser = ArgumentParser(description="Run a full backtest with institutional metrics.")
    parser.add_argument("--provider", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", required=True, type=parse_date)
    parser.add_argument("--end", required=True, type=parse_date)
    parser.add_argument("--signal", choices=["sma", "breakout"], default="breakout")
    parser.add_argument("--lookback", type=int, default=cfg["lookback"])
    parser.add_argument("--initial-capital", type=float, default=cfg["initial_capital"])
    parser.add_argument("--cost-bps", type=float, default=cfg["cost_bps"])
    parser.add_argument("--host", default=ibkr_cfg["host"])
    parser.add_argument("--port", type=int, default=ibkr_cfg["port"])
    parser.add_argument("--client-id", type=int, default=ibkr_cfg["client_id"])
    parser.add_argument("--save-parquet", action="store_true", help="Save bars to Parquet")
    parser.add_argument("--save-db", action="store_true", help="Append trades to DuckDB")
    parser.add_argument("--no-news", action="store_true",
                        help="Skip news fetch (faster, price-only backtest)")
    args = parser.parse_args()

    # ── Fetch bars ────────────────────────────────────────────────────────────
    provider_kwargs = {}
    if args.provider.lower() == "ibkr":
        from quantlab.providers.ibkr import ping_tws
        if not ping_tws(args.host, args.port):
            raise SystemExit(
                f"\nTWS / IB Gateway is not reachable at {args.host}:{args.port}.\n"
                "Start TWS or IB Gateway, enable API access, and try again."
            )
        provider_kwargs = {"host": args.host, "port": args.port, "client_id": args.client_id}

    provider = create_market_data_provider(args.provider, **provider_kwargs)
    bars = list(provider.get_daily_bars(args.symbol, args.start, args.end))

    if len(bars) <= args.lookback:
        raise SystemExit(f"Need more than {args.lookback} bars, received {len(bars)}.")

    if args.save_parquet:
        p = save_bars_parquet(args.symbol, bars)
        print(f"bars saved → {p}")

    print(f"\nsymbol={args.symbol}  bars={len(bars)}  signal={args.signal}  lookback={args.lookback}")

    # ── Generate signals and simulate ─────────────────────────────────────────
    trade_records: list[TradeRecord] = []
    positions = [0]
    equity_curve = [args.initial_capital]
    strategy_returns = [0.0]

    for i in range(1, len(bars)):
        bar_slice = bars[: i + 1]

        if args.signal == "breakout":
            result = breakout_signal(bar_slice, args.symbol, args.lookback)
        else:
            result = sma_signal(bar_slice, args.symbol, args.lookback)

        sig = result.signal if result else False

        # Next-bar execution — position today = yesterday's signal
        pos = positions[-1]
        positions.append(1 if sig else 0)

        daily_ret = (bars[i].close / bars[i - 1].close) - 1.0
        strat_ret = pos * daily_ret
        next_equity = equity_curve[-1] * (1.0 + strat_ret)
        equity_curve.append(next_equity)
        strategy_returns.append(strat_ret)

        # Record trade transitions
        if positions[-2] == 0 and positions[-1] == 1:
            # Entry
            fwd = forward_returns(bars, i, bars[i].close)
            stop = atr_stop_price(bars[: i + 1], bars[i].close)
            net_ret = apply_transaction_cost(fwd.get("ret_5d") or 0.0, args.cost_bps)
            trade_records.append(
                TradeRecord(
                    symbol=args.symbol,
                    signal_date=bars[i].as_of.isoformat(),
                    entry_date=bars[i].as_of.isoformat(),
                    entry_price=bars[i].close,
                    exit_date=None,
                    exit_price=None,
                    trade_return=None,  # filled on exit
                    ret_1d=fwd.get("ret_1d"),
                    ret_3d=fwd.get("ret_3d"),
                    ret_5d=fwd.get("ret_5d"),
                    mfe_5d=fwd.get("mfe_5d"),
                    mae_5d=fwd.get("mae_5d"),
                    atr_stop=stop,
                    cost_bps=args.cost_bps,
                )
            )
        elif positions[-2] == 1 and positions[-1] == 0 and trade_records:
            # Exit — fill the open trade
            last = trade_records[-1]
            raw_ret = (bars[i].close / last.entry_price) - 1.0
            trade_records[-1].exit_date = bars[i].as_of.isoformat()
            trade_records[-1].exit_price = bars[i].close
            trade_records[-1].trade_return = apply_transaction_cost(raw_ret, args.cost_bps)

    # ── Tag trades with news (IBKR only, skip when --no-news) ────────────────
    if args.provider.lower() == "ibkr" and not args.no_news and trade_records:
        _tag_trades_with_news(
            trade_records,
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
            host=args.host,
            port=args.port,
            news_client_id=ibkr_cfg.get("news_client_id", args.client_id + 40),
            timeout=ibkr_cfg.get("timeout", 10),
        )

    # ── Compute and print full metrics ────────────────────────────────────────
    metrics = compute_metrics(
        symbol=args.symbol,
        signal_type=args.signal,
        lookback=args.lookback,
        bars=bars,
        trades=trade_records,
        equity_curve=equity_curve,
        strategy_returns=strategy_returns,
        positions=positions,
    )

    print_metrics(metrics)
    print_grouped_summaries(trade_records)

    # ── Export ────────────────────────────────────────────────────────────────
    run_id = make_run_id(args.symbol, args.signal)
    csv_path = export_trades_csv(args.symbol, args.signal, trade_records, run_tag=run_id)
    print(f"\ncsv → {csv_path}")

    if args.save_db:
        append_trades_to_db(run_id, args.signal, args.lookback, trade_records)
        print(f"db  → appended {len(trade_records)} trades (run_id={run_id})")


if __name__ == "__main__":
    main()
