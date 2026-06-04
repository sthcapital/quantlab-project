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
    from quantlab.news import fetch_news

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

        from quantlab.news import tag_trades_with_news
        tagged = tag_trades_with_news(trades, news_items, lookback_days=7)
        print(f"  {tagged}/{len(trades)} trade(s) tagged with news")

    except Exception as exc:
        print(f"  [news] Fetch failed: {exc} — continuing without news tagging")
    finally:
        if ib.isConnected():
            ib.disconnect()


def _wyckoff_analysis(bars: list, trade_records: list, signal_type: str, lookback: int) -> None:
    """
    For each trade entry in trade_records, compute Wyckoff scores on the bar
    slice available at that point and print a per-layer pass/fail breakdown.

    This shows how many of the raw breakout signals also had a structurally
    confirmed Wyckoff base, vs signals that fired on thinner setups.
    """
    from quantlab.signals.wyckoff import (
        absorption_score,
        base_quality_score,
        volume_character_score,
        is_wyckoff_spring,
    )
    from quantlab.execution import score_conviction, ScanResult

    total = len(trade_records)
    if total == 0:
        print("\nWyckoff analysis: no trades to evaluate.")
        return

    # Build a date → bar-index map for O(1) lookup
    date_to_idx = {b.as_of.isoformat(): i for i, b in enumerate(bars)}

    counts = dict(base=0, absorption=0, vol_char=0, spring=0, any_wyckoff=0, high_conv=0)
    per_trade = []

    for trade in trade_records:
        idx = date_to_idx.get(trade.entry_date)
        if idx is None:
            continue
        slice_ = bars[: idx + 1]

        bq  = base_quality_score(slice_)
        ab  = absorption_score(slice_)
        vc  = volume_character_score(slice_)
        spr = is_wyckoff_spring(slice_)

        r = ScanResult(
            symbol=trade.symbol, scan_date=trade.entry_date,
            signal_type=signal_type, signal=True,
            entry_close=trade.entry_price, indicator_value=None,
            lookback=lookback,
            base_quality=bq, absorption=ab, volume_character=vc,
            wyckoff_spring=spr,
        )
        conv = score_conviction(r)

        if bq  >= 0.6: counts["base"] += 1
        if ab  >= 0.6: counts["absorption"] += 1
        if vc  >= 0.6: counts["vol_char"] += 1
        if spr:        counts["spring"] += 1
        if bq >= 0.6 or ab >= 0.6 or vc >= 0.6 or spr:
            counts["any_wyckoff"] += 1
        if conv >= 0.45:   # signal(0.30) + at least one Wyckoff layer
            counts["high_conv"] += 1

        per_trade.append((trade.entry_date, bq, ab, vc, spr, conv,
                          trade.trade_return))

    pct = lambda n: f"{n:>3}/{total}  ({n/total*100:4.1f}%)"

    print(f"\n{'='*62}")
    print(f"  Wyckoff Filter Analysis  ({total} breakout signals)")
    print(f"{'='*62}")
    print(f"  Base quality ≥ 0.6   : {pct(counts['base'])}")
    print(f"  Absorption ≥ 0.6     : {pct(counts['absorption'])}")
    print(f"  Vol character ≥ 0.6  : {pct(counts['vol_char'])}")
    print(f"  Spring detected      : {pct(counts['spring'])}")
    print(f"  Any Wyckoff layer    : {pct(counts['any_wyckoff'])}")
    print(f"  Conviction ≥ 0.45    : {pct(counts['high_conv'])}")
    print(f"{'='*62}")

    # Compare avg trade return: Wyckoff confirmed vs unconfirmed
    wyckoff_rets = [t[6] for t in per_trade
                    if t[6] is not None and (t[1] >= 0.6 or t[2] >= 0.6 or t[3] >= 0.6 or t[4])]
    plain_rets   = [t[6] for t in per_trade
                    if t[6] is not None and not (t[1] >= 0.6 or t[2] >= 0.6 or t[3] >= 0.6 or t[4])]

    def _avg_ret(rets):
        return f"{sum(rets)/len(rets)*100:+.3f}%  (n={len(rets)})" if rets else "  N/A"

    print(f"  Avg return — Wyckoff confirmed : {_avg_ret(wyckoff_rets)}")
    print(f"  Avg return — plain signal only : {_avg_ret(plain_rets)}")
    print(f"{'='*62}")


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

    # ── Wyckoff filter analysis ───────────────────────────────────────────────
    _wyckoff_analysis(bars, trade_records, args.signal, args.lookback)

    # ── Export ────────────────────────────────────────────────────────────────
    run_id = make_run_id(args.symbol, args.signal)
    csv_path = export_trades_csv(args.symbol, args.signal, trade_records, run_tag=run_id)
    print(f"\ncsv → {csv_path}")

    if args.save_db:
        append_trades_to_db(run_id, args.signal, args.lookback, trade_records)
        print(f"db  → appended {len(trade_records)} trades (run_id={run_id})")


if __name__ == "__main__":
    main()
