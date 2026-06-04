"""
scripts/scan_universe.py — Daily market scanner.

Scans a symbol universe, scores conviction on each setup, and prints
a ranked list of actionable signals ready for the risk gate.

Usage:
    python scripts/scan_universe.py --universe small --signal breakout --lookback 20
    python scripts/scan_universe.py --universe sp500_sample --signal breakout --min-conviction 0.5
    python scripts/scan_universe.py --symbols AAPL,MSFT,NVDA,TSLA --signal sma
"""

from argparse import ArgumentParser
from datetime import date

from quantlab.execution import run_universe_scan, load_universe
from quantlab.providers import create_market_data_provider
from quantlab.risk import fmt_pct, fmt_float
from quantlab.storage import append_scan_results, ensure_dirs
from quantlab.utils import setup_logging, parse_date, n_days_ago, make_run_id, get_config


def main() -> None:
    setup_logging()
    cfg = get_config("scanner")
    ibkr_cfg = get_config("ibkr")

    parser = ArgumentParser(description="Scan a universe of stocks for high-conviction setups.")
    parser.add_argument("--universe", default=cfg["universe"],
                        help="Universe name: small, sp500_sample, or comma-separated symbols")
    parser.add_argument("--symbols", default=None,
                        help="Override universe with comma-separated symbols")
    parser.add_argument("--signal", choices=["breakout", "sma"], default=cfg["signal_type"])
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--start", default=n_days_ago(365).isoformat(),
                        help="Bar history start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=date.today().isoformat(),
                        help="Bar history end date (YYYY-MM-DD)")
    parser.add_argument("--min-conviction", type=float, default=cfg["min_conviction"])
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--provider", default="ibkr")
    parser.add_argument("--host", default=ibkr_cfg["host"])
    parser.add_argument("--port", type=int, default=ibkr_cfg["port"])
    parser.add_argument("--client-id", type=int, default=ibkr_cfg["client_id"])
    parser.add_argument("--no-news", action="store_true",
                        help="Skip news fetching (faster, price-only scan)")
    parser.add_argument("--with-options", action="store_true",
                        help="Enrich results with IBKR options flow (PCR, IV skew, unusual calls)")
    parser.add_argument("--save-db", action="store_true",
                        help="Persist all scan results (not just actionable) to DuckDB")
    parser.add_argument("--add-to-watchlist", action="store_true",
                        help="Add setups scoring >= 0.70 to the DuckDB watchlist table")
    parser.add_argument("--multi-lookback", action="store_true",
                        help="Run a secondary scan to confirm signals across two lookbacks")
    parser.add_argument("--secondary-lookback", type=int, default=20,
                        help="Secondary lookback for multi-lookback confirmation (default 20)")
    args = parser.parse_args()

    ensure_dirs()

    symbols = (
        [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if args.symbols
        else load_universe(args.universe)
    )

    start_date = parse_date(args.start)
    end_date = parse_date(args.end)

    provider_kwargs = {}
    if args.provider == "ibkr":
        from quantlab.providers.ibkr import ping_tws
        if not ping_tws(args.host, args.port):
            raise SystemExit(
                f"\nTWS / IB Gateway is not reachable at {args.host}:{args.port}.\n"
                "Start TWS or IB Gateway, enable API access, and try again."
            )
        news_client_id = get_config("ibkr").get("news_client_id", args.client_id + 40)
        provider_kwargs = {"host": args.host, "port": args.port, "client_id": args.client_id}
    provider = create_market_data_provider(args.provider, **provider_kwargs)

    # Connect IBKR for news if not skipped
    ibkr_conn = None
    if not args.no_news and args.provider == "ibkr":
        try:
            from ib_insync import IB
            ibkr_conn = IB()
            ibkr_conn.connect(args.host, args.port, clientId=news_client_id, timeout=10)
        except Exception as e:
            print(f"[scanner] News connection failed ({e}) — running price-only scan")
            ibkr_conn = None

    print(f"\n{'='*60}")
    print(f"  QuantLab Universe Scanner")
    print(f"  {len(symbols)} symbols | signal={args.signal} | lookback={args.lookback}")
    print(f"  {start_date} → {end_date} | min_conviction={args.min_conviction}")
    print(f"{'='*60}\n")

    try:
        results = run_universe_scan(
            provider=provider,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            signal_type=args.signal,
            lookback=args.lookback,
            min_conviction=args.min_conviction,
            cost_bps=args.cost_bps,
            ibkr_connection=ibkr_conn,
        )
    finally:
        if ibkr_conn and ibkr_conn.isConnected():
            ibkr_conn.disconnect()

    if not results:
        print("No actionable setups found today.\n")
        return

    # ── Breadth tape summary ───────────────────────────────────────────────────
    from quantlab.signals.breadth import get_latest_snapshot
    _snap = get_latest_snapshot()
    if _snap:
        print(f"\n  {_snap.summary_line()}")
    else:
        print("\n  Breadth: no data (run update_breadth.py after market close)")

    # ── Multi-lookback confirmation ────────────────────────────────────────────
    # Run a fast secondary scan (bars already cached) to find symbols that also
    # fire a breakout signal at the secondary lookback.  Symbols confirmed on
    # both lookbacks earn a +0.05 structural-confirmation bonus and get a ✓ marker.
    if args.multi_lookback:
        from quantlab.execution import scan_symbol, score_conviction
        secondary_fired: set[str] = set()
        for symbol in symbols:
            try:
                bars2 = list(provider.get_daily_bars(symbol, start_date, end_date))
                r2 = scan_symbol(symbol, bars2, signal_type=args.signal,
                                 lookback=args.secondary_lookback)
                if r2 and r2.signal:
                    secondary_fired.add(symbol)
            except Exception:
                pass

        for r in results:
            if r.symbol in secondary_fired:
                r.multi_lookback_confirmed = True
                r.conviction_score = score_conviction(r)

        results.sort(key=lambda r: r.conviction_score, reverse=True)
        n_confirmed = sum(1 for r in results if r.multi_lookback_confirmed)
        print(f"  Multi-lookback (lb={args.lookback}+{args.secondary_lookback}): "
              f"{n_confirmed}/{len(results)} confirmed  ✓")

    print(f"\n{'─'*60}")
    print(f"  {len(results)} actionable setup(s) — ranked by conviction")
    print(f"{'─'*60}")

    for i, r in enumerate(results, 1):
        from quantlab.execution import _SECTOR_ABBREV
        stop_str    = f"  stop={r.atr_stop:.2f}" if r.atr_stop else ""
        rv_str      = f"  rel_vol={r.rel_volume:.2f}x" if r.rel_volume else ""
        ea_str      = f"  ea={r.earnings_acceleration:.2f}" \
                      if r.earnings_acceleration > 0 else "  ea=0.00"
        vol_str     = (
            f"  ar={r.accumulation_ratio:.2f}"
            f"  vt={r.volume_trend:.2f}"
            f"  cv={r.climactic_volume:.2f}"
        )
        multi_str   = " ✓" if r.multi_lookback_confirmed else "  "
        opt_str     = f"  opt={r.options_conviction:.2f}" if r.options_conviction > 0 else ""
        sector_abbr = _SECTOR_ABBREV.get(r.sector, r.sector[:6]) if r.sector else "?"
        sector_str  = f"  [{sector_abbr}{'⚑' if r.sector_cluster else ''}]"
        rs_str      = f"  rs={r.rs_score:.2f}" if r.rs_score > 0 else "  rs=--"
        print(
            f"  {i:2d}. {r.symbol:<8} "
            f"conviction={r.conviction_score:.2f}{multi_str}  "
            f"close={r.entry_close:.2f}  "
            f"signal={r.signal_type}  "
            f"regime={'bull' if r.regime_bullish else 'bear'}  "
            f"news={r.news_category}({r.news_count})"
            f"{ea_str}{rs_str}{opt_str}{sector_str}{vol_str}{rv_str}{stop_str}"
        )

    print(f"\n{'='*60}\n")

    # ── Options flow enrichment (IBKR connection on options_client_id) ─────────
    if args.with_options and args.provider == "ibkr" and results:
        from quantlab.signals.options_flow import options_conviction_score
        from quantlab.execution import score_conviction
        from ib_insync import IB

        options_client_id = ibkr_cfg.get("options_chain_client_id", 21)
        print(f"Fetching options flow ({len(results)} symbols, "
              f"client_id={options_client_id}) ...")
        ib_opt = IB()
        try:
            ib_opt.connect(args.host, args.port,
                           clientId=options_client_id, timeout=10)
            for r in results:
                try:
                    bars = list(provider.get_daily_bars(r.symbol, start_date, end_date))
                    opt_score = options_conviction_score(r.symbol, bars, ib_opt)
                    r.options_conviction = opt_score
                    r.conviction_score   = score_conviction(r)
                    flag = " ▲" if opt_score >= 0.6 else ""
                    print(f"  {r.symbol:<8}  opt={opt_score:.2f}  "
                          f"conv={r.conviction_score:.2f}{flag}")
                except Exception as e:
                    print(f"  {r.symbol:<8}  options ERROR — {e}")
        except Exception as e:
            print(f"[options] Connection failed: {e} — skipping options enrichment")
        finally:
            if ib_opt.isConnected():
                ib_opt.disconnect()

        results.sort(key=lambda r: r.conviction_score, reverse=True)
        print()

    # ── Persist to DuckDB ─────────────────────────────────────────────────────
    if args.save_db:
        scan_id = make_run_id(args.universe.upper() if not args.symbols else "CUSTOM",
                              args.signal)
        append_scan_results(scan_id, results)
        print(f"db  → {len(results)} scan result(s) stored (scan_id={scan_id})")

    # ── Watchlist ──────────────────────────────────────────────────────────────
    if args.add_to_watchlist:
        from quantlab.watchlist import add_to_watchlist
        added = sum(1 for r in results if add_to_watchlist(r))
        total_qualifying = sum(1 for r in results if r.conviction_score >= 0.70)
        print(f"watchlist → {added}/{total_qualifying} setup(s) added "
              f"(conviction ≥ 0.70)")


if __name__ == "__main__":
    main()
