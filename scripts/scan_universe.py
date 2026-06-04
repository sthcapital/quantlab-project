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
from quantlab.storage import append_trades_to_db, ensure_dirs
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

    print(f"\n{'─'*60}")
    print(f"  {len(results)} actionable setup(s) — ranked by conviction")
    print(f"{'─'*60}")

    for i, r in enumerate(results, 1):
        stop_str = f"  stop={r.atr_stop:.2f}" if r.atr_stop else ""
        rv_str = f"  rel_vol={r.rel_volume:.2f}x" if r.rel_volume else ""
        print(
            f"  {i:2d}. {r.symbol:<8} "
            f"conviction={r.conviction_score:.2f}  "
            f"close={r.entry_close:.2f}  "
            f"signal={r.signal_type}  "
            f"regime={'bull' if r.regime_bullish else 'bear'}  "
            f"news={r.news_category}({r.news_count})"
            f"{rv_str}{stop_str}"
        )

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
