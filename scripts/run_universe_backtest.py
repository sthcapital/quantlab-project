"""
scripts/run_universe_backtest.py — Universe walk-forward backtest.

Runs a walk-forward backtest across a symbol universe, stores every result
in DuckDB, and prints a ranked summary by average out-of-sample Sharpe.

Usage (mock provider — no IBKR required):
    python scripts/run_universe_backtest.py \
        --provider mock --universe sp500_sample \
        --signal breakout --lookback 5

Usage (IBKR — TWS must be running):
    python scripts/run_universe_backtest.py \
        --provider ibkr --universe sp500_sample \
        --signal breakout --lookback 5 \
        --start 2023-01-02 --end 2025-12-31 \
        --save-db
"""

from argparse import ArgumentParser
from datetime import date

from quantlab.backtest import run_universe_backtest, print_universe_ranking
from quantlab.execution import load_universe
from quantlab.providers import create_market_data_provider
from quantlab.storage import (
    append_backtest_run,
    append_walk_forward_windows,
    ensure_dirs,
)
from quantlab.utils import get_config, make_run_id, n_days_ago, parse_date, setup_logging


def _ibkr_provider(args, ibkr_cfg: dict):
    """Build an IbkrProvider after a pre-flight TWS ping."""
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
    cfg = get_config("backtest")
    ibkr_cfg = get_config("ibkr")

    parser = ArgumentParser(
        description="Run walk-forward backtest across a symbol universe."
    )
    parser.add_argument(
        "--provider", default="mock", choices=["ibkr", "mock", "http"],
        help="Market data provider (default: mock)",
    )
    parser.add_argument(
        "--universe", default="small",
        help="Universe: small | sp500_sample | AAPL,MSFT,... (default: small)",
    )
    parser.add_argument(
        "--signal", choices=["breakout", "sma"], default="breakout",
    )
    parser.add_argument("--lookback", type=int, default=5)
    parser.add_argument(
        "--start", type=parse_date, default=n_days_ago(730),
        help="Bar history start date YYYY-MM-DD (default: 2 years ago)",
    )
    parser.add_argument(
        "--end", type=parse_date, default=date.today(),
        help="Bar history end date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--is-bars", type=int, default=252,
        help="In-sample window length in bars (default: 252 ≈ 1 yr)",
    )
    parser.add_argument(
        "--oos-bars", type=int, default=63,
        help="Out-of-sample window length in bars (default: 63 ≈ 1 qtr)",
    )
    parser.add_argument("--cost-bps", type=float, default=cfg["cost_bps"])
    parser.add_argument(
        "--top-n", type=int, default=10,
        help="Number of symbols to show in ranking (default: 10)",
    )
    parser.add_argument(
        "--save-db", action="store_true",
        help="Persist all results to quantlab.duckdb",
    )
    parser.add_argument("--host", default=ibkr_cfg["host"])
    parser.add_argument("--port", type=int, default=ibkr_cfg["port"])
    parser.add_argument("--client-id", type=int, default=ibkr_cfg["client_id"])
    args = parser.parse_args()

    symbols = load_universe(args.universe)
    run_id = make_run_id(args.universe.upper(), args.signal)

    print(f"\n{'='*66}")
    print(f"  QuantLab Universe Backtest")
    print(f"  run_id   : {run_id}")
    print(f"  universe : {args.universe} ({len(symbols)} symbols)")
    print(f"  provider : {args.provider}")
    print(f"  signal   : {args.signal}  lookback={args.lookback}  cost={args.cost_bps} bps")
    print(f"  dates    : {args.start} → {args.end}")
    print(f"  windows  : IS={args.is_bars} bars  OOS={args.oos_bars} bars")
    print(f"  save-db  : {args.save_db}")
    print(f"{'='*66}\n")

    # ── Build provider ─────────────────────────────────────────────────────────
    if args.provider == "ibkr":
        provider = _ibkr_provider(args, ibkr_cfg)
    else:
        provider = create_market_data_provider(args.provider)

    # ── Run — use persistent IBKR connection to avoid per-symbol reconnects ───
    if args.provider == "ibkr":
        from quantlab.providers.ibkr import IbkrProvider
        with provider:
            results = run_universe_backtest(
                provider, symbols, args.start, args.end,
                signal_type=args.signal,
                lookback=args.lookback,
                is_bars=args.is_bars,
                oos_bars=args.oos_bars,
                cost_bps=args.cost_bps,
                verbose=True,
            )
    else:
        results = run_universe_backtest(
            provider, symbols, args.start, args.end,
            signal_type=args.signal,
            lookback=args.lookback,
            is_bars=args.is_bars,
            oos_bars=args.oos_bars,
            cost_bps=args.cost_bps,
            verbose=True,
        )

    # ── Persist ────────────────────────────────────────────────────────────────
    if args.save_db and results:
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
        print(f"  {len(results)} symbol runs stored (run_id prefix: {run_id})")

    # ── Print ranking ──────────────────────────────────────────────────────────
    print_universe_ranking(results, top_n=args.top_n)


if __name__ == "__main__":
    main()
