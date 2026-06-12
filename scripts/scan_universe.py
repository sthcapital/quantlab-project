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
from collections import defaultdict
from datetime import date

from quantlab.execution import run_universe_scan, load_universe
from quantlab.providers import create_market_data_provider
from quantlab.risk import fmt_pct, fmt_float
from quantlab.storage import append_scan_results, ensure_dirs
from quantlab.utils import setup_logging, parse_date, n_days_ago, make_run_id, get_config


def select_top_candidates(
    results,
    iwl_state: dict,
    is_cs_fn,
    max_per_day: int = 5,
    max_per_sector: int = 3,
    options_counts_as_confirmation: bool | None = None,
) -> list:
    """
    Apply the strict paper-trading watchlist filter and return up to max_per_day
    candidates, with sector concentration capped at max_per_sector per sector.

    Qualification rules (ALL must be true):
      - stage == 2
      - is_cs_fn(symbol) → common stock
      - conviction_score >= 0.70
      - earnings_score > 0.0 (real earnings data available)
      - at least one confirming signal: volume_dry_up OR consecutive_days >= 2
        (already on institutional_watchlist).  options_signal counts only when
        options_counts_as_confirmation is enabled (None → scanner config) —
        the unusual-options detector is uncalibrated (347/357 symbols flagged
        on 2026-06-11) and is display-only until the recalibration lands.

    Ranking: consecutive_days DESC → conviction_score DESC → earnings_score DESC.

    Returns:
        List of (ScanResult, earn_score, consecutive_days) tuples.
    """
    if options_counts_as_confirmation is None:
        options_counts_as_confirmation = bool(
            get_config("scanner").get("options_counts_as_confirmation", False)
        )

    qualifying = []
    for r in results:
        if r.stage != 2:
            continue
        if not is_cs_fn(r.symbol):
            continue
        if r.conviction_score < 0.70:
            continue

        earn = (r.edgar_acceleration if r.edgar_acceleration is not None
                else r.earnings_acceleration)
        if earn is None or earn <= 0.0:
            continue

        eps_g = getattr(r, "eps_growth", None)
        if eps_g is not None and eps_g < -0.10:
            continue

        iwl_e = iwl_state.get(r.symbol, {})
        opts  = bool(iwl_e.get("options_signal", False))
        vdu   = bool(iwl_e.get("volume_dry_up", False))
        cdays = int(iwl_e.get("consecutive_days", 1))

        if not ((opts and options_counts_as_confirmation) or vdu or cdays >= 2):
            continue

        qualifying.append((r, earn, cdays))

    qualifying.sort(key=lambda x: (-x[2], -x[0].conviction_score, -x[1]))

    # Sector cap: no more than max_per_sector from the same GICS sector
    sector_count: dict[str, int] = defaultdict(int)
    selected = []
    for item in qualifying:
        sector = getattr(item[0], "sector", "") or "unknown"
        if sector_count[sector] >= max_per_sector:
            continue
        selected.append(item)
        sector_count[sector] += 1
        if len(selected) >= max_per_day:
            break

    return selected


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
    parser.add_argument("--ignore-breadth", action="store_true",
                        help="Skip breadth regime check — don't auto-raise min_conviction")
    args = parser.parse_args()

    ensure_dirs()

    # ── Breadth regime check (before scan — adjusts threshold if bear tape) ───
    from quantlab.signals.breadth import get_latest_snapshot
    breadth_snap      = get_latest_snapshot()
    breadth_override  = False
    bear_tape_active  = False

    if breadth_snap and not args.ignore_breadth:
        _ratio = breadth_snap.ratio_10d
        _mc    = breadth_snap.mcclellan_oscillator
        _ad    = breadth_snap.ad_line
        ratio_str = f"{_ratio:.2f}" if _ratio is not None else "--"
        mc_str    = f"{_mc:+.0f}"   if _mc    is not None else "--"
        ad_str    = f"{_ad:+,}"     if _ad    is not None else "--"
        print(f"\n  Tape: {breadth_snap.tape} | "
              f"10d-ratio={ratio_str} | McClellan={mc_str} | AD={ad_str}")

        if breadth_snap.tape == "BEAR":
            bear_tape_active = True
            if args.min_conviction < 0.80:
                print(f"  WARNING: Bear tape active — "
                      f"conviction threshold raised to 0.80 (was {args.min_conviction:.2f})")
                args.min_conviction = 0.80
    elif not breadth_snap:
        print("\n  Breadth: no data — run update_breadth.py after market close")

    # ── Symbol list / tradeable universe build ────────────────────────────────
    universe_stats = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    elif args.universe in ("tradeable", "tradeable_no_options"):
        from quantlab.universe import UniverseManager
        from datetime import date as _today_date
        today = _today_date.today()
        if args.provider in ("polygon", "ibkr", "flatfile"):
            from quantlab.providers.polygon import PolygonProvider
            pg  = PolygonProvider()
            mgr = UniverseManager()
            symbols, universe_stats = mgr.build_tradeable_universe(
                trade_date       = today,
                polygon_provider = pg,
                ib               = None,
                optionable_only  = False,
            )
            if universe_stats.date != today.isoformat():
                print(
                    f"\n  NOTE: today's grouped daily not yet available — "
                    f"using {universe_stats.date} universe"
                )
            if not symbols:
                print("  No universe data available — falling back to sp500_sample")
                symbols = load_universe("sp500_sample")
            else:
                print(f"\n  {universe_stats.summary()}")
        else:
            print("  Falling back to sp500_sample. Use --provider polygon or flatfile to build.")
            symbols = load_universe("sp500_sample")
    else:
        symbols = load_universe(args.universe)

    start_date = parse_date(args.start)
    end_date = parse_date(args.end)

    # news_client_id is needed regardless of bar provider (IBKR used for news)
    news_client_id = get_config("ibkr").get("news_client_id", args.client_id + 40)

    provider_kwargs = {}
    if args.provider == "ibkr":
        from quantlab.providers.ibkr import ping_tws
        if not ping_tws(args.host, args.port):
            raise SystemExit(
                f"\nTWS / IB Gateway is not reachable at {args.host}:{args.port}.\n"
                "Start TWS or IB Gateway, enable API access, and try again."
            )
        provider_kwargs = {"host": args.host, "port": args.port, "client_id": args.client_id}
    provider = create_market_data_provider(args.provider, **provider_kwargs)

    # Connect IBKR for news when requested — works with any bar provider.
    # For flatfile scans, bars come from local Parquet; news still uses IBKR.
    ibkr_conn = None
    if not args.no_news:
        try:
            from ib_insync import IB
            ibkr_conn = IB()
            ibkr_conn.connect(args.host, args.port, clientId=news_client_id, timeout=10)
        except Exception as e:
            print(f"[scanner] News connection failed ({e}) — running price-only scan")
            ibkr_conn = None

    bear_flag = "  ⚠ BEAR TAPE" if bear_tape_active else ""
    print(f"\n{'='*60}")
    print(f"  QuantLab Universe Scanner{bear_flag}")
    if universe_stats:
        print(f"  {universe_stats.summary()}")
    else:
        print(f"  {len(symbols)} symbols | universe={args.universe}")
    print(f"  signal={args.signal} | lookback={args.lookback}")
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
        edgar_accel_str = (
            f"  edgar_accel={r.edgar_acceleration:.2f}"
            if r.edgar_acceleration is not None else ""
        )
        ohlcv_accel_str = f"  ohlcv_accel={r.earnings_acceleration:.2f}"
        vol_str     = (
            f"  ar={r.accumulation_ratio:.2f}"
            f"  vt={r.volume_trend:.2f}"
            f"  cv={r.climactic_volume:.2f}"
        )
        multi_str   = " ✓" if r.multi_lookback_confirmed else "  "
        _tier       = r.market_cap_tier or "?"
        _opt_val    = r.options_score if r.options_score > 0 else r.options_conviction
        if r.unusual_options_score > 0:
            opt_str = f"  unusual_opts={r.unusual_options_score:.2f} [{_tier}]"
        elif _opt_val > 0:
            opt_str = f"  opt={_opt_val:.2f} [{_tier}]"
        else:
            opt_str = f"  [{_tier}]" if _tier and _tier != "?" else ""
        sector_abbr = _SECTOR_ABBREV.get(r.sector, r.sector[:6]) if r.sector else "?"
        sector_str  = f"  [{sector_abbr}{'⚑' if r.sector_cluster else ''}]"
        rs_str      = f"  rs={r.rs_score:.2f}" if r.rs_score > 0 else "  rs=--"
        # Earnings info: earnings=2026-07-16(41d) [neutral]
        earnings_str = ""
        try:
            from quantlab.providers.edgar import get_next_earnings_date as _gned
            _next = _gned(r.symbol)
            if _next:
                _nd, _tdays = _next
                earnings_str = f"  earnings={_nd.isoformat()}({_tdays}d) [{r.earnings_proximity}]"
            elif r.earnings_proximity != "neutral":
                earnings_str = f"  [{r.earnings_proximity}]"
        except Exception:
            pass
        print(
            f"  {i:2d}. {r.symbol:<8} "
            f"conviction={r.conviction_score:.2f}{multi_str}  "
            f"close={r.entry_close:.2f}  "
            f"signal={r.signal_type}  "
            f"regime={'bull' if r.regime_bullish else 'bear'}  "
            f"news={r.news_category}({r.news_count})"
            f"{edgar_accel_str}{ohlcv_accel_str}{rs_str}{opt_str}{sector_str}{vol_str}{rv_str}{stop_str}"
            f"{earnings_str}"
        )

    print(f"\n{'='*60}\n")

    # ── Options flow enrichment ────────────────────────────────────────────────
    # Primary source: Polygon/Massive (POLYGON_API_KEY) — Greeks, IV, OI, PCR.
    # Fallback: IBKR live chain when provider=ibkr and no Polygon key.
    if args.with_options and results:
        from quantlab.config import settings as _opt_cfg
        from quantlab.execution import score_conviction
        polygon_key = getattr(_opt_cfg, "polygon_api_key", "") or ""

        if polygon_key:
            from quantlab.providers.massive_options import MassiveOptionsProvider
            from quantlab.providers.flat_files import FlatFileProvider
            from quantlab.signals.unusual_options import (
                detect_unusual_activity, score_unusual_activity,
            )
            from quantlab.execution import market_cap_tier as _mct

            mp   = MassiveOptionsProvider(api_key=polygon_key)
            flat = FlatFileProvider()

            # Polygon publishes the daily options flat file hours after the
            # close, so at evening-scan time end_date's file usually doesn't
            # exist yet.  Detecting against a missing file silently zeroes
            # every unusual-activity score — fall back to the most recent
            # cached session instead.
            from datetime import timedelta as _opt_td
            opt_asof = end_date
            if not flat.options_cache_path(opt_asof).exists():
                try:
                    # One probe in case today's file was just published
                    flat.get_options_chain_from_flatfile("SPY", opt_asof)
                except Exception:
                    pass
            if not flat.options_cache_path(opt_asof).exists():
                for _back in range(1, 8):
                    _prev = end_date - _opt_td(days=_back)
                    if flat.options_cache_path(_prev).exists():
                        opt_asof = _prev
                        break
            if opt_asof != end_date:
                print(f"\n  NOTE: options flat file for {end_date} not yet "
                      f"published — unusual-activity detection uses {opt_asof}")

            print(f"\nOptions enrichment — tier-aware ({len(results)} symbols) ...")
            for r in results:
                tier = r.market_cap_tier or _mct(r.symbol)
                r.market_cap_tier = tier
                try:
                    if tier == "mid_cap":
                        sigs = detect_unusual_activity(
                            r.symbol, opt_asof, flat, r.entry_close,
                            volume_ratio_threshold=5.0,
                        )
                        u_score = score_unusual_activity(sigs)
                        r.unusual_options_score = u_score
                        r.conviction_score = score_conviction(r)
                        flag = " ▲" if u_score >= 0.7 else ""
                        print(f"  {r.symbol:<8}  unusual_opts={u_score:.2f} [mid_cap]"
                              f"  conv={r.conviction_score:.2f}{flag}")
                    elif tier == "large_cap":
                        sigs = detect_unusual_activity(
                            r.symbol, opt_asof, flat, r.entry_close,
                            volume_ratio_threshold=3.0,
                        )
                        u_score = score_unusual_activity(sigs)
                        r.unusual_options_score = u_score
                        r.conviction_score = score_conviction(r)
                        flag = " ▲" if u_score >= 0.6 else ""
                        print(f"  {r.symbol:<8}  unusual_opts={u_score:.2f} [large_cap]"
                              f"  conv={r.conviction_score:.2f}{flag}")
                    else:
                        # mega_cap / small_cap: PCR + IV skew from REST
                        opt_score = mp.compute_options_score(r.symbol, r.entry_close)
                        r.options_score    = opt_score
                        r.conviction_score = score_conviction(r)
                        flag = " ▲" if opt_score >= 0.6 else ""
                        print(f"  {r.symbol:<8}  opt_score={opt_score:.2f} [{tier}]"
                              f"  conv={r.conviction_score:.2f}{flag}")
                except Exception as e:
                    print(f"  {r.symbol:<8}  options ERROR — {e}")

        else:
            print("\n--with-options: set POLYGON_API_KEY for Polygon options data.")

        results.sort(key=lambda r: r.conviction_score, reverse=True)
        print()

    # ── Persist to DuckDB ─────────────────────────────────────────────────────
    if args.save_db:
        scan_id = make_run_id(args.universe.upper() if not args.symbols else "CUSTOM",
                              args.signal)
        append_scan_results(scan_id, results)
        print(f"db  → {len(results)} scan result(s) stored (scan_id={scan_id})")

    # ── CS filter (shared by IWL and paper watchlist) ─────────────────────────
    from quantlab.universe import _is_excluded_symbol as _is_excl_sym, EXCLUDE_SYMBOLS
    _cs_set: set | None = None
    try:
        from quantlab.universe import _cs_cache_path
        import pyarrow.parquet as _pq
        from datetime import timedelta as _td2
        for _d in range(7):
            _cp = _cs_cache_path(date.today() - _td2(days=_d))
            if _cp.exists():
                _cs_set = set(_pq.read_table(str(_cp)).to_pydict().get("symbol", []))
                break
    except Exception:
        pass

    def _is_common_stock(sym: str) -> bool:
        if _cs_set is not None:
            return sym in _cs_set
        return not _is_excl_sym(sym)

    # ── Institutional watchlist — broad Stage 2 CS multi-day tracking ─────────
    # Upserts ALL Stage 2 CS symbols above the scanner's min_conviction (≥0.40).
    # This is the wide monitoring layer; the strict top-5 filter below governs
    # only the paper-trading watchlist table.
    iwl = None
    try:
        from quantlab.watchlist import InstitutionalWatchlist
        iwl = InstitutionalWatchlist()

        for r in results:
            if _is_common_stock(r.symbol):
                iwl.upsert(r.symbol, r)

        # Identify PRIORITY candidates: conviction >= 0.70 AND 2+ consecutive days
        priority_entries = [
            e for e in iwl.get_multi_day(min_days=2)
            if e["conviction_score"] >= 0.70
        ]
        if priority_entries:
            print(f"\n  ⭐ PRIORITY candidates ({len(priority_entries)}):")
            for e in priority_entries:
                opts = "✓" if e["options_signal"] else "–"
                vdu  = "✓" if e["volume_dry_up"]  else "–"
                print(
                    f"     {e['symbol']:<8}  day={e['consecutive_days']}  "
                    f"conv={e['conviction_score']:.2f}  "
                    f"stage={e['stage']}  opts={opts}  vdu={vdu}  "
                    f"tape={e['tape']}"
                )

        # Housekeeping: remove symbols absent for more than 5 trading days
        removed = iwl.remove_stale(max_days_inactive=5)
        if removed > 0:
            print(f"  institutional_watchlist: removed {removed} stale entry/entries")
    except Exception as _iwl_err:
        print(f"  WARNING: institutional watchlist update failed: {_iwl_err}")

    # ── Paper trading watchlist — strict top-5 filter ─────────────────────────
    if args.add_to_watchlist:
        from quantlab.watchlist import add_to_watchlist

        # Build IWL state for confirming-signal fields (options, vdu, consecutive_days)
        iwl_state: dict[str, dict] = {}
        if iwl is not None:
            try:
                iwl_state = {e["symbol"]: e for e in iwl.get_candidates()}
            except Exception:
                pass

        # Select top-5 using strict qualification + sector concentration cap
        top5 = select_top_candidates(results, iwl_state, _is_common_stock)
        top5_symbols = {item[0].symbol for item in top5}

        # ── Clean today's watchlist: keep only top-5, drop known non-CS ────────
        today_str = date.today().isoformat()
        try:
            import duckdb as _ddb
            from quantlab.storage import DB_PATH as _DB
            _con = _ddb.connect(str(_DB))

            if top5_symbols:
                _ph = ",".join(["?"] * len(top5_symbols))
                _con.execute(
                    f"DELETE FROM watchlist "
                    f"WHERE date_added=? AND status='watching' AND symbol NOT IN ({_ph})",
                    [today_str, *top5_symbols],
                )
            else:
                _con.execute(
                    "DELETE FROM watchlist WHERE date_added=? AND status='watching'",
                    [today_str],
                )

            for _bad in EXCLUDE_SYMBOLS:
                _con.execute(
                    "DELETE FROM watchlist WHERE symbol=? AND status='watching'", [_bad]
                )
            _con.close()
        except Exception as _cl_err:
            print(f"  watchlist cleanup warning: {_cl_err}")

        # ── Build set of symbols already on watchlist (duplicate guard) ─────
        already_watching: set[str] = set()
        try:
            import duckdb as _ddb2
            from quantlab.storage import DB_PATH as _DB2
            _con2 = _ddb2.connect(str(_DB2))
            _rows = _con2.execute(
                "SELECT symbol FROM watchlist WHERE status='watching'"
            ).fetchall()
            already_watching = {row[0] for row in _rows}
            _con2.close()
        except Exception:
            pass

        # ── Add top-5 to paper watchlist (skip if already tracking) ──────────
        added = sum(
            1 for item in top5
            if item[0].symbol not in already_watching
            and add_to_watchlist(item[0])
        )
        n_qual = len([
            r for r in results
            if r.stage == 2 and _is_common_stock(r.symbol) and r.conviction_score >= 0.70
        ])
        print(f"\nwatchlist → {n_qual} qualifying | top {len(top5)} selected | {added} added")
        for item in top5:
            r, earn, cdays = item
            iwl_e = iwl_state.get(r.symbol, {})
            print(
                f"  ★ {r.symbol:<8}  conv={r.conviction_score:.2f}  "
                f"day={cdays}  earn={earn:.2f}  "
                f"opts={'✓' if iwl_e.get('options_signal') else '–'}  "
                f"vdu={'✓' if iwl_e.get('volume_dry_up') else '–'}"
            )


if __name__ == "__main__":
    main()
