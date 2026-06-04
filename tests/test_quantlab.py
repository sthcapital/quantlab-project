"""
tests/test_quantlab.py — Full test suite for all 7 layers.

Run with:  pytest -q
All tests use mock/stub data — no IBKR connection required.
"""

from datetime import date
from pathlib import Path
import sys, os

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quantlab.providers.base import Bar
from quantlab.providers.providers import MockMarketDataProvider
from quantlab.providers import create_market_data_provider
from quantlab.signals import sma_signal, breakout_signal, atr_stop_price, relative_volume, sma, regime_is_bullish
from quantlab.news import clean_headline, classify_headline, compute_news_features, NewsItem
from quantlab.research import forward_returns, compute_metrics, TradeRecord, MIN_TRADES
from quantlab.risk import apply_transaction_cost, apply_costs_to_trades, fmt_pct
from quantlab.execution import score_conviction, ScanResult, load_universe, scan_symbol
from quantlab.utils import parse_date, make_run_id, n_days_ago
from quantlab.backtest import (
    run_backtest, BacktestOutput,
    sensitivity_sweep, DEFAULT_LOOKBACKS,
    walk_forward, WalkForwardWindow,
    print_sensitivity_table, print_walk_forward_summary,
    run_universe_backtest, UniverseBacktestResult,
    print_universe_ranking,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_bars(n: int = 50, start_price: float = 100.0, trend: float = 0.003) -> list[Bar]:
    """Generate n synthetic bars with a slight upward trend."""
    bars = []
    price = start_price
    for i in range(n):
        d = date(2026, 1, 1).__class__.fromordinal(date(2026, 1, 2).toordinal() + i)
        price = price * (1 + trend + (i % 3 - 1) * 0.005)
        bars.append(Bar(
            as_of=d,
            open=price * 0.999,
            high=price * 1.008,
            low=price * 0.992,
            close=price,
            volume=1_000_000.0 + i * 10_000,
        ))
    return bars


def make_flat_bars(n: int = 50, price: float = 100.0) -> list[Bar]:
    bars = []
    for i in range(n):
        d = date(2026, 1, 1).__class__.fromordinal(date(2026, 1, 2).toordinal() + i)
        bars.append(Bar(
            as_of=d,
            open=price, high=price * 1.002,
            low=price * 0.998, close=price,
            volume=500_000.0,
        ))
    return bars


# ══════════════════════════════════════════════════════════════════════════════
# Layer 1: Providers
# ══════════════════════════════════════════════════════════════════════════════

class TestProviders:

    def test_mock_provider_returns_bars(self):
        p = MockMarketDataProvider(seed=42)
        bars = p.get_daily_bars("AAPL", date(2026, 1, 1), date(2026, 3, 31))
        assert len(bars) > 40, "Expected at least 40 trading days in Q1"

    def test_mock_provider_bar_structure(self):
        p = MockMarketDataProvider()
        bars = p.get_daily_bars("MSFT", date(2026, 1, 2), date(2026, 1, 31))
        for b in bars:
            assert b.high >= b.low
            assert b.high >= b.close >= b.low or b.close >= b.high  # close can be at high
            assert b.volume > 0

    def test_mock_provider_is_deterministic(self):
        p1 = MockMarketDataProvider(seed=7)
        p2 = MockMarketDataProvider(seed=7)
        b1 = p1.get_daily_bars("TSLA", date(2026, 1, 2), date(2026, 2, 28))
        b2 = p2.get_daily_bars("TSLA", date(2026, 1, 2), date(2026, 2, 28))
        assert [b.close for b in b1] == [b.close for b in b2]

    def test_factory_creates_mock(self):
        p = create_market_data_provider("mock", seed=1)
        assert isinstance(p, MockMarketDataProvider)

    def test_factory_creates_ibkr(self):
        from quantlab.providers.ibkr import IbkrProvider
        p = create_market_data_provider("ibkr", host="127.0.0.1", port=7497, client_id=1)
        assert isinstance(p, IbkrProvider)

    def test_factory_rejects_unknown(self):
        try:
            create_market_data_provider("unknown_provider")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Unknown" in str(e)

    def test_bar_pct_change(self):
        b1 = Bar(date(2026, 1, 2), 100, 102, 99, 100, 1e6)
        b2 = Bar(date(2026, 1, 3), 100, 105, 100, 110, 1e6)
        assert abs(b2.pct_change(b1) - 0.10) < 1e-9

    def test_bar_true_range(self):
        prev = Bar(date(2026, 1, 2), 100, 102, 99, 100, 1e6)
        curr = Bar(date(2026, 1, 3), 98, 103, 97, 101, 1e6)
        tr = curr.true_range(prev)
        assert tr == max(103 - 97, abs(103 - 100), abs(97 - 100))


# ══════════════════════════════════════════════════════════════════════════════
# Layer 2: News
# ══════════════════════════════════════════════════════════════════════════════

class TestNews:

    def test_clean_headline_strips_metadata(self):
        raw = "{A:800015:L:en:K:0.97:C:0.97}!Apple upgraded by Bernstein"
        assert clean_headline(raw) == "Apple upgraded by Bernstein"

    def test_clean_headline_html_entities(self):
        raw = "{A:123}Monness Crespi &amp; Hardt reiterated Buy"
        assert "&amp;" not in clean_headline(raw)
        assert "&" in clean_headline(raw)

    def test_classify_upgrade(self):
        assert classify_headline("Goldman upgraded Apple to Buy") == "upgrade"

    def test_classify_downgrade(self):
        assert classify_headline("Barclays downgraded AAPL to Underweight") == "downgrade"

    def test_classify_earnings(self):
        assert classify_headline("Apple Q3 earnings beat guidance") == "earnings"

    def test_classify_management(self):
        assert classify_headline("Apple CEO Tim Cook steps down") == "management"

    def test_classify_analyst_action(self):
        assert classify_headline("UBS reiterated Neutral with target $287") == "analyst_action"

    def test_classify_other(self):
        assert classify_headline("Supply chain news update") == "other"

    def test_news_features_empty(self):
        feat = compute_news_features([], "2026-06-03", lookback_days=7)
        assert feat.total_count == 0
        assert feat.dominant_category == "none"
        assert not feat.has_news()

    def test_news_features_counts(self):
        from datetime import datetime
        items = [
            NewsItem(datetime(2026, 6, 1), "2026-06-01", "BRFG", "id1", "upgrade", "Upgrade headline", 0.9, 0.8),
            NewsItem(datetime(2026, 6, 2), "2026-06-02", "BRFUPDN", "id2", "earnings", "Earnings beat", 0.7, 0.6),
            NewsItem(datetime(2026, 5, 20), "2026-05-20", "DJNL", "id3", "upgrade", "Old upgrade", 0.5, 0.5),
        ]
        feat = compute_news_features(items, "2026-06-03", lookback_days=7)
        assert feat.total_count == 2  # only last 7 days
        assert feat.upgrade_count == 1
        assert feat.earnings_count == 1
        assert feat.dominant_category in {"upgrade", "earnings"}
        assert feat.has_news()


# ══════════════════════════════════════════════════════════════════════════════
# Layer 4: Signals
# ══════════════════════════════════════════════════════════════════════════════

class TestSignals:

    def test_sma_signal_not_enough_bars(self):
        bars = make_bars(5)
        result = sma_signal(bars, "AAPL", lookback=20)
        assert result is None

    def test_sma_signal_fires_on_uptrend(self):
        bars = make_bars(60, trend=0.005)  # strong uptrend
        result = sma_signal(bars, "AAPL", lookback=20)
        assert result is not None
        assert result.signal is True
        assert result.signal_type == "sma"

    def test_sma_signal_no_fire_on_downtrend(self):
        bars = make_bars(60, trend=-0.005)  # downtrend
        result = sma_signal(bars, "AAPL", lookback=20)
        assert result is not None
        assert result.signal is False

    def test_breakout_signal_fires_on_new_high(self):
        bars = make_bars(60, trend=0.005)  # rising — should break out
        result = breakout_signal(bars, "AAPL", lookback=20)
        assert result is not None
        assert result.signal_type == "breakout"

    def test_breakout_signal_not_enough_bars(self):
        bars = make_bars(5)
        result = breakout_signal(bars, "AAPL", lookback=20)
        assert result is None

    def test_atr_stop_below_entry(self):
        bars = make_bars(40)
        entry = bars[-1].close
        stop = atr_stop_price(bars, entry, atr_period=14, atr_multiplier=2.0)
        assert stop is not None
        assert stop < entry

    def test_relative_volume_not_enough_bars(self):
        bars = make_bars(5)
        rv = relative_volume(bars, period=20)
        assert rv is None

    def test_regime_bullish_on_uptrend(self):
        bars = make_bars(250, trend=0.002)
        assert regime_is_bullish(bars, sma_period=200) is True

    def test_regime_bearish_on_downtrend(self):
        bars = make_bars(250, trend=-0.002)
        assert regime_is_bullish(bars, sma_period=200) is False

    def test_sma_helper(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert sma(values, 3) == 4.0
        assert sma(values, 10) is None


# ══════════════════════════════════════════════════════════════════════════════
# Layer 4: Research / backtesting
# ══════════════════════════════════════════════════════════════════════════════

class TestResearch:

    def test_forward_returns_correct(self):
        bars = make_bars(20)
        result = forward_returns(bars, entry_index=10, entry_price=bars[10].close)
        assert "ret_1d" in result
        assert "ret_3d" in result
        assert "ret_5d" in result
        assert "mfe_5d" in result
        assert "mae_5d" in result

    def test_forward_returns_na_at_boundary(self):
        bars = make_bars(15)
        result = forward_returns(bars, entry_index=14, entry_price=bars[14].close)
        assert result["ret_1d"] is None
        assert result["ret_5d"] is None

    def test_mfe_greater_than_mae(self):
        bars = make_bars(30, trend=0.003)
        result = forward_returns(bars, entry_index=20, entry_price=bars[20].close)
        if result["mfe_5d"] is not None and result["mae_5d"] is not None:
            assert result["mfe_5d"] >= result["mae_5d"]

    def test_min_trades_constant(self):
        assert MIN_TRADES == 30


# ══════════════════════════════════════════════════════════════════════════════
# Layer 6: Risk
# ══════════════════════════════════════════════════════════════════════════════

class TestRisk:

    def test_transaction_cost_reduces_return(self):
        raw = 0.05
        net = apply_transaction_cost(raw, cost_bps=10.0)
        assert net < raw
        assert abs(net - (raw - 0.001)) < 1e-10

    def test_transaction_cost_zero_bps(self):
        raw = 0.03
        net = apply_transaction_cost(raw, cost_bps=0.0)
        assert net == raw

    def test_fmt_pct_none(self):
        assert fmt_pct(None) == "NA"

    def test_fmt_pct_value(self):
        assert fmt_pct(0.0519) == "5.19%"

    def test_costs_applied_to_trades(self):
        t = TradeRecord(
            symbol="AAPL", signal_date="2026-01-01", entry_date="2026-01-01",
            entry_price=100.0, exit_date="2026-01-10", exit_price=105.0,
            trade_return=0.05, ret_1d=0.01, ret_3d=0.03, ret_5d=0.05,
            mfe_5d=0.06, mae_5d=-0.01, atr_stop=97.0,
        )
        apply_costs_to_trades([t], cost_bps=10.0)
        assert t.trade_return < 0.05
        assert t.cost_bps == 10.0


# ══════════════════════════════════════════════════════════════════════════════
# Layer 7: Execution / scanner
# ══════════════════════════════════════════════════════════════════════════════

class TestExecution:

    def test_load_universe_small(self):
        u = load_universe("small")
        assert len(u) == 7
        assert "AAPL" in u

    def test_load_universe_custom(self):
        u = load_universe("AAPL,TSLA,NVDA")
        assert u == ["AAPL", "TSLA", "NVDA"]

    def test_scan_symbol_returns_result(self):
        bars = make_bars(60, trend=0.005)
        result = scan_symbol("AAPL", bars, signal_type="breakout", lookback=20)
        assert result is not None
        assert result.symbol == "AAPL"
        assert isinstance(result.conviction_score, float)
        assert 0.0 <= result.conviction_score <= 1.0

    def test_scan_symbol_not_enough_bars(self):
        bars = make_bars(5)
        result = scan_symbol("AAPL", bars, signal_type="breakout", lookback=20)
        assert result is None

    def test_conviction_zero_without_signal(self):
        r = ScanResult(
            symbol="AAPL", scan_date="2026-06-03",
            signal_type="breakout", signal=False,
            entry_close=300.0, indicator_value=305.0, lookback=20,
        )
        assert score_conviction(r) == 0.0

    def test_conviction_increases_with_layers(self):
        # Signal only
        r1 = ScanResult(
            symbol="AAPL", scan_date="2026-06-03",
            signal_type="breakout", signal=True,
            entry_close=310.0, indicator_value=309.0, lookback=20,
            regime_bullish=False, news_count=0,
        )
        s1 = score_conviction(r1)

        # Signal + regime
        r2 = ScanResult(
            symbol="AAPL", scan_date="2026-06-03",
            signal_type="breakout", signal=True,
            entry_close=310.0, indicator_value=309.0, lookback=20,
            regime_bullish=True, news_count=0,
        )
        s2 = score_conviction(r2)

        # Signal + regime + earnings news
        r3 = ScanResult(
            symbol="AAPL", scan_date="2026-06-03",
            signal_type="breakout", signal=True,
            entry_close=310.0, indicator_value=309.0, lookback=20,
            regime_bullish=True, news_count=2, news_category="earnings",
            news_c_score=0.85, rel_volume=1.8,
        )
        s3 = score_conviction(r3)

        assert s1 < s2 < s3
        assert s3 <= 1.0

    def test_load_universe_sp500_sample(self):
        u = load_universe("sp500_sample")
        assert len(u) == 50
        assert "AAPL" in u
        assert "GS" in u

    def test_is_actionable(self):
        r = ScanResult(
            symbol="AAPL", scan_date="2026-06-03",
            signal_type="breakout", signal=True,
            entry_close=310.0, indicator_value=309.0, lookback=20,
            regime_bullish=True, news_count=2, news_category="earnings",
            conviction_score=0.75,
        )
        assert r.is_actionable(min_conviction=0.5)
        assert not r.is_actionable(min_conviction=0.9)


# ══════════════════════════════════════════════════════════════════════════════
# Utils
# ══════════════════════════════════════════════════════════════════════════════

class TestUtils:

    def test_parse_date_valid(self):
        d = parse_date("2026-06-03")
        assert d == date(2026, 6, 3)

    def test_parse_date_invalid(self):
        try:
            parse_date("06/03/2026")
            assert False
        except ValueError:
            pass

    def test_n_days_ago(self):
        d = n_days_ago(365)
        assert (date.today() - d).days == 365

    def test_make_run_id(self):
        run_id = make_run_id("AAPL", "breakout", "20260603_120000")
        assert run_id == "AAPL_breakout_20260603_120000"


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3 Item 1: Backtest engine with transaction costs
# ══════════════════════════════════════════════════════════════════════════════

class TestBacktest:

    def test_run_backtest_returns_output(self):
        bars = make_bars(100, trend=0.002)
        out = run_backtest(bars, "AAPL", signal_type="breakout", lookback=20)
        assert isinstance(out, BacktestOutput)
        assert len(out.equity_curve) == len(bars)
        assert out.metrics is not None

    def test_run_backtest_equity_starts_at_initial_capital(self):
        bars = make_bars(80)
        out = run_backtest(bars, "TSLA", lookback=20, initial_capital=50_000.0)
        assert out.equity_curve[0] == 50_000.0

    def test_run_backtest_cost_reduces_trade_return(self):
        bars = make_bars(200, trend=0.003)
        out_free = run_backtest(bars, "AAPL", lookback=20, cost_bps=0.0)
        out_cost = run_backtest(bars, "AAPL", lookback=20, cost_bps=10.0)
        free_returns = [t.trade_return for t in out_free.trades if t.trade_return is not None]
        cost_returns = [t.trade_return for t in out_cost.trades if t.trade_return is not None]
        if free_returns and cost_returns:
            assert sum(cost_returns) < sum(free_returns)

    def test_run_backtest_sma_signal_type(self):
        bars = make_bars(100, trend=0.002)
        out = run_backtest(bars, "MSFT", signal_type="sma", lookback=20)
        assert out.signal_type == "sma"
        assert out.cost_bps == 10.0  # default

    def test_run_backtest_insufficient_sample_flagged(self):
        bars = make_bars(40)
        out = run_backtest(bars, "AAPL", lookback=20)
        assert not out.metrics.sufficient_sample

    def test_run_backtest_cost_stored_on_trade(self):
        bars = make_bars(200, trend=0.003)
        out = run_backtest(bars, "AAPL", lookback=20, cost_bps=10.0)
        for t in out.trades:
            assert t.cost_bps == 10.0

    def test_run_backtest_unknown_signal_raises(self):
        bars = make_bars(50)
        try:
            run_backtest(bars, "AAPL", signal_type="unknown")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Unknown signal_type" in str(e)


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3 Item 4: Parameter sensitivity sweep
# ══════════════════════════════════════════════════════════════════════════════

class TestSensitivity:

    def test_sweep_returns_all_valid_lookbacks(self):
        bars = make_bars(200, trend=0.002)
        results = sensitivity_sweep(bars, "AAPL", lookbacks=[5, 10, 20, 50])
        assert set(results.keys()) == {5, 10, 20, 50}

    def test_sweep_skips_lookback_exceeding_bar_count(self):
        bars = make_bars(30)
        results = sensitivity_sweep(bars, "AAPL", lookbacks=[5, 10, 20, 50])
        assert 50 not in results  # 50 >= 30 bars
        assert 5 in results

    def test_sweep_uses_default_lookbacks(self):
        bars = make_bars(200, trend=0.002)
        results = sensitivity_sweep(bars, "AAPL")
        for lb in DEFAULT_LOOKBACKS:
            if lb < len(bars):
                assert lb in results

    def test_sweep_metrics_are_performance_metrics(self):
        from quantlab.research import PerformanceMetrics
        bars = make_bars(100, trend=0.002)
        results = sensitivity_sweep(bars, "AAPL", lookbacks=[10, 20])
        for m in results.values():
            assert isinstance(m, PerformanceMetrics)

    def test_sweep_print_runs_without_error(self, capsys):
        bars = make_bars(100, trend=0.002)
        results = sensitivity_sweep(bars, "AAPL", lookbacks=[10, 20])
        print_sensitivity_table(results)
        captured = capsys.readouterr()
        assert "Sensitivity" in captured.out


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3 Item 5: Walk-forward validation
# ══════════════════════════════════════════════════════════════════════════════

class TestWalkForward:

    def test_walk_forward_produces_multiple_windows(self):
        bars = make_bars(400, trend=0.001)
        windows = walk_forward(bars, "AAPL", lookback=20, is_bars=150, oos_bars=50)
        assert len(windows) >= 2

    def test_walk_forward_oos_follows_is(self):
        bars = make_bars(400, trend=0.001)
        windows = walk_forward(bars, "AAPL", lookback=20, is_bars=150, oos_bars=50)
        for w in windows:
            assert w.oos_start_bar == w.is_end_bar

    def test_walk_forward_windows_step_by_oos_size(self):
        bars = make_bars(400, trend=0.001)
        windows = walk_forward(bars, "AAPL", lookback=20, is_bars=150, oos_bars=50)
        for i in range(1, len(windows)):
            assert windows[i].is_start_bar == windows[i - 1].is_start_bar + 50

    def test_walk_forward_too_few_bars_returns_empty(self):
        bars = make_bars(50)
        windows = walk_forward(bars, "AAPL", lookback=20, is_bars=200, oos_bars=60)
        assert len(windows) == 0

    def test_walk_forward_in_sample_metrics_present(self):
        from quantlab.research import PerformanceMetrics
        bars = make_bars(400, trend=0.001)
        windows = walk_forward(bars, "AAPL", lookback=20, is_bars=150, oos_bars=50)
        assert len(windows) > 0
        for w in windows:
            assert isinstance(w.in_sample, PerformanceMetrics)
            assert isinstance(w, WalkForwardWindow)

    def test_walk_forward_print_runs_without_error(self, capsys):
        bars = make_bars(400, trend=0.001)
        windows = walk_forward(bars, "AAPL", lookback=20, is_bars=150, oos_bars=50)
        print_walk_forward_summary(windows)
        captured = capsys.readouterr()
        assert "Walk-Forward" in captured.out


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3 Item 6: CSV trade log + equity curve chart
# ══════════════════════════════════════════════════════════════════════════════

class TestStoragePhase3:

    def test_equity_chart_creates_png(self, tmp_path, monkeypatch):
        import quantlab.storage as _storage
        monkeypatch.setattr(_storage, "OUTPUT_DIR", tmp_path)
        from quantlab.storage import save_equity_curve_chart
        bars = make_bars(60)
        equity = [10_000.0 * (1 + i * 0.001) for i in range(60)]
        path = save_equity_curve_chart(equity, bars, "AAPL", "breakout", run_tag="testrun")
        assert path.exists()
        assert path.suffix == ".png"
        assert "AAPL_breakout" in path.name

    def test_equity_chart_no_tag(self, tmp_path, monkeypatch):
        import quantlab.storage as _storage
        monkeypatch.setattr(_storage, "OUTPUT_DIR", tmp_path)
        from quantlab.storage import save_equity_curve_chart
        bars = make_bars(40)
        equity = [10_000.0] * 40
        path = save_equity_curve_chart(equity, bars, "MSFT", "sma")
        assert path.exists()

    def test_export_trades_csv_creates_file(self, tmp_path, monkeypatch):
        import quantlab.storage as _storage
        monkeypatch.setattr(_storage, "OUTPUT_DIR", tmp_path)
        from quantlab.storage import export_trades_csv
        bars = make_bars(100, trend=0.003)
        out = run_backtest(bars, "AAPL", lookback=20, cost_bps=10.0)
        path = export_trades_csv("AAPL", "breakout", out.trades, run_tag="test")
        assert path.exists()
        assert path.suffix == ".csv"


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3 Item 7: DuckDB backtest run persistence
# ══════════════════════════════════════════════════════════════════════════════

class TestDuckDBStorage:

    def test_append_backtest_run_stores_row(self, tmp_path, monkeypatch):
        import quantlab.storage as _storage
        monkeypatch.setattr(_storage, "DB_PATH", tmp_path / "test.duckdb")
        from quantlab.storage import append_backtest_run
        bars = make_bars(100, trend=0.002)
        out = run_backtest(bars, "AAPL", lookback=20)
        append_backtest_run(
            "run_001", "AAPL", "breakout", 20,
            bars[0].as_of, bars[-1].as_of, out.metrics,
        )
        import duckdb
        con = duckdb.connect(str(tmp_path / "test.duckdb"))
        rows = con.execute("SELECT run_id, trade_count FROM backtest_runs").fetchall()
        con.close()
        assert len(rows) == 1
        assert rows[0][0] == "run_001"
        assert rows[0][1] == out.metrics.trade_count

    def test_append_multiple_runs(self, tmp_path, monkeypatch):
        import quantlab.storage as _storage
        monkeypatch.setattr(_storage, "DB_PATH", tmp_path / "test2.duckdb")
        from quantlab.storage import append_backtest_run
        bars = make_bars(100, trend=0.002)
        for i, lb in enumerate([10, 20]):
            out = run_backtest(bars, "AAPL", lookback=lb)
            append_backtest_run(
                f"run_{i:03d}", "AAPL", "breakout", lb,
                bars[0].as_of, bars[-1].as_of, out.metrics,
            )
        import duckdb
        con = duckdb.connect(str(tmp_path / "test2.duckdb"))
        count = con.execute("SELECT COUNT(*) FROM backtest_runs").fetchone()[0]
        con.close()
        assert count == 2

    def test_append_trades_to_db_works(self, tmp_path, monkeypatch):
        import quantlab.storage as _storage
        monkeypatch.setattr(_storage, "DB_PATH", tmp_path / "trades.duckdb")
        from quantlab.storage import append_trades_to_db
        bars = make_bars(200, trend=0.003)
        out = run_backtest(bars, "AAPL", lookback=20)
        completed = [t for t in out.trades if t.trade_return is not None]
        if completed:
            append_trades_to_db("run_abc", "breakout", 20, completed)
            import duckdb
            con = duckdb.connect(str(tmp_path / "trades.duckdb"))
            count = con.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
            con.close()
            assert count == len(completed)


# ══════════════════════════════════════════════════════════════════════════════
# Phase 4: Universe backtest + walk-forward storage
# ══════════════════════════════════════════════════════════════════════════════

class TestUniverseBacktest:

    def test_run_universe_returns_result_per_symbol(self):
        provider = MockMarketDataProvider(seed=42)
        symbols = load_universe("small")  # 7 symbols — fast
        results = run_universe_backtest(
            provider, symbols,
            date(2024, 1, 2), date(2025, 12, 31),
            lookback=5, is_bars=252, oos_bars=63, verbose=False,
        )
        assert len(results) == len(symbols)
        assert all(isinstance(r, UniverseBacktestResult) for r in results)

    def test_results_sorted_by_oos_sharpe_desc(self):
        provider = MockMarketDataProvider(seed=42)
        symbols = load_universe("small")
        results = run_universe_backtest(
            provider, symbols,
            date(2024, 1, 2), date(2025, 12, 31),
            lookback=5, is_bars=252, oos_bars=63, verbose=False,
        )
        valid = [r for r in results if r.avg_oos_sharpe is not None]
        for i in range(1, len(valid)):
            assert valid[i - 1].avg_oos_sharpe >= valid[i].avg_oos_sharpe

    def test_each_result_has_windows_and_baseline(self):
        provider = MockMarketDataProvider(seed=42)
        results = run_universe_backtest(
            provider, ["AAPL", "MSFT"],
            date(2024, 1, 2), date(2025, 12, 31),
            lookback=5, is_bars=252, oos_bars=63, verbose=False,
        )
        for r in results:
            assert len(r.windows) > 0
            assert isinstance(r.baseline, BacktestOutput)
            assert r.bar_count > 0

    def test_print_universe_ranking_runs(self, capsys):
        provider = MockMarketDataProvider(seed=42)
        results = run_universe_backtest(
            provider, load_universe("small"),
            date(2024, 1, 2), date(2025, 12, 31),
            lookback=5, is_bars=252, oos_bars=63, verbose=False,
        )
        print_universe_ranking(results, top_n=5)
        captured = capsys.readouterr()
        assert "Ranking" in captured.out


class TestWalkForwardStorage:

    def test_append_walk_forward_windows_row_count(self, tmp_path, monkeypatch):
        import quantlab.storage as _storage
        monkeypatch.setattr(_storage, "DB_PATH", tmp_path / "wfw.duckdb")
        from quantlab.storage import append_walk_forward_windows
        bars = make_bars(400, trend=0.001)
        windows = walk_forward(bars, "AAPL", lookback=5, is_bars=150, oos_bars=50)
        append_walk_forward_windows("run_x", "AAPL", "breakout", 5, windows)
        import duckdb
        con = duckdb.connect(str(tmp_path / "wfw.duckdb"))
        count = con.execute("SELECT COUNT(*) FROM walk_forward_windows").fetchone()[0]
        con.close()
        assert count == len(windows)

    def test_query_oos_ranking_returns_sorted_rows(self, tmp_path, monkeypatch):
        import quantlab.storage as _storage
        monkeypatch.setattr(_storage, "DB_PATH", tmp_path / "rank.duckdb")
        from quantlab.storage import append_walk_forward_windows, query_oos_ranking
        # sma_signal fires on make_bars uptrend data; breakout requires higher spread
        for sym in ["AAPL", "MSFT"]:
            bars = make_bars(400, trend=0.002)
            windows = walk_forward(bars, sym, signal_type="sma", lookback=5, is_bars=150, oos_bars=50)
            append_walk_forward_windows("run_y", sym, "sma", 5, windows)
        ranking = query_oos_ranking("run_y", top_n=10)
        assert len(ranking) >= 1
        assert "symbol" in ranking[0]
        assert "avg_oos_sharpe" in ranking[0]
        for i in range(1, len(ranking)):
            assert ranking[i - 1]["avg_oos_sharpe"] >= ranking[i]["avg_oos_sharpe"]

    def test_query_oos_ranking_filters_by_run_id(self, tmp_path, monkeypatch):
        import quantlab.storage as _storage
        monkeypatch.setattr(_storage, "DB_PATH", tmp_path / "filter.duckdb")
        from quantlab.storage import append_walk_forward_windows, query_oos_ranking
        bars = make_bars(400, trend=0.002)
        windows = walk_forward(bars, "AAPL", signal_type="sma", lookback=5, is_bars=150, oos_bars=50)
        append_walk_forward_windows("run_alpha", "AAPL", "sma", 5, windows)
        append_walk_forward_windows("run_beta", "AAPL", "sma", 5, windows)
        alpha = query_oos_ranking("run_alpha")
        beta = query_oos_ranking("run_beta")
        assert len(alpha) == 1
        assert len(beta) == 1


# ══════════════════════════════════════════════════════════════════════════════
# IBKR provider fixes (items 1–3, 6–9)
# All tests are offline — no TWS connection required.
# ══════════════════════════════════════════════════════════════════════════════

class TestIbkrProviderFixes:
    """Tests for the 9 blocking/quality fixes to IbkrProvider."""

    # ── Item 9: ping_tws ──────────────────────────────────────────────────────

    def test_ping_tws_closed_port_returns_false(self):
        from quantlab.providers.ibkr import ping_tws
        # Port 1 is almost never open; timeout=0.5s so the test is fast
        assert ping_tws("127.0.0.1", port=1, timeout=0.5) is False

    def test_ping_tws_bad_host_returns_false(self):
        from quantlab.providers.ibkr import ping_tws
        assert ping_tws("192.0.2.1", port=7497, timeout=0.5) is False  # TEST-NET, RFC 5737

    # ── Items 1 & 6: _duration_str and _end_date_time ────────────────────────

    def test_duration_str_sub_year(self):
        from quantlab.providers.ibkr import IbkrProvider
        start = date(2025, 1, 1)
        end   = date(2025, 6, 30)
        ds = IbkrProvider._duration_str(start, end)
        assert ds.endswith(" D")
        days = int(ds.split()[0])
        assert days >= (end - start).days   # must cover the range

    def test_duration_str_two_years(self):
        from quantlab.providers.ibkr import IbkrProvider
        start = date(2023, 1, 1)
        end   = date(2025, 1, 1)
        ds = IbkrProvider._duration_str(start, end)
        assert ds.endswith(" Y")
        assert int(ds.split()[0]) >= 2

    def test_duration_str_near_year_uses_years(self):
        from quantlab.providers.ibkr import IbkrProvider
        start = date(2025, 1, 1)
        end   = date(2025, 12, 31)
        # 364 days + 7 buffer = 371 > 365, so code correctly returns "Y" format
        ds = IbkrProvider._duration_str(start, end)
        assert ds.endswith(" Y")
        assert int(ds.split()[0]) >= 1

    def test_end_date_time_today_returns_empty(self):
        from quantlab.providers.ibkr import IbkrProvider
        assert IbkrProvider._end_date_time(date.today()) == ""

    def test_end_date_time_future_returns_empty(self):
        from quantlab.providers.ibkr import IbkrProvider
        future = date(2099, 12, 31)
        assert IbkrProvider._end_date_time(future) == ""

    def test_end_date_time_historical_returns_timestamp(self):
        from quantlab.providers.ibkr import IbkrProvider
        past = date(2024, 6, 15)
        ts = IbkrProvider._end_date_time(past)
        assert ts == "20240615 23:59:59"

    # ── Item 3: persistent connection API ─────────────────────────────────────

    def test_ibkr_provider_has_connect_disconnect(self):
        from quantlab.providers.ibkr import IbkrProvider
        p = IbkrProvider()
        assert hasattr(p, "connect")
        assert hasattr(p, "disconnect")
        assert hasattr(p, "__enter__")
        assert hasattr(p, "__exit__")

    def test_ibkr_provider_no_connection_on_init(self):
        from quantlab.providers.ibkr import IbkrProvider
        p = IbkrProvider()
        assert p._ib is None

    # ── Item 8: explicit spot_client_id ──────────────────────────────────────

    def test_spot_client_id_default_offset(self):
        from quantlab.providers.ibkr import IbkrProvider
        p = IbkrProvider(client_id=1)
        assert p.spot_client_id == 51   # default: client_id + 50

    def test_spot_client_id_explicit_override(self):
        from quantlab.providers.ibkr import IbkrProvider
        p = IbkrProvider(client_id=1, spot_client_id=99)
        assert p.spot_client_id == 99

    def test_spot_client_id_differs_from_client_id(self):
        from quantlab.providers.ibkr import IbkrProvider
        p = IbkrProvider(client_id=5)
        assert p.spot_client_id != p.client_id

    # ── Item 4: factory.py fixed ──────────────────────────────────────────────

    def test_factory_py_imports_correctly(self):
        # factory.py previously imported the wrong class name; verify it loads
        from quantlab.providers.factory import create_market_data_provider
        from quantlab.providers.ibkr import IbkrProvider
        p = create_market_data_provider("ibkr")
        assert isinstance(p, IbkrProvider)

    # ── Item 8: config has explicit client IDs ────────────────────────────────

    def test_config_has_spot_and_news_client_ids(self):
        from quantlab.utils import get_config
        cfg = get_config("ibkr")
        assert "spot_client_id" in cfg
        assert "news_client_id" in cfg
        assert cfg["spot_client_id"] != cfg["client_id"]
        assert cfg["news_client_id"] != cfg["client_id"]
        assert cfg["spot_client_id"] != cfg["news_client_id"]

    # ── Item 7: cache path helper ─────────────────────────────────────────────

    def test_cache_path_returns_parquet(self):
        from quantlab.providers.ibkr import IbkrProvider
        p = IbkrProvider()
        path = p._cache_path("AAPL")
        assert path.name == "AAPL_bars.parquet"

    def test_cache_miss_on_absent_file(self):
        from quantlab.providers.ibkr import IbkrProvider
        p = IbkrProvider()
        result = p._load_from_cache("____NOSUCHSYMBOL____", date(2024, 1, 1), date(2024, 12, 31))
        assert result is None

    def test_cache_roundtrip(self, tmp_path, monkeypatch):
        import quantlab.storage as _storage
        monkeypatch.setattr(_storage, "DATA_PROCESSED", tmp_path)
        monkeypatch.setattr(_storage, "OUTPUT_DIR", tmp_path)
        from quantlab.providers.ibkr import IbkrProvider
        bars = make_bars(60)
        p = IbkrProvider()
        p._save_to_cache("TEST", bars)
        loaded = p._load_from_cache(
            "TEST", bars[0].as_of, bars[-1].as_of
        )
        assert loaded is not None
        assert len(loaded) == len(bars)
        assert loaded[0].as_of == bars[0].as_of

    def test_cache_miss_on_partial_coverage(self, tmp_path, monkeypatch):
        import quantlab.storage as _storage
        monkeypatch.setattr(_storage, "DATA_PROCESSED", tmp_path)
        monkeypatch.setattr(_storage, "OUTPUT_DIR", tmp_path)
        from quantlab.providers.ibkr import IbkrProvider
        bars = make_bars(30)   # small window
        p = IbkrProvider()
        p._save_to_cache("PARTIAL", bars)
        # Request extends beyond what was cached — should be a miss
        beyond = bars[-1].as_of.__class__.fromordinal(bars[-1].as_of.toordinal() + 10)
        result = p._load_from_cache("PARTIAL", bars[0].as_of, beyond)
        assert result is None


class TestIbkrScript:
    """Smoke-tests for the new run_universe_backtest.py script (mock provider)."""

    def test_run_universe_backtest_script_importable(self):
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "run_universe_backtest",
            Path(__file__).parent.parent / "scripts" / "run_universe_backtest.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "main")
