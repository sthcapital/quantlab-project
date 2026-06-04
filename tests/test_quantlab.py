"""
tests/test_quantlab.py — Full test suite for all 7 layers.

Run with:  pytest -q
All tests use mock/stub data — no IBKR connection required.
"""

import pytest
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
        assert "BRK B" in u       # IBKR-format ticker (space, not dot)
        assert "BRK.B" not in u   # dot form fails contract qualification

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


# ══════════════════════════════════════════════════════════════════════════════
# tag_trades_with_news (shared pure function)
# ══════════════════════════════════════════════════════════════════════════════

class TestTagTradesWithNews:
    """tag_trades_with_news is pure — no IBKR connection needed."""

    def _make_trade(self, signal_date: str) -> TradeRecord:
        return TradeRecord(
            symbol="AAPL", signal_date=signal_date,
            entry_date=signal_date, entry_price=150.0,
            exit_date=None, exit_price=None, trade_return=0.02,
            ret_1d=None, ret_3d=None, ret_5d=None,
            mfe_5d=None, mae_5d=None, atr_stop=None,
        )

    def _make_news(self, date_str: str, category: str = "upgrade") -> "NewsItem":
        from datetime import datetime
        from quantlab.news import NewsItem
        return NewsItem(
            time=datetime.strptime(date_str, "%Y-%m-%d"),
            date=date_str,
            provider="BRFG",
            article_id="id1",
            category=category,
            headline="Test headline",
            k_score=0.8,
            c_score=0.9,
        )

    def test_tags_trade_with_matching_news(self):
        from quantlab.news import tag_trades_with_news
        trade = self._make_trade("2026-01-10")
        news = [self._make_news("2026-01-08", "upgrade")]
        tagged = tag_trades_with_news([trade], news, lookback_days=7)
        assert tagged == 1
        assert trade.news_count == 1
        assert trade.news_category == "upgrade"

    def test_does_not_tag_trade_with_old_news(self):
        from quantlab.news import tag_trades_with_news
        trade = self._make_trade("2026-01-10")
        news = [self._make_news("2025-12-01", "earnings")]  # 40 days before signal
        tagged = tag_trades_with_news([trade], news, lookback_days=7)
        assert tagged == 0
        assert trade.news_count == 0
        assert trade.news_category == "none"

    def test_tags_multiple_trades_independently(self):
        from quantlab.news import tag_trades_with_news
        t1 = self._make_trade("2026-01-05")
        t2 = self._make_trade("2026-01-20")
        news = [
            self._make_news("2026-01-04", "upgrade"),    # within t1 window only
            self._make_news("2026-01-18", "earnings"),   # within t2 window only
        ]
        tagged = tag_trades_with_news([t1, t2], news, lookback_days=7)
        assert tagged == 2
        assert t1.news_category == "upgrade"
        assert t2.news_category == "earnings"

    def test_returns_zero_with_empty_news(self):
        from quantlab.news import tag_trades_with_news
        trade = self._make_trade("2026-01-10")
        tagged = tag_trades_with_news([trade], [], lookback_days=7)
        assert tagged == 0

    def test_does_not_overwrite_existing_tags(self):
        """Once a trade is tagged, calling again with no matching news leaves it."""
        from quantlab.news import tag_trades_with_news
        trade = self._make_trade("2026-01-10")
        news = [self._make_news("2026-01-08", "downgrade")]
        tag_trades_with_news([trade], news)
        # second call with empty news — trade already tagged, won't be touched
        tag_trades_with_news([trade], [])
        assert trade.news_category == "downgrade"  # unchanged


# ══════════════════════════════════════════════════════════════════════════════
# Execution improvements: category-weighted news, edge score, LOW_EDGE_SYMBOLS
# ══════════════════════════════════════════════════════════════════════════════

class TestCategoryWeightedConviction:
    """Verify the updated score_conviction() uses NEWS_CATEGORY_WEIGHTS."""

    def _result(self, news_category: str = "none", news_count: int = 0,
                regime: bool = True, rel_vol: float | None = None,
                c_score: float | None = None) -> ScanResult:
        return ScanResult(
            symbol="AAPL", scan_date="2026-01-10",
            signal_type="breakout", signal=True,
            entry_close=180.0, indicator_value=179.0, lookback=5,
            regime_bullish=regime, news_count=news_count,
            news_category=news_category, news_c_score=c_score,
            rel_volume=rel_vol,
        )

    def test_no_signal_always_zero(self):
        r = self._result()
        r.signal = False
        assert score_conviction(r) == 0.0

    def test_earnings_gives_highest_news_weight(self):
        from quantlab.execution import score_conviction, NEWS_CATEGORY_WEIGHTS
        earnings = score_conviction(self._result("earnings", news_count=1))
        upgrade  = score_conviction(self._result("upgrade",  news_count=1))
        analyst  = score_conviction(self._result("analyst_action", news_count=1))
        assert earnings > upgrade > analyst

    def test_downgrade_reduces_below_no_news(self):
        from quantlab.execution import score_conviction
        no_news  = score_conviction(self._result(news_count=0))
        downgrade = score_conviction(self._result("downgrade", news_count=1))
        assert downgrade < no_news

    def test_other_category_no_lift(self):
        from quantlab.execution import score_conviction
        no_news = score_conviction(self._result(news_count=0))
        other   = score_conviction(self._result("other", news_count=1))
        assert other == no_news   # +0.00 weight, no change

    def test_management_equals_earnings_weight(self):
        from quantlab.execution import score_conviction, NEWS_CATEGORY_WEIGHTS
        assert NEWS_CATEGORY_WEIGHTS["management"] == NEWS_CATEGORY_WEIGHTS["earnings"]
        mgmt_score = score_conviction(self._result("management", news_count=1))
        earn_score = score_conviction(self._result("earnings",   news_count=1))
        assert mgmt_score == earn_score

    def test_score_clamped_to_zero_on_downgrade_no_regime(self):
        from quantlab.execution import score_conviction
        # signal(0.30) + no_regime(0) + downgrade(-0.15) = 0.15 — still positive
        r = self._result("downgrade", news_count=1, regime=False)
        assert 0.0 < score_conviction(r) < 0.30

    def test_score_never_exceeds_one(self):
        from quantlab.execution import score_conviction
        r = self._result("earnings", news_count=5, regime=True,
                         rel_vol=2.0, c_score=0.9)
        assert score_conviction(r) <= 1.0

    def test_existing_layering_still_holds(self):
        # Regression: s_signal_only < s_signal_regime < s_signal_regime_earnings
        from quantlab.execution import score_conviction
        s1 = score_conviction(self._result(regime=False, news_count=0))
        s2 = score_conviction(self._result(regime=True,  news_count=0))
        s3 = score_conviction(self._result("earnings", news_count=2,
                                           regime=True, rel_vol=1.8, c_score=0.85))
        assert s1 < s2 < s3


class TestHistoricalEdgeScore:
    """historical_edge_score() must be safe with and without a real DB."""

    def test_returns_neutral_when_no_db(self, tmp_path):
        from quantlab.execution import historical_edge_score
        # Point at a non-existent DB path
        score = historical_edge_score("AAPL", db_path=str(tmp_path / "missing.duckdb"))
        assert score == 0.5

    def test_returns_neutral_for_unknown_symbol(self, tmp_path, monkeypatch):
        import quantlab.storage as _storage
        monkeypatch.setattr(_storage, "DB_PATH", tmp_path / "edge.duckdb")
        from quantlab.execution import historical_edge_score
        # DB exists but has no walk_forward_windows rows for this symbol
        score = historical_edge_score("____NOSYM____",
                                      db_path=str(tmp_path / "edge.duckdb"))
        assert score == 0.5

    def test_positive_oos_sharpe_scores_above_half(self, tmp_path, monkeypatch):
        import quantlab.storage as _storage
        monkeypatch.setattr(_storage, "DB_PATH", tmp_path / "edge.duckdb")
        # Seed the DB with a synthetic walk_forward_windows row
        import duckdb
        con = duckdb.connect(str(tmp_path / "edge.duckdb"))
        con.execute("""
            CREATE TABLE walk_forward_windows (
                run_id VARCHAR, symbol VARCHAR, signal_type VARCHAR,
                lookback INTEGER, window_index INTEGER,
                is_start_bar INTEGER, is_end_bar INTEGER,
                oos_start_bar INTEGER, oos_end_bar INTEGER,
                is_sharpe DOUBLE, is_total_return DOUBLE, is_trade_count INTEGER,
                is_sufficient BOOLEAN,
                oos_sharpe DOUBLE, oos_total_return DOUBLE,
                oos_trade_count INTEGER, oos_sufficient BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        con.execute(
            "INSERT INTO walk_forward_windows VALUES "
            "('run_x','TEST_SYM','breakout',5,0,0,252,252,315,"
            " 1.5, 0.05, 10, true, 4.0, 0.03, 8, true, CURRENT_TIMESTAMP)"
        )
        con.close()
        from quantlab.execution import historical_edge_score
        score = historical_edge_score("TEST_SYM", db_path=str(tmp_path / "edge.duckdb"))
        assert score > 0.5   # avg_oos=4.0 → (4+10)/20 = 0.70

    def test_negative_oos_sharpe_scores_below_half(self, tmp_path, monkeypatch):
        import duckdb
        import quantlab.storage as _storage
        monkeypatch.setattr(_storage, "DB_PATH", tmp_path / "edge2.duckdb")
        con = duckdb.connect(str(tmp_path / "edge2.duckdb"))
        con.execute("""
            CREATE TABLE walk_forward_windows (
                run_id VARCHAR, symbol VARCHAR, signal_type VARCHAR,
                lookback INTEGER, window_index INTEGER,
                is_start_bar INTEGER, is_end_bar INTEGER,
                oos_start_bar INTEGER, oos_end_bar INTEGER,
                is_sharpe DOUBLE, is_total_return DOUBLE, is_trade_count INTEGER,
                is_sufficient BOOLEAN,
                oos_sharpe DOUBLE, oos_total_return DOUBLE,
                oos_trade_count INTEGER, oos_sufficient BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        con.execute(
            "INSERT INTO walk_forward_windows VALUES "
            "('run_y','BAD_SYM','breakout',5,0,0,252,252,315,"
            " -2.0, -0.05, 8, false, -5.0, -0.04, 6, false, CURRENT_TIMESTAMP)"
        )
        con.close()
        from quantlab.execution import historical_edge_score
        score = historical_edge_score("BAD_SYM", db_path=str(tmp_path / "edge2.duckdb"))
        assert score < 0.5   # avg_oos=-5.0 → (-5+10)/20 = 0.25

    def test_extreme_values_clamped(self, tmp_path, monkeypatch):
        import duckdb
        import quantlab.storage as _storage
        monkeypatch.setattr(_storage, "DB_PATH", tmp_path / "edge3.duckdb")
        con = duckdb.connect(str(tmp_path / "edge3.duckdb"))
        con.execute("""
            CREATE TABLE walk_forward_windows (
                run_id VARCHAR, symbol VARCHAR, signal_type VARCHAR,
                lookback INTEGER, window_index INTEGER,
                is_start_bar INTEGER, is_end_bar INTEGER,
                oos_start_bar INTEGER, oos_end_bar INTEGER,
                is_sharpe DOUBLE, is_total_return DOUBLE, is_trade_count INTEGER,
                is_sufficient BOOLEAN,
                oos_sharpe DOUBLE, oos_total_return DOUBLE,
                oos_trade_count INTEGER, oos_sufficient BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Insert extreme outlier (like CRM mock data: 271)
        con.execute(
            "INSERT INTO walk_forward_windows VALUES "
            "('run_z','OUTLIER','breakout',5,0,0,252,252,315,"
            " 5.0, 0.10, 15, true, 271.0, 0.08, 12, true, CURRENT_TIMESTAMP)"
        )
        con.close()
        from quantlab.execution import historical_edge_score
        score = historical_edge_score("OUTLIER", db_path=str(tmp_path / "edge3.duckdb"))
        assert score == 1.0   # clamped to clip_hi


class TestLowEdgeSymbols:

    def test_user_named_symbols_present(self):
        from quantlab.execution import LOW_EDGE_SYMBOLS
        for sym in ("BAC", "PG", "NEE"):
            assert sym in LOW_EDGE_SYMBOLS, f"{sym} should be in LOW_EDGE_SYMBOLS"

    def test_strongly_negative_sharpe_symbols_present(self):
        from quantlab.execution import LOW_EDGE_SYMBOLS
        # All had full-period Sharpe < -1.5 in live IBKR run
        for sym in ("AMGN", "AMZN", "NEE", "PEP", "CVX", "META"):
            assert sym in LOW_EDGE_SYMBOLS

    def test_top_performers_not_flagged(self):
        from quantlab.execution import LOW_EDGE_SYMBOLS
        for sym in ("AAPL", "CAT", "NVDA", "LLY", "CSCO"):
            assert sym not in LOW_EDGE_SYMBOLS

    def test_is_frozenset(self):
        from quantlab.execution import LOW_EDGE_SYMBOLS
        assert isinstance(LOW_EDGE_SYMBOLS, frozenset)


# ══════════════════════════════════════════════════════════════════════════════
# Wyckoff signals (quantlab.signals.wyckoff)
# All tests use synthetic Bar sequences — no market data required.
# ══════════════════════════════════════════════════════════════════════════════

class TestWyckoffSignals:

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _bars(n: int, start_price: float = 100.0,
              trend: float = 0.0, vol_base: float = 1_000_000.0,
              vol_scale: float = 1.0) -> list[Bar]:
        """Generate simple bars with configurable trend and uniform volume."""
        from datetime import timedelta
        bars = []
        price = start_price
        d = date(2025, 1, 6)
        for i in range(n):
            price = price * (1 + trend)
            bars.append(Bar(
                as_of=d + timedelta(days=i),
                open=price * 0.999,
                high=price * 1.005,
                low=price  * 0.995,
                close=price,
                volume=vol_base * vol_scale,
            ))
        return bars

    # ── absorption_score ──────────────────────────────────────────────────────

    def test_absorption_neutral_on_uniform_bars(self):
        from quantlab.signals.wyckoff import absorption_score
        bars = self._bars(80)
        # Flat price, uniform volume → no heavy-volume bars → neutral 0.5
        assert absorption_score(bars) == 0.5

    def test_absorption_high_when_volume_spikes_without_new_lows(self):
        from quantlab.signals.wyckoff import absorption_score
        from datetime import timedelta
        # Build a stable base at price ~100, then inject high-volume bars
        # that do NOT make new lows (classic absorption)
        base = self._bars(40, start_price=100.0, trend=0.0)
        # Add 20 bars: high volume, close around 100, low stays >= 98 (base low)
        d = base[-1].as_of
        for i in range(20):
            base.append(Bar(
                as_of=d + timedelta(days=i + 1),
                open=100.0, high=101.0, low=99.0, close=100.0,
                volume=3_000_000.0,   # 3× average → heavy
            ))
        score = absorption_score(base, volume_threshold=1.3, lookback=60)
        assert score > 0.5   # heavy volume, no new lows → absorption dominant

    def test_absorption_low_when_volume_spikes_with_new_lows(self):
        from quantlab.signals.wyckoff import absorption_score
        from datetime import timedelta
        # Base at 100, then heavy-volume bars that make progressively lower lows
        base = self._bars(40, start_price=100.0, trend=0.0)
        d = base[-1].as_of
        price = 100.0
        for i in range(20):
            price -= 2.5   # aggressive decline: lows drop well below support
            base.append(Bar(
                as_of=d + timedelta(days=i + 1),
                open=price + 0.2, high=price + 0.5,
                low=price - 0.5,
                close=price,
                volume=3_000_000.0,
            ))
        # base_support ≈ 99.5; bars quickly go below 99.5 * 0.985 = 98.0
        score = absorption_score(base, volume_threshold=1.3, lookback=60)
        assert score < 0.5   # heavy volume + lows breaking through support

    def test_absorption_returns_neutral_on_too_few_bars(self):
        from quantlab.signals.wyckoff import absorption_score
        bars = self._bars(10)
        assert absorption_score(bars) == 0.5

    # ── base_quality_score ────────────────────────────────────────────────────

    def test_base_quality_low_on_few_bars(self):
        from quantlab.signals.wyckoff import base_quality_score
        bars = self._bars(20)   # too short for a 12-week base
        assert base_quality_score(bars, min_weeks=12) == 0.0

    def test_base_quality_higher_on_tight_long_base(self):
        from quantlab.signals.wyckoff import base_quality_score
        # 100 bars in a 3% range — tight base, good duration
        tight = self._bars(100, start_price=100.0, trend=0.0)
        score = base_quality_score(tight, min_weeks=8)
        assert score > 0.5

    def test_base_quality_lower_on_wide_volatile_bars(self):
        from quantlab.signals.wyckoff import base_quality_score
        from datetime import timedelta
        # Wide-ranging bars: 30% swings — no base forming
        bars = []
        d = date(2025, 1, 6)
        price = 100.0
        for i in range(80):
            swing = 1.15 if i % 2 == 0 else 0.85   # alternating 15% up/down
            price *= swing
            bars.append(Bar(
                as_of=d + timedelta(days=i),
                open=price * 0.99, high=price * 1.02,
                low=price * 0.98, close=price,
                volume=1_000_000.0,
            ))
        score = base_quality_score(bars, min_weeks=8)
        assert score < 0.5

    def test_base_quality_returns_float_in_range(self):
        from quantlab.signals.wyckoff import base_quality_score
        bars = self._bars(80)
        score = base_quality_score(bars, min_weeks=8)
        assert 0.0 <= score <= 1.0

    # ── volume_character_score ────────────────────────────────────────────────

    def test_volume_character_neutral_on_uniform_volume(self):
        from quantlab.signals.wyckoff import volume_character_score
        # Same volume every bar → no above-average bars → neutral 0.5
        bars = self._bars(80, trend=0.001)
        score = volume_character_score(bars)
        assert score == 0.5

    def test_volume_character_above_half_on_accumulation_pattern(self):
        from quantlab.signals.wyckoff import volume_character_score
        from datetime import timedelta
        # Up-days on 3× volume, down-days on 0.5× volume — pure accumulation
        bars = []
        d = date(2025, 1, 6)
        price = 100.0
        for i in range(80):
            is_up = (i % 2 == 0)
            price *= (1.002 if is_up else 0.999)
            vol = 3_000_000.0 if is_up else 500_000.0
            bars.append(Bar(
                as_of=d + timedelta(days=i),
                open=price * (0.999 if is_up else 1.001),
                high=price * 1.003, low=price * 0.997,
                close=price, volume=vol,
            ))
        score = volume_character_score(bars, lookback=60)
        assert score > 0.55

    def test_volume_character_below_half_on_distribution_pattern(self):
        from quantlab.signals.wyckoff import volume_character_score
        from datetime import timedelta
        # Down-days on 3× volume, up-days on 0.5× volume — distribution
        bars = []
        d = date(2025, 1, 6)
        price = 100.0
        for i in range(80):
            is_up = (i % 2 == 0)
            price *= (1.001 if is_up else 0.998)
            vol = 500_000.0 if is_up else 3_000_000.0   # reversed
            bars.append(Bar(
                as_of=d + timedelta(days=i),
                open=price * (0.999 if is_up else 1.001),
                high=price * 1.002, low=price * 0.998,
                close=price, volume=vol,
            ))
        score = volume_character_score(bars, lookback=60)
        assert score < 0.45

    def test_volume_character_returns_float_in_range(self):
        from quantlab.signals.wyckoff import volume_character_score
        bars = self._bars(80, trend=0.001)
        score = volume_character_score(bars)
        assert 0.0 <= score <= 1.0

    # ── is_wyckoff_spring ─────────────────────────────────────────────────────

    def test_spring_detected_when_undercut_and_recovery(self):
        from quantlab.signals.wyckoff import is_wyckoff_spring
        from datetime import timedelta
        # 70-bar base at 100, then one bar dips to 97.5 (2.5% undercut) and
        # recovers, followed by a close back above 100
        base = self._bars(70, start_price=100.0, trend=0.0)
        d = base[-1].as_of
        support = min(b.low for b in base)  # ≈ 99.5
        # Spring bar: dip below support, high volume
        base.append(Bar(
            as_of=d + timedelta(days=1),
            open=99.8, high=100.5,
            low=support * 0.97,   # undercuts by ~3%
            close=100.2,          # closes above support → recovery
            volume=2_500_000.0,   # elevated volume
        ))
        assert is_wyckoff_spring(base, lookback=60, undercut_pct=0.015,
                                 recovery_bars=3, volume_confirmation=True,
                                 volume_threshold=1.2) is True

    def test_spring_not_detected_on_flat_bars(self):
        from quantlab.signals.wyckoff import is_wyckoff_spring
        bars = self._bars(80, start_price=100.0, trend=0.0)
        # No undercut at all → no spring
        assert is_wyckoff_spring(bars, lookback=60, undercut_pct=0.015) is False

    def test_spring_not_detected_on_too_few_bars(self):
        from quantlab.signals.wyckoff import is_wyckoff_spring
        bars = self._bars(10)
        assert is_wyckoff_spring(bars) is False

    def test_spring_not_detected_when_no_recovery(self):
        from quantlab.signals.wyckoff import is_wyckoff_spring
        from datetime import timedelta
        # Price undercuts support but keeps falling — no recovery
        base = self._bars(70, start_price=100.0, trend=0.0)
        d = base[-1].as_of
        support = min(b.low for b in base)
        # Three bars all staying below support
        for i in range(3):
            base.append(Bar(
                as_of=d + timedelta(days=i + 1),
                open=96.0, high=97.0,
                low=95.0,           # stays well below support
                close=96.0,         # closes below support → no recovery
                volume=2_500_000.0,
            ))
        assert is_wyckoff_spring(base, lookback=60, undercut_pct=0.015,
                                 recovery_bars=3) is False


# ══════════════════════════════════════════════════════════════════════════════
# Wyckoff layers wired into score_conviction()
# ══════════════════════════════════════════════════════════════════════════════

class TestWyckoffConvictionIntegration:
    """Verify Wyckoff fields on ScanResult are used by score_conviction()."""

    def _base_result(self, **kwargs) -> ScanResult:
        defaults = dict(
            symbol="AAPL", scan_date="2026-01-10",
            signal_type="breakout", signal=True,
            entry_close=180.0, indicator_value=179.0, lookback=5,
            regime_bullish=False,   # keep other layers off for isolation
            news_count=0,
            base_quality=0.0, absorption=0.0,
            volume_character=0.0, wyckoff_spring=False,
        )
        defaults.update(kwargs)
        return ScanResult(**defaults)

    def test_base_quality_removed_from_scorer(self):
        # base_quality is anti-predictive on large-caps (AAPL analysis).
        # It must no longer contribute to the conviction score.
        low  = score_conviction(self._base_result(base_quality=0.4))
        high = score_conviction(self._base_result(base_quality=0.7))
        assert high == low  # BQ field on ScanResult is still stored, just not scored

    def test_base_quality_field_exists_but_ignored(self):
        # BQ value stored on ScanResult, score unchanged regardless of value
        r_low = self._base_result(base_quality=0.0)
        r_high = self._base_result(base_quality=1.0)
        assert score_conviction(r_low) == score_conviction(r_high)

    def test_absorption_above_threshold_boosts(self):
        low  = score_conviction(self._base_result(absorption=0.4))
        high = score_conviction(self._base_result(absorption=0.7))
        assert high - low == pytest.approx(0.10, abs=1e-9)

    def test_volume_character_above_threshold_boosts(self):
        low  = score_conviction(self._base_result(volume_character=0.4))
        high = score_conviction(self._base_result(volume_character=0.7))
        assert high - low == pytest.approx(0.10, abs=1e-9)

    def test_wyckoff_spring_boosts(self):
        no_spring  = score_conviction(self._base_result(wyckoff_spring=False))
        with_spring = score_conviction(self._base_result(wyckoff_spring=True))
        assert with_spring - no_spring == pytest.approx(0.10, abs=1e-9)

    def test_three_wyckoff_layers_stack(self):
        # base_quality removed; only absorption + vol_character + spring remain
        no_wyckoff  = score_conviction(self._base_result())
        all_wyckoff = score_conviction(self._base_result(
            absorption=0.8, volume_character=0.8, wyckoff_spring=True,
        ))
        # +0.10 + 0.10 + 0.10 = +0.30
        assert all_wyckoff - no_wyckoff == pytest.approx(0.30, abs=1e-9)

    def test_fully_confirmed_wyckoff_plus_regime_plus_earnings_clamped(self):
        r = self._base_result(
            regime_bullish=True, news_count=1, news_category="earnings",
            rel_volume=2.0, news_c_score=0.9,
            base_quality=0.8, absorption=0.8,   # base_quality ignored
            volume_character=0.8, wyckoff_spring=True,
        )
        # 0.30+0.20+0.20+0.10+0.10+0.10+0.10+0.10 = 1.20 → clamped to 1.0
        assert score_conviction(r) == pytest.approx(1.0, abs=1e-9)

    def test_downgrade_reduces_below_signal_only(self):
        # Downgrade veto still works; base_quality no longer cancels it
        with_downgrade = score_conviction(self._base_result(
            news_count=1, news_category="downgrade",
        ))
        signal_only = score_conviction(self._base_result())
        assert with_downgrade < signal_only  # -0.15 from downgrade

    def test_scan_result_wyckoff_fields_default_to_zero(self):
        r = ScanResult(
            symbol="AAPL", scan_date="2026-01-10",
            signal_type="breakout", signal=True,
            entry_close=180.0, indicator_value=None, lookback=5,
        )
        assert r.base_quality == 0.0
        assert r.absorption == 0.0
        assert r.volume_character == 0.0
        assert r.wyckoff_spring is False

    def test_scan_symbol_populates_wyckoff_fields(self):
        """scan_symbol() must compute and return non-None Wyckoff scores."""
        bars = make_bars(100, trend=0.002)
        result = scan_symbol("AAPL", bars, signal_type="breakout", lookback=5)
        assert result is not None
        assert isinstance(result.base_quality, float)
        assert isinstance(result.absorption, float)
        assert isinstance(result.volume_character, float)
        assert isinstance(result.wyckoff_spring, bool)
        assert 0.0 <= result.base_quality <= 1.0
        assert 0.0 <= result.absorption <= 1.0
        assert 0.0 <= result.volume_character <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# stock_profile() classification
# ══════════════════════════════════════════════════════════════════════════════

class TestStockProfile:

    def test_mega_cap_symbols_classified_correctly(self):
        from quantlab.execution import stock_profile
        for sym in ("AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META"):
            assert stock_profile(sym) == "mega_cap_liquid", sym

    def test_sp500_non_mega_classified_as_large_cap(self):
        from quantlab.execution import stock_profile
        for sym in ("CAT", "LLY", "JPM", "XOM", "CSCO", "GS"):
            assert stock_profile(sym) == "large_cap_growth", sym

    def test_unknown_symbol_classified_as_mid_cap(self):
        from quantlab.execution import stock_profile
        assert stock_profile("HYPOTHETICAL") == "mid_cap_growth"
        assert stock_profile("XYZ") == "mid_cap_growth"

    def test_mega_cap_set_is_frozenset(self):
        from quantlab.execution import MEGA_CAP_LIQUID
        assert isinstance(MEGA_CAP_LIQUID, frozenset)
        assert len(MEGA_CAP_LIQUID) == 6

    def test_tsla_classified_large_cap_not_mega(self):
        # TSLA is in SP500_SAMPLE but not MEGA_CAP_LIQUID
        from quantlab.execution import stock_profile
        assert stock_profile("TSLA") == "large_cap_growth"

    def test_watchlist_small_symbols_are_large_cap(self):
        from quantlab.execution import stock_profile, WATCHLIST_SMALL
        for sym in WATCHLIST_SMALL:
            profile = stock_profile(sym)
            # All WATCHLIST_SMALL are either mega or large cap
            assert profile in ("mega_cap_liquid", "large_cap_growth"), sym
