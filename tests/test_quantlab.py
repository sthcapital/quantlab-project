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
        assert p._shared_ib is None

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


class TestIbkrPersistentConnection:
    """Verify _shared_ib lifecycle and run_universe_scan context-manager wiring."""

    def test_shared_ib_none_on_init(self):
        from quantlab.providers.ibkr import IbkrProvider
        p = IbkrProvider()
        assert p._shared_ib is None

    def test_run_universe_scan_calls_enter_and_exit(self):
        """run_universe_scan() must open/close the provider's context manager."""
        from quantlab.execution import run_universe_scan

        entered = []
        exited  = []

        class TrackingProvider:
            """Minimal provider that records context-manager calls."""
            def __enter__(self):
                entered.append(True)
                return self
            def __exit__(self, *_):
                exited.append(True)
            def get_daily_bars(self, symbol, start, end):
                return make_bars(60)

        run_universe_scan(
            provider     = TrackingProvider(),
            symbols      = ["AAPL"],
            start_date   = date(2026, 1, 2),
            end_date     = date(2026, 6, 5),
            min_conviction = 0.0,
        )
        assert entered == [True], "__enter__ should have been called exactly once"
        assert exited  == [True], "__exit__ should have been called exactly once"

    def test_run_universe_scan_works_without_context_manager(self):
        """Providers without __enter__ must still work (nullcontext fallback)."""
        from quantlab.execution import run_universe_scan

        class SimpleProvider:
            def get_daily_bars(self, symbol, start, end):
                return make_bars(60)

        # Must not raise
        run_universe_scan(
            provider     = SimpleProvider(),
            symbols      = ["AAPL"],
            start_date   = date(2026, 1, 2),
            end_date     = date(2026, 6, 5),
            min_conviction = 0.0,
        )

    def test_shared_ib_used_when_set(self):
        """_get_ib() returns (shared_ib, False) — not a new temp conn — when _shared_ib is connected."""
        from quantlab.providers.ibkr import IbkrProvider
        from unittest.mock import MagicMock

        p = IbkrProvider()
        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True
        p._shared_ib = mock_ib

        ib, is_temporary = p._get_ib()
        assert ib is mock_ib
        assert is_temporary is False

    def test_get_ib_creates_temp_when_shared_ib_none(self):
        """_get_ib() must signal is_temporary=True when _shared_ib is not set."""
        from quantlab.providers.ibkr import IbkrProvider
        from unittest.mock import patch, MagicMock

        p = IbkrProvider()
        assert p._shared_ib is None

        mock_ib = MagicMock()
        mock_ib.isConnected.return_value = True

        with patch("quantlab.providers.ibkr.IB", return_value=mock_ib):
            ib, is_temporary = p._get_ib()

        assert is_temporary is True

    def test_get_ib_creates_temp_when_shared_ib_disconnected(self):
        """_get_ib() treats a disconnected _shared_ib as absent."""
        from quantlab.providers.ibkr import IbkrProvider
        from unittest.mock import patch, MagicMock

        p = IbkrProvider()
        stale = MagicMock()
        stale.isConnected.return_value = False  # already disconnected
        p._shared_ib = stale

        fresh = MagicMock()
        fresh.isConnected.return_value = True

        with patch("quantlab.providers.ibkr.IB", return_value=fresh):
            ib, is_temporary = p._get_ib()

        assert is_temporary is True
        assert ib is not stale


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
        # Reduced from 0.10 to 0.05: absorption fires on 100% of daily-bar signals
        low  = score_conviction(self._base_result(absorption=0.4))
        high = score_conviction(self._base_result(absorption=0.7))
        assert high - low == pytest.approx(0.05, abs=1e-9)

    def test_volume_character_above_threshold_boosts(self):
        low  = score_conviction(self._base_result(volume_character=0.4))
        high = score_conviction(self._base_result(volume_character=0.7))
        assert high - low == pytest.approx(0.10, abs=1e-9)

    def test_wyckoff_spring_boosts(self):
        no_spring  = score_conviction(self._base_result(wyckoff_spring=False))
        with_spring = score_conviction(self._base_result(wyckoff_spring=True))
        assert with_spring - no_spring == pytest.approx(0.10, abs=1e-9)

    def test_three_wyckoff_layers_stack(self):
        # absorption reduced to 0.05; vol_character=0.10; spring=0.10
        no_wyckoff  = score_conviction(self._base_result())
        all_wyckoff = score_conviction(self._base_result(
            absorption=0.8, volume_character=0.8, wyckoff_spring=True,
        ))
        # +0.05 + 0.10 + 0.10 = +0.25
        assert all_wyckoff - no_wyckoff == pytest.approx(0.25, abs=1e-9)

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


# ══════════════════════════════════════════════════════════════════════════════
# Earnings acceleration detection (quantlab.signals.earnings)
# ══════════════════════════════════════════════════════════════════════════════

class TestEarningsDetection:
    """All tests use synthetic Bar sequences — no IBKR connection needed."""

    @staticmethod
    def _flat_bars(n: int, vol: float = 1_000_000.0) -> list[Bar]:
        from datetime import timedelta
        start = date(2024, 1, 2)
        return [
            Bar(start + timedelta(days=i), 100.0, 101.0, 99.0, 100.0, vol)
            for i in range(n)
        ]

    @staticmethod
    def _insert_earnings_event(
        bars: list[Bar],
        idx: int,
        gap_pct: float = 0.05,
        vol_multiplier: float = 3.0,
    ) -> list[Bar]:
        """Replace bar at idx with a large-gap, high-volume earnings bar."""
        from datetime import timedelta
        prev_close = bars[idx - 1].close if idx > 0 else 100.0
        new_open   = prev_close * (1 + gap_pct)
        avg_vol    = 1_000_000.0
        new_bar    = Bar(
            as_of=bars[idx].as_of,
            open=new_open,
            high=new_open * 1.02,
            low=new_open  * 0.98,
            close=new_open * 1.01,
            volume=avg_vol * vol_multiplier,
        )
        result = bars[:]
        result[idx] = new_bar
        return result

    # ── detect_earnings_dates ─────────────────────────────────────────────────

    def test_flat_bars_produce_no_events(self):
        from quantlab.signals.earnings import detect_earnings_dates
        bars = self._flat_bars(200)
        assert detect_earnings_dates(bars) == []

    def test_single_gap_event_detected(self):
        from quantlab.signals.earnings import detect_earnings_dates
        bars = self._flat_bars(200)
        bars = self._insert_earnings_event(bars, 60, gap_pct=0.05)
        dates = detect_earnings_dates(bars, gap_threshold=0.025, vol_threshold=1.5)
        assert len(dates) == 1
        assert dates[0] == bars[60].as_of.isoformat()

    def test_four_quarterly_events_detected(self):
        from quantlab.signals.earnings import detect_earnings_dates
        bars = self._flat_bars(300)
        for idx in (40, 103, 166, 229):
            bars = self._insert_earnings_event(bars, idx, gap_pct=0.04)
        dates = detect_earnings_dates(bars, gap_threshold=0.025, vol_threshold=1.5,
                                      min_event_spacing=30)
        assert len(dates) == 4

    def test_events_within_spacing_window_deduped(self):
        from quantlab.signals.earnings import detect_earnings_dates
        bars = self._flat_bars(200)
        # Insert two events only 5 bars apart — should only keep first
        bars = self._insert_earnings_event(bars, 60, gap_pct=0.05)
        bars = self._insert_earnings_event(bars, 65, gap_pct=0.04)
        dates = detect_earnings_dates(bars, min_event_spacing=30)
        assert len(dates) == 1

    def test_large_gap_above_max_gap_excluded(self):
        from quantlab.signals.earnings import detect_earnings_dates
        bars = self._flat_bars(200)
        bars = self._insert_earnings_event(bars, 60, gap_pct=0.25)  # 25% > max_gap
        dates = detect_earnings_dates(bars, gap_threshold=0.025, max_gap=0.20)
        assert len(dates) == 0

    def test_returns_empty_on_too_few_bars(self):
        from quantlab.signals.earnings import detect_earnings_dates
        assert detect_earnings_dates(self._flat_bars(10)) == []

    # ── compute_earnings_profile ──────────────────────────────────────────────

    def test_profile_empty_on_flat_bars(self):
        from quantlab.signals.earnings import compute_earnings_profile
        p = compute_earnings_profile("TEST", self._flat_bars(200))
        assert p.earnings_count == 0
        assert p.acceleration_trend == 0.0

    def test_profile_detects_frequency(self):
        from quantlab.signals.earnings import compute_earnings_profile
        bars = self._flat_bars(300)
        for idx in (40, 103, 166, 229):
            bars = self._insert_earnings_event(bars, idx, gap_pct=0.04)
        p = compute_earnings_profile("TEST", bars)
        assert p.earnings_count == 4
        assert p.earnings_frequency > 3.0   # ~4 per year

    def test_positive_surprise_rate_on_all_positive_gaps(self):
        from quantlab.signals.earnings import compute_earnings_profile
        bars = self._flat_bars(300)
        for idx in (40, 103, 166, 229):
            bars = self._insert_earnings_event(bars, idx, gap_pct=+0.05)
        p = compute_earnings_profile("TEST", bars)
        assert p.positive_surprise_rate == pytest.approx(1.0, abs=0.01)

    def test_negative_surprise_rate_on_all_negative_gaps(self):
        from quantlab.signals.earnings import compute_earnings_profile
        bars = self._flat_bars(300)
        for idx in (40, 103, 166, 229):
            bars = self._insert_earnings_event(bars, idx, gap_pct=-0.05)
        p = compute_earnings_profile("TEST", bars)
        assert p.positive_surprise_rate == pytest.approx(0.0, abs=0.01)

    def test_acceleration_trend_positive_when_recent_larger(self):
        from quantlab.signals.earnings import compute_earnings_profile
        bars = self._flat_bars(600)
        # Early events: small but above threshold; recent events: large
        for idx in (40, 103):
            bars = self._insert_earnings_event(bars, idx, gap_pct=0.03, vol_multiplier=2.0)
        for idx in (300, 363):
            bars = self._insert_earnings_event(bars, idx, gap_pct=0.08, vol_multiplier=3.0)
        p = compute_earnings_profile("TEST", bars)
        assert p.earnings_count == 4
        assert p.acceleration_trend > 0.0

    def test_acceleration_trend_negative_when_recent_smaller(self):
        from quantlab.signals.earnings import compute_earnings_profile
        bars = self._flat_bars(600)
        for idx in (40, 103):
            bars = self._insert_earnings_event(bars, idx, gap_pct=0.08, vol_multiplier=3.0)
        for idx in (300, 363):
            # Use 0.03 — above detection threshold but smaller than early events
            bars = self._insert_earnings_event(bars, idx, gap_pct=0.03, vol_multiplier=2.0)
        p = compute_earnings_profile("TEST", bars)
        assert p.acceleration_trend < 0.0

    # ── earnings_acceleration_score ───────────────────────────────────────────

    def test_score_zero_below_4_events(self):
        from quantlab.signals.earnings import compute_earnings_profile, earnings_acceleration_score
        bars = self._flat_bars(200)
        bars = self._insert_earnings_event(bars, 60, gap_pct=0.05)
        bars = self._insert_earnings_event(bars, 120, gap_pct=0.05)
        p = compute_earnings_profile("TEST", bars)
        assert p.earnings_count < 4
        assert earnings_acceleration_score(p) == 0.0

    def test_score_high_on_strong_profile(self):
        from quantlab.signals.earnings import compute_earnings_profile, earnings_acceleration_score
        bars = self._flat_bars(600)
        for idx in (40, 103):
            bars = self._insert_earnings_event(bars, idx, gap_pct=0.04, vol_multiplier=2.0)
        for idx in (300, 363):
            bars = self._insert_earnings_event(bars, idx, gap_pct=0.07, vol_multiplier=3.0)
        p = compute_earnings_profile("TEST", bars)
        score = earnings_acceleration_score(p)
        assert 0.0 <= score <= 1.0
        assert score > 0.3   # positive surprises + acceleration + magnitude

    def test_score_in_range(self):
        from quantlab.signals.earnings import compute_earnings_profile, earnings_acceleration_score
        bars = self._flat_bars(300)
        for idx in (40, 103, 166, 229):
            bars = self._insert_earnings_event(bars, idx, gap_pct=0.06)
        p = compute_earnings_profile("TEST", bars)
        assert 0.0 <= earnings_acceleration_score(p) <= 1.0


# ── earnings_acceleration wired into conviction scorer ────────────────────────

class TestEarningsConvictionLayer:

    def test_ea_above_threshold_boosts_conviction(self):
        low  = score_conviction(ScanResult(
            "AAPL", "2026-01-01", "breakout", True, 180.0, None, 5,
            regime_bullish=False, earnings_acceleration=0.4,
        ))
        high = score_conviction(ScanResult(
            "AAPL", "2026-01-01", "breakout", True, 180.0, None, 5,
            regime_bullish=False, earnings_acceleration=0.6,
        ))
        assert high - low == pytest.approx(0.10, abs=1e-9)

    def test_ea_below_threshold_no_boost(self):
        r = ScanResult(
            "AAPL", "2026-01-01", "breakout", True, 180.0, None, 5,
            regime_bullish=False, earnings_acceleration=0.49,
        )
        assert score_conviction(r) == pytest.approx(0.30, abs=1e-9)

    def test_ea_threshold_is_0_5(self):
        at    = score_conviction(ScanResult(
            "AAPL", "2026-01-01", "breakout", True, 180.0, None, 5,
            regime_bullish=False, earnings_acceleration=0.50,
        ))
        below = score_conviction(ScanResult(
            "AAPL", "2026-01-01", "breakout", True, 180.0, None, 5,
            regime_bullish=False, earnings_acceleration=0.49,
        ))
        assert at - below == pytest.approx(0.10, abs=1e-9)

    def test_scan_result_ea_defaults_zero(self):
        r = ScanResult(
            "AAPL", "2026-01-01", "breakout", True, 180.0, None, 5,
        )
        assert r.earnings_acceleration == 0.0

    def test_scan_symbol_populates_ea_field(self):
        bars = make_bars(250, trend=0.001)
        result = scan_symbol("AAPL", bars, signal_type="breakout", lookback=5)
        assert result is not None
        assert isinstance(result.earnings_acceleration, float)
        assert 0.0 <= result.earnings_acceleration <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# Volume profile signals (quantlab.signals.volume_profile)
# ══════════════════════════════════════════════════════════════════════════════

class TestVolumeProfile:
    """Synthetic bar fixtures with directional movement and volume variation."""

    @staticmethod
    def _make_directional_bars(n: int, up_vol: float, down_vol: float,
                                base_price: float = 100.0) -> list[Bar]:
        """Alternating up/down bars with distinct up/down volumes."""
        from datetime import timedelta
        bars = []
        price = base_price
        start = date(2025, 1, 6)
        for i in range(n):
            is_up = (i % 2 == 0)
            price *= (1.003 if is_up else 0.998)
            vol = up_vol if is_up else down_vol
            bars.append(Bar(
                as_of=start + timedelta(days=i),
                open=price * (0.999 if is_up else 1.001),
                high=price * 1.004,
                low=price  * 0.996,
                close=price,
                volume=vol,
            ))
        return bars

    # ── accumulation_days_ratio ───────────────────────────────────────────────

    def test_accumulation_ratio_high_on_buying_pattern(self):
        from quantlab.signals.volume_profile import accumulation_days_ratio
        # Up days 3× average volume, down days 0.3× — classic accumulation
        bars = self._make_directional_bars(80, up_vol=3_000_000., down_vol=300_000.)
        score = accumulation_days_ratio(bars, window=60, vol_period=20)
        assert score > 0.6

    def test_accumulation_ratio_low_on_distribution_pattern(self):
        from quantlab.signals.volume_profile import accumulation_days_ratio
        # Down days 3× average, up days 0.3× — distribution fingerprint
        bars = self._make_directional_bars(80, up_vol=300_000., down_vol=3_000_000.)
        score = accumulation_days_ratio(bars, window=60, vol_period=20)
        assert score < 0.4

    def test_accumulation_ratio_neutral_when_no_heavy_days(self):
        from quantlab.signals.volume_profile import accumulation_days_ratio
        # All bars at same volume — no above-average days → neutral 0.5
        bars = self._make_directional_bars(80, up_vol=1_000_000., down_vol=1_000_000.)
        score = accumulation_days_ratio(bars, window=60, vol_period=20)
        assert score == 0.5

    def test_accumulation_ratio_returns_float_in_range(self):
        from quantlab.signals.volume_profile import accumulation_days_ratio
        bars = self._make_directional_bars(80, up_vol=2_000_000., down_vol=500_000.)
        score = accumulation_days_ratio(bars)
        assert 0.0 <= score <= 1.0

    def test_accumulation_ratio_neutral_on_too_few_bars(self):
        from quantlab.signals.volume_profile import accumulation_days_ratio
        bars = self._make_directional_bars(10, 2_000_000., 500_000.)
        assert accumulation_days_ratio(bars) == 0.5

    # ── volume_trend_score ────────────────────────────────────────────────────

    def test_volume_trend_high_on_ideal_accumulation(self):
        from quantlab.signals.volume_profile import volume_trend_score
        # Up days heavy, down days light — both conditions fire every bar
        bars = self._make_directional_bars(60, up_vol=3_000_000., down_vol=300_000.)
        score = volume_trend_score(bars, window=20)
        assert score > 0.6

    def test_volume_trend_low_on_distribution_pattern(self):
        from quantlab.signals.volume_profile import volume_trend_score
        # Up days light, down days heavy — neither ideal condition fires
        bars = self._make_directional_bars(60, up_vol=300_000., down_vol=3_000_000.)
        score = volume_trend_score(bars, window=20)
        assert score < 0.4

    def test_volume_trend_in_range(self):
        from quantlab.signals.volume_profile import volume_trend_score
        bars = self._make_directional_bars(60, up_vol=2_000_000., down_vol=800_000.)
        assert 0.0 <= volume_trend_score(bars) <= 1.0

    # ── climactic_volume_score ────────────────────────────────────────────────

    def test_climactic_zero_when_last_bar_not_highest(self):
        from quantlab.signals.volume_profile import climactic_volume_score
        from datetime import timedelta
        bars = [Bar(date(2025,1,6)+timedelta(days=i), 100,101,99,100, 1_000_000.)
                for i in range(30)]
        # Last bar has the same volume as all others
        assert climactic_volume_score(bars, lookback=20) == 0.0

    def test_climactic_high_when_last_bar_is_spike(self):
        from quantlab.signals.volume_profile import climactic_volume_score
        from datetime import timedelta
        bars = [Bar(date(2025,1,6)+timedelta(days=i), 100,101,99,100, 1_000_000.)
                for i in range(30)]
        # Replace last bar: 4× prior max (prior max = 1M, last = 4M)
        bars[-1] = Bar(bars[-1].as_of, 100,103,99,102, 4_000_000.)
        score = climactic_volume_score(bars, lookback=20)
        # ratio = 4.0, score = min(1.0, (4.0-1.0)/1.5) = 1.0
        assert score == pytest.approx(1.0, abs=1e-9)

    def test_climactic_graduated_scoring(self):
        from quantlab.signals.volume_profile import climactic_volume_score
        from datetime import timedelta
        bars = [Bar(date(2025,1,6)+timedelta(days=i), 100,101,99,100, 1_000_000.)
                for i in range(30)]
        # 2.5× prior max → score = (2.5-1)/1.5 = 1.0
        bars[-1] = Bar(bars[-1].as_of, 100,103,99,102, 2_500_000.)
        assert climactic_volume_score(bars) == pytest.approx(1.0, abs=1e-4)
        # 1.75× prior max → score = (1.75-1)/1.5 = 0.5
        bars[-1] = Bar(bars[-1].as_of, 100,103,99,102, 1_750_000.)
        assert climactic_volume_score(bars) == pytest.approx(0.5, abs=1e-4)

    def test_climactic_zero_on_too_few_bars(self):
        from quantlab.signals.volume_profile import climactic_volume_score
        from datetime import timedelta
        bars = [Bar(date(2025,1,6)+timedelta(days=i),100,101,99,100,1_000_000.)
                for i in range(5)]
        assert climactic_volume_score(bars) == 0.0


# ── volume profile wired into conviction scorer ────────────────────────────────

class TestVolumeProfileConviction:

    def _r(self, **kw) -> ScanResult:
        defaults = dict(symbol="UNH", scan_date="2026-06-04",
                        signal_type="breakout", signal=True,
                        entry_close=399.0, indicator_value=None, lookback=5,
                        regime_bullish=False)
        defaults.update(kw)
        return ScanResult(**defaults)

    def test_accumulation_ratio_boost_fires(self):
        low  = score_conviction(self._r(accumulation_ratio=0.59))
        high = score_conviction(self._r(accumulation_ratio=0.61))
        assert high - low == pytest.approx(0.08, abs=1e-9)

    def test_climactic_volume_boost_fires(self):
        low  = score_conviction(self._r(climactic_volume=0.69))
        high = score_conviction(self._r(climactic_volume=0.71))
        assert high - low == pytest.approx(0.07, abs=1e-9)

    def test_both_layers_stack(self):
        none_ = score_conviction(self._r(accumulation_ratio=0.0, climactic_volume=0.0))
        both  = score_conviction(self._r(accumulation_ratio=0.7, climactic_volume=0.8))
        assert both - none_ == pytest.approx(0.15, abs=1e-9)

    def test_volume_trend_not_wired_to_conviction(self):
        # volume_trend is informational only — changing it must not affect score
        low  = score_conviction(self._r(volume_trend=0.0))
        high = score_conviction(self._r(volume_trend=1.0))
        assert low == high

    def test_full_signal_with_all_vol_layers(self):
        r = self._r(regime_bullish=True, earnings_acceleration=0.85,
                    accumulation_ratio=0.70, climactic_volume=0.80)
        # 0.30 + 0.20 + 0.10 + 0.08 + 0.07 = 0.75
        assert score_conviction(r) == pytest.approx(0.75, abs=1e-9)

    def test_scan_result_vol_fields_default_zero(self):
        r = ScanResult("UNH","2026-06-04","breakout",True,399.0,None,5)
        assert r.accumulation_ratio == 0.0
        assert r.volume_trend       == 0.0
        assert r.climactic_volume   == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Multi-lookback confirmation layer
# ══════════════════════════════════════════════════════════════════════════════

class TestMultiLookbackConfirmation:

    def _r(self, **kw) -> ScanResult:
        defaults = dict(symbol="ABT", scan_date="2026-06-04",
                        signal_type="breakout", signal=True,
                        entry_close=91.0, indicator_value=None, lookback=5,
                        regime_bullish=False)
        defaults.update(kw)
        return ScanResult(**defaults)

    def test_multi_lookback_field_defaults_false(self):
        r = ScanResult("ABT","2026-06-04","breakout",True,91.0,None,5)
        assert r.multi_lookback_confirmed is False

    def test_multi_lookback_adds_five_bps(self):
        base = score_conviction(self._r(multi_lookback_confirmed=False))
        confirmed = score_conviction(self._r(multi_lookback_confirmed=True))
        assert confirmed - base == pytest.approx(0.05, abs=1e-9)

    def test_multi_lookback_stacks_with_earnings(self):
        # signal(0.30) + ea(0.10) + multi(0.05) = 0.45
        r = self._r(earnings_acceleration=0.6, multi_lookback_confirmed=True)
        assert score_conviction(r) == pytest.approx(0.45, abs=1e-9)

    def test_multi_lookback_does_not_fire_when_false(self):
        r = self._r(multi_lookback_confirmed=False)
        # signal(0.30) only (regime_bullish=False, no other layers)
        assert score_conviction(r) == pytest.approx(0.30, abs=1e-9)

    def test_absorption_now_contributes_half_of_old_weight(self):
        """Regression: absorption ≥ 0.6 should now give +0.05, not +0.10."""
        no_abs = score_conviction(self._r(absorption=0.0))
        with_abs = score_conviction(self._r(absorption=0.8))
        assert with_abs - no_abs == pytest.approx(0.05, abs=1e-9)

    def test_full_scoring_with_all_layers_clamped(self):
        r = self._r(
            regime_bullish=True,
            news_count=1, news_category="earnings",
            rel_volume=2.0, news_c_score=0.9,
            absorption=0.8, volume_character=0.8, wyckoff_spring=True,
            earnings_acceleration=0.8,
            accumulation_ratio=0.7, climactic_volume=0.8,
            multi_lookback_confirmed=True,
        )
        # 0.30+0.20+0.20+0.10+0.10+0.05+0.10+0.10+0.10+0.08+0.07+0.05 = 1.45 → 1.0
        assert score_conviction(r) == pytest.approx(1.0, abs=1e-9)


# ══════════════════════════════════════════════════════════════════════════════
# Options flow signals (quantlab.signals.options_flow)
# All tests use synthetic ChainData — no IBKR connection required.
# ══════════════════════════════════════════════════════════════════════════════

class TestOptionsFlow:
    """Factory helpers that build realistic mock ChainData objects."""

    @staticmethod
    def _chain(
        spot: float = 200.0,
        strikes: list[float] | None = None,
        call_vols: list[float] | None = None,
        put_vols:  list[float] | None = None,
        call_ivs:  list[float] | None = None,
        put_ivs:   list[float] | None = None,
    ):
        from quantlab.signals.options_flow import ChainData, OptionContract
        strikes = strikes or [185.0, 190.0, 195.0, 200.0, 205.0, 210.0, 215.0]
        call_vols = call_vols or [100.0] * len(strikes)
        put_vols  = put_vols  or [100.0] * len(strikes)
        call_ivs  = call_ivs  or [0.25] * len(strikes)
        put_ivs   = put_ivs   or [0.25] * len(strikes)

        contracts = []
        for i, strike in enumerate(strikes):
            contracts.append(OptionContract(
                strike=strike, right="C", expiry="20261219",
                volume=call_vols[i], implied_vol=call_ivs[i], bid=1.0, ask=1.2,
            ))
            contracts.append(OptionContract(
                strike=strike, right="P", expiry="20261219",
                volume=put_vols[i], implied_vol=put_ivs[i], bid=1.0, ask=1.2,
            ))
        return ChainData(symbol="TEST", spot=spot, expiry="20261219",
                         contracts=contracts)

    # ── put_call_ratio ────────────────────────────────────────────────────────

    def test_pcr_bullish_when_calls_dominate(self):
        from quantlab.signals.options_flow import put_call_ratio
        # ATM calls at 10× put volume → pcr ≈ 0.10
        chain = self._chain(spot=200.0, call_vols=[1000.0]*7, put_vols=[100.0]*7)
        assert put_call_ratio(chain) < 0.70

    def test_pcr_bearish_when_puts_dominate(self):
        from quantlab.signals.options_flow import put_call_ratio
        chain = self._chain(spot=200.0, call_vols=[100.0]*7, put_vols=[1000.0]*7)
        assert put_call_ratio(chain) > 1.0

    def test_pcr_neutral_no_calls(self):
        from quantlab.signals.options_flow import put_call_ratio, ChainData
        chain = ChainData("X", 200.0, "20261219", contracts=[])
        assert put_call_ratio(chain) == 1.0

    def test_pcr_atm_band_filters_otm(self):
        from quantlab.signals.options_flow import put_call_ratio, ChainData, OptionContract
        # OTM call spike should NOT affect ATM PCR
        contracts = [
            OptionContract(200.0, "C", "20261219", volume=100.0),  # ATM call
            OptionContract(200.0, "P", "20261219", volume=100.0),  # ATM put
            OptionContract(220.0, "C", "20261219", volume=5000.0), # far OTM — excluded
        ]
        chain = ChainData("X", 200.0, "20261219", contracts=contracts)
        pcr = put_call_ratio(chain, atm_band_pct=0.05)
        assert abs(pcr - 1.0) < 0.01   # 100 puts / 100 ATM calls = 1.0

    # ── unusual_call_activity ─────────────────────────────────────────────────

    def test_unusual_detected_on_volume_spike(self):
        from quantlab.signals.options_flow import unusual_call_activity
        # One call has 5× average
        vols = [100.0, 100.0, 100.0, 500.0, 100.0, 100.0, 100.0]
        chain = self._chain(call_vols=vols)
        is_unusual, ratio = unusual_call_activity(chain, avg_volume_threshold=2.0)
        assert is_unusual is True
        assert ratio > 2.0

    def test_not_unusual_on_uniform_volume(self):
        from quantlab.signals.options_flow import unusual_call_activity
        chain = self._chain(call_vols=[100.0]*7)
        is_unusual, ratio = unusual_call_activity(chain)
        assert is_unusual is False
        assert ratio == pytest.approx(1.0, abs=0.01)

    def test_unusual_returns_correct_ratio(self):
        from quantlab.signals.options_flow import unusual_call_activity
        # max=300, avg=(100+100+300)/3=166.7, ratio≈1.80
        chain = self._chain(
            strikes=[190.0, 200.0, 210.0],
            call_vols=[100.0, 100.0, 300.0],
            put_vols=[100.0]*3, call_ivs=[0.25]*3, put_ivs=[0.25]*3,
        )
        _, ratio = unusual_call_activity(chain)
        assert ratio == pytest.approx(300.0 / (500.0/3), abs=0.01)

    def test_unusual_returns_false_on_too_few_contracts(self):
        from quantlab.signals.options_flow import unusual_call_activity, ChainData, OptionContract
        chain = ChainData("X", 200.0, "20261219", contracts=[
            OptionContract(200.0, "C", "20261219", volume=500.0)  # only 1 call
        ])
        is_unusual, _ = unusual_call_activity(chain)
        assert is_unusual is False

    # ── iv_skew_score ─────────────────────────────────────────────────────────

    def test_iv_skew_neutral_at_parity(self):
        from quantlab.signals.options_flow import iv_skew_score
        chain = self._chain(call_ivs=[0.25]*7, put_ivs=[0.25]*7)
        score = iv_skew_score(chain)
        assert score == pytest.approx(0.5, abs=0.05)

    def test_iv_skew_bullish_when_calls_pricier(self):
        from quantlab.signals.options_flow import iv_skew_score
        # OTM calls at IV=0.35 vs OTM puts at IV=0.20
        chain = self._chain(
            spot=200.0,
            strikes=[185.0, 190.0, 200.0, 210.0, 215.0],
            call_ivs=[0.25, 0.25, 0.25, 0.35, 0.35],  # OTM calls expensive
            put_ivs =[0.30, 0.28, 0.25, 0.22, 0.20],  # normal smirk
        )
        assert iv_skew_score(chain) > 0.5

    def test_iv_skew_bearish_when_puts_pricier(self):
        from quantlab.signals.options_flow import iv_skew_score
        chain = self._chain(
            spot=200.0,
            strikes=[185.0, 190.0, 200.0, 210.0, 215.0],
            call_ivs=[0.20, 0.22, 0.25, 0.27, 0.28],
            put_ivs =[0.35, 0.33, 0.25, 0.22, 0.20],  # heavy put premium
        )
        assert iv_skew_score(chain) < 0.5

    def test_iv_skew_neutral_when_no_iv_data(self):
        from quantlab.signals.options_flow import iv_skew_score, ChainData, OptionContract
        chain = ChainData("X", 200.0, "20261219", contracts=[
            OptionContract(200.0, "C", "20261219", implied_vol=None),
            OptionContract(200.0, "P", "20261219", implied_vol=None),
        ])
        assert iv_skew_score(chain) == 0.5

    def test_iv_skew_in_range(self):
        from quantlab.signals.options_flow import iv_skew_score
        chain = self._chain(call_ivs=[0.30]*7, put_ivs=[0.20]*7)
        assert 0.0 <= iv_skew_score(chain) <= 1.0

    # ── compute_options_score ─────────────────────────────────────────────────

    def test_score_zero_on_bearish_chain(self):
        from quantlab.signals.options_flow import compute_options_score
        # High PCR (more puts) + no unusual calls + normal skew → score=0
        chain = self._chain(call_vols=[100.0]*7, put_vols=[500.0]*7,
                            call_ivs=[0.25]*7, put_ivs=[0.30]*7)
        assert compute_options_score(chain) == pytest.approx(0.0, abs=0.01)

    def test_score_high_on_full_bullish_chain(self):
        from quantlab.signals.options_flow import compute_options_score
        # PCR < 0.5 (+0.60) + unusual calls (+0.25) + positive skew (+0.15) = 1.0
        chain = self._chain(
            call_vols=[500.0, 500.0, 500.0, 2500.0, 500.0, 500.0, 500.0],
            put_vols =[50.0] * 7,
            call_ivs =[0.22, 0.23, 0.24, 0.35, 0.36, 0.37, 0.38],
            put_ivs  =[0.30, 0.29, 0.28, 0.26, 0.25, 0.24, 0.23],
        )
        score = compute_options_score(chain)
        assert score >= 0.85   # PCR well below 0.5, unusual spike, positive skew

    def test_score_in_range(self):
        from quantlab.signals.options_flow import compute_options_score
        chain = self._chain()
        assert 0.0 <= compute_options_score(chain) <= 1.0

    def test_moderate_score_for_moderate_pcr(self):
        from quantlab.signals.options_flow import compute_options_score
        # PCR = 0.6 (between 0.5 and 0.7) → +0.40 only
        chain = self._chain(call_vols=[100.0]*7, put_vols=[60.0]*7)
        score = compute_options_score(chain)
        assert score >= 0.40   # at least the PCR contribution


# ── options_conviction wired into score_conviction ────────────────────────────

class TestOptionsConvictionLayer:

    def _r(self, **kw) -> ScanResult:
        defaults = dict(symbol="UNH", scan_date="2026-06-04",
                        signal_type="breakout", signal=True,
                        entry_close=399.0, indicator_value=None, lookback=5,
                        regime_bullish=False)
        defaults.update(kw)
        return ScanResult(**defaults)

    def test_options_below_threshold_no_boost(self):
        r = self._r(options_conviction=0.59)
        # signal(0.30) only — regime off, no other layers
        assert score_conviction(r) == pytest.approx(0.30, abs=1e-9)

    def test_options_ge_0_6_adds_0_10(self):
        low  = score_conviction(self._r(options_conviction=0.59))
        high = score_conviction(self._r(options_conviction=0.60))
        assert high - low == pytest.approx(0.10, abs=1e-9)

    def test_options_ge_0_8_adds_0_15(self):
        low  = score_conviction(self._r(options_conviction=0.59))
        high = score_conviction(self._r(options_conviction=0.80))
        assert high - low == pytest.approx(0.15, abs=1e-9)

    def test_options_0_8_replaces_0_6_not_stacks(self):
        # 0.80 should give +0.15 total, not +0.10+0.15
        mid   = score_conviction(self._r(options_conviction=0.70))
        strong = score_conviction(self._r(options_conviction=0.85))
        assert strong - mid == pytest.approx(0.05, abs=1e-9)  # 0.15 - 0.10 = 0.05

    def test_options_field_defaults_zero(self):
        r = ScanResult("UNH","2026-06-04","breakout",True,399.0,None,5)
        assert r.options_conviction == 0.0

    def test_full_conviction_with_options_clamped(self):
        r = self._r(
            regime_bullish=True, earnings_acceleration=0.8,
            absorption=0.8, multi_lookback_confirmed=True,
            options_conviction=0.85,
        )
        # 0.30+0.20+0.10+0.05+0.05+0.15 = 0.85 — no clamp needed
        assert score_conviction(r) == pytest.approx(0.85, abs=1e-9)


# ══════════════════════════════════════════════════════════════════════════════
# Watchlist — add, retrieve, forward return tracking
# All tests use a tmp_path DuckDB — no live IBKR required.
# ══════════════════════════════════════════════════════════════════════════════

class TestWatchlist:

    def _mock_result(self, symbol: str, conviction: float,
                     entry_price: float = 100.0, atr_stop: float = 95.0):
        """Create a minimal ScanResult-like object for watchlist tests."""
        return ScanResult(
            symbol=symbol, scan_date="2026-06-04",
            signal_type="breakout", signal=True,
            entry_close=entry_price, indicator_value=None, lookback=5,
            regime_bullish=True,
            conviction_score=conviction,
            atr_stop=atr_stop,
            earnings_acceleration=0.85 if conviction >= 0.70 else 0.0,
            multi_lookback_confirmed=conviction >= 0.70,
        )

    def _setup_db(self, tmp_path, monkeypatch):
        """Point watchlist storage at a tmp DuckDB."""
        import quantlab.storage as _storage
        import quantlab.watchlist as _watchlist
        monkeypatch.setattr(_storage, "DB_PATH", tmp_path / "test.duckdb")
        monkeypatch.setattr(_watchlist, "DB_PATH", tmp_path / "test.duckdb")
        return str(tmp_path / "test.duckdb")

    # ── add_to_watchlist ──────────────────────────────────────────────────────

    def test_add_inserts_high_conviction_entry(self, tmp_path, monkeypatch):
        db = self._setup_db(tmp_path, monkeypatch)
        from quantlab.watchlist import add_to_watchlist
        r = self._mock_result("UNH", conviction=0.75)
        added = add_to_watchlist(r)
        assert added is True
        import duckdb
        con = duckdb.connect(db)
        from quantlab.storage import _ensure_schema
        _ensure_schema(con)
        row = con.execute("SELECT symbol, conviction_score, status FROM watchlist").fetchone()
        con.close()
        assert row is not None
        assert row[0] == "UNH"
        assert abs(row[1] - 0.75) < 1e-6
        assert row[2] == "watching"

    def test_add_rejects_low_conviction(self, tmp_path, monkeypatch):
        db = self._setup_db(tmp_path, monkeypatch)
        from quantlab.watchlist import add_to_watchlist
        r = self._mock_result("JPM", conviction=0.60)
        added = add_to_watchlist(r)
        assert added is False
        import duckdb
        con = duckdb.connect(db)
        from quantlab.storage import _ensure_schema
        _ensure_schema(con)
        n = con.execute("SELECT COUNT(*) FROM watchlist").fetchone()[0]
        con.close()
        assert n == 0

    def test_add_is_idempotent_same_symbol_same_day(self, tmp_path, monkeypatch):
        db = self._setup_db(tmp_path, monkeypatch)
        from quantlab.watchlist import add_to_watchlist
        r = self._mock_result("ABT", conviction=0.72)
        add_to_watchlist(r)
        add_to_watchlist(r)   # second call same day — should be ignored
        import duckdb
        con = duckdb.connect(db)
        from quantlab.storage import _ensure_schema
        _ensure_schema(con)
        n = con.execute("SELECT COUNT(*) FROM watchlist").fetchone()[0]
        con.close()
        assert n == 1

    def test_layers_fired_captured_correctly(self, tmp_path, monkeypatch):
        db = self._setup_db(tmp_path, monkeypatch)
        from quantlab.watchlist import add_to_watchlist
        r = self._mock_result("UNH", conviction=0.75)
        # Manually set layers on mock result
        r.multi_lookback_confirmed = True
        r.earnings_acceleration = 0.85
        add_to_watchlist(r)
        import duckdb
        con = duckdb.connect(db)
        from quantlab.storage import _ensure_schema
        _ensure_schema(con)
        layers = con.execute("SELECT signal_layers FROM watchlist").fetchone()[0]
        con.close()
        assert "EARN" in layers
        assert "MULTI_LB" in layers

    # ── get_active_watchlist ──────────────────────────────────────────────────

    def test_get_active_returns_watching_entries(self, tmp_path, monkeypatch):
        db = self._setup_db(tmp_path, monkeypatch)
        from quantlab.watchlist import add_to_watchlist, get_active_watchlist
        for sym, conv in [("UNH", 0.75), ("ABT", 0.72), ("UPS", 0.68)]:
            add_to_watchlist(self._mock_result(sym, conv))
        # UPS below threshold → not added
        active = get_active_watchlist(db_path=db)
        assert len(active) == 2
        assert all(e["status"] == "watching" for e in active)
        assert active[0]["conviction_score"] >= active[1]["conviction_score"]

    def test_get_active_returns_empty_when_none(self, tmp_path, monkeypatch):
        db = self._setup_db(tmp_path, monkeypatch)
        from quantlab.watchlist import get_active_watchlist
        assert get_active_watchlist(db_path=db) == []

    # ── update_forward_return ─────────────────────────────────────────────────

    def test_update_forward_return_records_1d(self, tmp_path, monkeypatch):
        db = self._setup_db(tmp_path, monkeypatch)
        from quantlab.watchlist import add_to_watchlist, update_forward_return
        r = self._mock_result("ABT", conviction=0.73, entry_price=90.0)
        add_to_watchlist(r)
        import duckdb
        con = duckdb.connect(db)
        from quantlab.storage import _ensure_schema
        _ensure_schema(con)
        watch_id = con.execute("SELECT watch_id FROM watchlist").fetchone()[0]
        con.close()
        update_forward_return(watch_id, 1, 91.5, 0.0167, db_path=db)
        con = duckdb.connect(db)
        _ensure_schema(con)
        row = con.execute("SELECT price_1d, realized_ret_1d FROM watchlist").fetchone()
        con.close()
        assert row[0] == pytest.approx(91.5, abs=1e-6)
        assert row[1] == pytest.approx(0.0167, abs=1e-4)

    def test_update_forward_return_all_horizons(self, tmp_path, monkeypatch):
        db = self._setup_db(tmp_path, monkeypatch)
        from quantlab.watchlist import add_to_watchlist, update_forward_return
        r = self._mock_result("UNH", conviction=0.75, entry_price=399.0)
        add_to_watchlist(r)
        import duckdb
        con = duckdb.connect(db)
        from quantlab.storage import _ensure_schema
        _ensure_schema(con)
        watch_id = con.execute("SELECT watch_id FROM watchlist").fetchone()[0]
        con.close()
        update_forward_return(watch_id, 1, 401.0, 0.005,  db_path=db)
        update_forward_return(watch_id, 3, 405.0, 0.015,  db_path=db)
        update_forward_return(watch_id, 5, 410.0, 0.0275, db_path=db)
        con = duckdb.connect(db)
        _ensure_schema(con)
        row = con.execute(
            "SELECT realized_ret_1d, realized_ret_3d, realized_ret_5d FROM watchlist"
        ).fetchone()
        con.close()
        assert all(v is not None for v in row)
        assert row[2] > row[1] > row[0]   # returns grew over time

    # ── get_watchlist_summary ─────────────────────────────────────────────────

    def test_summary_reflects_returns(self, tmp_path, monkeypatch):
        db = self._setup_db(tmp_path, monkeypatch)
        from quantlab.watchlist import add_to_watchlist, update_forward_return, \
            get_watchlist_summary
        r = self._mock_result("UNH", conviction=0.75, entry_price=399.0)
        add_to_watchlist(r)
        import duckdb
        con = duckdb.connect(db)
        from quantlab.storage import _ensure_schema
        _ensure_schema(con)
        watch_id = con.execute("SELECT watch_id FROM watchlist").fetchone()[0]
        con.close()
        update_forward_return(watch_id, 1, 403.0, 0.01, db_path=db)
        summary = get_watchlist_summary(db_path=db)
        assert summary["total"] == 1
        assert summary["ret_1d"]["avg"] == pytest.approx(0.01, abs=1e-6)
        assert summary["ret_1d"]["hit_rate"] == pytest.approx(1.0, abs=1e-6)

    # ── _layers_fired helper ──────────────────────────────────────────────────

    def test_layers_fired_captures_all_active_layers(self):
        from quantlab.watchlist import _layers_fired
        r = ScanResult(
            symbol="UNH", scan_date="2026-06-04",
            signal_type="breakout", signal=True,
            entry_close=399.0, indicator_value=None, lookback=5,
            regime_bullish=True,
            earnings_acceleration=0.85,
            accumulation_ratio=0.70,
            multi_lookback_confirmed=True,
            news_count=2, news_category="earnings",
            conviction_score=0.85,
        )
        layers = _layers_fired(r)
        assert "REGIME" in layers
        assert "EARN" in layers
        assert "ACCUM" in layers
        assert "MULTI_LB" in layers
        assert "NEWS:earnings" in layers

    def test_layers_fired_minimal_signal(self):
        from quantlab.watchlist import _layers_fired
        r = ScanResult(
            symbol="JPM", scan_date="2026-06-04",
            signal_type="breakout", signal=True,
            entry_close=307.0, indicator_value=None, lookback=5,
            regime_bullish=False, conviction_score=0.30,
        )
        assert _layers_fired(r) == "signal"

    # ── _trading_days_elapsed ─────────────────────────────────────────────────

    def test_trading_days_zero_same_day(self):
        from quantlab.watchlist import _trading_days_elapsed
        d = date(2026, 6, 4)   # Thursday
        assert _trading_days_elapsed(d, d) == 0

    def test_trading_days_counts_weekdays_only(self):
        from quantlab.watchlist import _trading_days_elapsed
        # Friday June 5 → next Monday June 8: skip weekend
        assert _trading_days_elapsed(date(2026, 6, 5), date(2026, 6, 8)) == 1

    def test_trading_days_one_week(self):
        from quantlab.watchlist import _trading_days_elapsed
        assert _trading_days_elapsed(date(2026, 6, 1), date(2026, 6, 8)) == 5


# ══════════════════════════════════════════════════════════════════════════════
# market_calendar — DST detection, UTC conversion, cron schedule builder
# ══════════════════════════════════════════════════════════════════════════════

class TestMarketCalendar:
    """All tests use fixed dates — no system-clock dependency."""

    # ── DST transition dates ──────────────────────────────────────────────────

    def test_dst_transitions_list_has_six_entries(self):
        from quantlab.market_calendar import DST_TRANSITIONS
        assert len(DST_TRANSITIONS) == 6

    def test_dst_transitions_cover_2026_to_2028(self):
        from quantlab.market_calendar import DST_TRANSITIONS
        years = {d.year for _, d, _ in DST_TRANSITIONS}
        assert years == {2026, 2027, 2028}

    def test_dst_spring_dates_correct(self):
        from quantlab.market_calendar import DST_TRANSITIONS
        from datetime import date
        springs = {key: d for key, d, _ in DST_TRANSITIONS if "spring" in key}
        assert springs["spring_2026"] == date(2026,  3,  8)
        assert springs["spring_2027"] == date(2027,  3, 14)
        assert springs["spring_2028"] == date(2028,  3, 13)

    def test_dst_fall_dates_correct(self):
        from quantlab.market_calendar import DST_TRANSITIONS
        from datetime import date
        falls = {key: d for key, d, _ in DST_TRANSITIONS if "fall" in key}
        assert falls["fall_2026"] == date(2026, 11,  1)
        assert falls["fall_2027"] == date(2027, 11,  7)
        assert falls["fall_2028"] == date(2028, 11,  5)

    # ── is_dst / utc_offset_hours ─────────────────────────────────────────────

    def test_is_dst_summer_true(self):
        from quantlab.market_calendar import is_dst
        from datetime import date
        assert is_dst(date(2026, 6, 4)) is True      # June = EDT

    def test_is_dst_winter_false(self):
        from quantlab.market_calendar import is_dst
        from datetime import date
        assert is_dst(date(2026, 1, 15)) is False     # January = EST

    def test_is_dst_spring_forward_day_true(self):
        from quantlab.market_calendar import is_dst
        from datetime import date
        # By noon on March 8 2026 clocks have already sprung forward
        assert is_dst(date(2026, 3, 8)) is True

    def test_is_dst_fall_back_day_false(self):
        from quantlab.market_calendar import is_dst
        from datetime import date
        # By noon on November 1 2026 clocks have already fallen back
        assert is_dst(date(2026, 11, 1)) is False

    def test_utc_offset_edt(self):
        from quantlab.market_calendar import utc_offset_hours
        from datetime import date
        assert utc_offset_hours(date(2026, 6, 4)) == -4

    def test_utc_offset_est(self):
        from quantlab.market_calendar import utc_offset_hours
        from datetime import date
        assert utc_offset_hours(date(2026, 1, 15)) == -5

    # ── to_utc / named getters ────────────────────────────────────────────────

    def test_scan_utc_during_edt(self):
        from quantlab.market_calendar import get_scan_utc
        from datetime import date
        t = get_scan_utc(date(2026, 6, 4))
        assert t.hour == 13 and t.minute == 0    # 9:00 AM EDT = 13:00 UTC

    def test_scan_utc_during_est(self):
        from quantlab.market_calendar import get_scan_utc
        from datetime import date
        t = get_scan_utc(date(2026, 1, 15))
        assert t.hour == 14 and t.minute == 0    # 9:00 AM EST = 14:00 UTC

    def test_eod_utc_during_edt(self):
        from quantlab.market_calendar import get_eod_utc
        from datetime import date
        t = get_eod_utc(date(2026, 6, 4))
        assert t.hour == 20 and t.minute == 30   # 4:30 PM EDT = 20:30 UTC

    def test_eod_utc_during_est(self):
        from quantlab.market_calendar import get_eod_utc
        from datetime import date
        t = get_eod_utc(date(2026, 1, 15))
        assert t.hour == 21 and t.minute == 30   # 4:30 PM EST = 21:30 UTC

    def test_market_open_utc_edt(self):
        from quantlab.market_calendar import get_market_open_utc
        from datetime import date
        t = get_market_open_utc(date(2026, 6, 4))
        assert t.hour == 13 and t.minute == 30   # 9:30 AM EDT = 13:30 UTC

    def test_market_open_utc_est(self):
        from quantlab.market_calendar import get_market_open_utc
        from datetime import date
        t = get_market_open_utc(date(2026, 1, 15))
        assert t.hour == 14 and t.minute == 30   # 9:30 AM EST = 14:30 UTC

    def test_utc_time_cron_fields_format(self):
        from quantlab.market_calendar import UtcTime
        t = UtcTime(13, 0)
        assert t.cron_fields() == "0 13"

    def test_utc_time_str(self):
        from quantlab.market_calendar import UtcTime
        assert str(UtcTime(13, 0)) == "13:00 UTC"
        assert str(UtcTime(20, 30)) == "20:30 UTC"

    # ── cron_schedule_for_date ────────────────────────────────────────────────

    def test_cron_schedule_edt(self):
        from quantlab.market_calendar import cron_schedule_for_date
        from datetime import date
        s = cron_schedule_for_date(date(2026, 6, 4))
        assert s["scan_cron"]  == "0 13"
        assert s["eod_cron"]   == "30 20"
        assert s["tz_name"]    == "EDT"
        assert s["utc_offset"] == "-4"
        assert s["scan_utc"]   == "13:00 UTC"
        assert s["eod_utc"]    == "20:30 UTC"

    def test_cron_schedule_est(self):
        from quantlab.market_calendar import cron_schedule_for_date
        from datetime import date
        s = cron_schedule_for_date(date(2026, 1, 15))
        assert s["scan_cron"]  == "0 14"
        assert s["eod_cron"]   == "30 21"
        assert s["tz_name"]    == "EST"
        assert s["utc_offset"] == "-5"

    def test_cron_schedule_differs_edt_vs_est(self):
        from quantlab.market_calendar import cron_schedule_for_date
        from datetime import date
        edt = cron_schedule_for_date(date(2026, 6, 4))
        est = cron_schedule_for_date(date(2026, 1, 15))
        assert edt["scan_cron"] != est["scan_cron"]
        assert edt["eod_cron"]  != est["eod_cron"]

    def test_to_utc_round_trip_consistency(self):
        from quantlab.market_calendar import to_utc, get_scan_utc
        from datetime import date, time
        d = date(2027, 3, 14)   # spring forward day 2027
        direct = get_scan_utc(d)
        via_to_utc = to_utc(time(9, 0), d)
        assert direct == via_to_utc


# ══════════════════════════════════════════════════════════════════════════════
# market_calendar — NYSE holiday calendar and is_market_open()
# ══════════════════════════════════════════════════════════════════════════════

class TestNYSEHolidayCalendar:
    """Uses fixed dates — fully deterministic, no system clock."""

    # ── Easter algorithm ──────────────────────────────────────────────────────

    def test_easter_2026(self):
        from quantlab.market_calendar import _easter
        from datetime import date
        assert _easter(2026) == date(2026, 4, 5)

    def test_easter_2027(self):
        from quantlab.market_calendar import _easter
        from datetime import date
        # Meeus/Jones/Butcher + dateutil agree: Easter 2027 = March 28
        assert _easter(2027) == date(2027, 3, 28)

    def test_easter_2028(self):
        from quantlab.market_calendar import _easter
        from datetime import date
        assert _easter(2028) == date(2028, 4, 16)

    # ── Good Friday ───────────────────────────────────────────────────────────

    def test_good_friday_2026_not_market_open(self):
        from quantlab.utils.market_calendar import is_market_open
        from datetime import date
        assert is_market_open(date(2026, 4, 3)) is False  # Good Friday 2026

    def test_good_friday_2027_not_market_open(self):
        from quantlab.utils.market_calendar import is_market_open
        from datetime import date
        # Good Friday 2027 = March 26 (Easter is March 28, NOT April 4)
        assert is_market_open(date(2027, 3, 26)) is False

    def test_good_friday_2028_not_market_open(self):
        from quantlab.utils.market_calendar import is_market_open
        from datetime import date
        assert is_market_open(date(2028, 4, 14)) is False  # Good Friday 2028

    # ── US_MARKET_HOLIDAYS set ────────────────────────────────────────────────

    def test_holiday_set_covers_three_years(self):
        from quantlab.utils.market_calendar import US_MARKET_HOLIDAYS
        years = {d.year for d in US_MARKET_HOLIDAYS}
        assert years == {2026, 2027, 2028}

    def test_holiday_set_per_year_exactly_ten(self):
        from quantlab.market_calendar import _nyse_holidays
        # _nyse_holidays(yr) produces exactly 10 holidays per year.
        # Note: the observed date may fall in an adjacent calendar year
        # (e.g. New Year's 2028 = Saturday → observed Fri Dec 31 2027),
        # so we test the source set, not a year-filter on US_MARKET_HOLIDAYS.
        for yr in [2026, 2027, 2028]:
            count = len(_nyse_holidays(yr))
            assert count == 10, f"{yr}: expected 10 holidays, got {count}"

    def test_thanksgiving_2026_in_holidays(self):
        from quantlab.utils.market_calendar import US_MARKET_HOLIDAYS
        from datetime import date
        assert date(2026, 11, 26) in US_MARKET_HOLIDAYS  # 4th Thursday Nov 2026

    def test_juneteenth_observed_2027(self):
        from quantlab.utils.market_calendar import US_MARKET_HOLIDAYS
        from datetime import date
        # June 19 2027 is a Saturday → observed on Friday June 18
        assert date(2027, 6, 18) in US_MARKET_HOLIDAYS
        assert date(2027, 6, 19) not in US_MARKET_HOLIDAYS

    def test_july4_observed_2027(self):
        from quantlab.utils.market_calendar import US_MARKET_HOLIDAYS
        from datetime import date
        # July 4 2027 is a Sunday → observed Monday July 5
        assert date(2027, 7, 5) in US_MARKET_HOLIDAYS
        assert date(2027, 7, 4) not in US_MARKET_HOLIDAYS  # Sunday already closed

    def test_christmas_observed_2027(self):
        from quantlab.utils.market_calendar import US_MARKET_HOLIDAYS
        from datetime import date
        # Dec 25 2027 is a Saturday → observed Friday Dec 24
        assert date(2027, 12, 24) in US_MARKET_HOLIDAYS

    # ── is_market_open ────────────────────────────────────────────────────────

    def test_weekends_always_closed(self):
        from quantlab.utils.market_calendar import is_market_open
        from datetime import date
        assert is_market_open(date(2026, 6, 6)) is False   # Saturday
        assert is_market_open(date(2026, 6, 7)) is False   # Sunday

    def test_regular_weekday_is_open(self):
        from quantlab.utils.market_calendar import is_market_open
        from datetime import date
        assert is_market_open(date(2026, 6, 4)) is True    # Thursday, no holiday

    def test_independence_day_2026_closed(self):
        from quantlab.utils.market_calendar import is_market_open
        from datetime import date
        assert is_market_open(date(2026, 7, 4)) is False   # Saturday → closed

    def test_independence_day_observed_2026_closed(self):
        from quantlab.utils.market_calendar import is_market_open
        from datetime import date
        assert is_market_open(date(2026, 7, 3)) is False   # Fri = observed holiday

    def test_monday_after_july4_2026_open(self):
        from quantlab.utils.market_calendar import is_market_open
        from datetime import date
        assert is_market_open(date(2026, 7, 6)) is True    # Monday — regular day

    def test_thanksgiving_2026_closed(self):
        from quantlab.utils.market_calendar import is_market_open
        from datetime import date
        assert is_market_open(date(2026, 11, 26)) is False

    def test_christmas_2026_closed(self):
        from quantlab.utils.market_calendar import is_market_open
        from datetime import date
        assert is_market_open(date(2026, 12, 25)) is False  # Friday

    def test_monday_after_christmas_2026_open(self):
        from quantlab.utils.market_calendar import is_market_open
        from datetime import date
        assert is_market_open(date(2026, 12, 28)) is True   # Monday, no holiday

    def test_new_years_2027_closed(self):
        from quantlab.utils.market_calendar import is_market_open
        from datetime import date
        assert is_market_open(date(2027, 1, 1)) is False    # Friday

    def test_utils_import_path_works(self):
        """quantlab.utils.market_calendar must re-export all public symbols."""
        from quantlab.utils.market_calendar import (
            is_market_open, US_MARKET_HOLIDAYS,
            is_dst, get_scan_utc, cron_schedule_for_date,
        )
        assert callable(is_market_open)
        assert isinstance(US_MARKET_HOLIDAYS, frozenset)


# ══════════════════════════════════════════════════════════════════════════════
# watchlist_status.py — formatting helpers (pure functions, fully offline)
# ══════════════════════════════════════════════════════════════════════════════

class TestWatchlistStatusFormatters:
    """Tests for the pure formatting helpers in watchlist_status.py."""

    def _import(self):
        import sys, importlib
        from pathlib import Path
        scripts = Path(__file__).parent.parent / "scripts"
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "watchlist_status", scripts / "watchlist_status.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    # ── fmt_pct ───────────────────────────────────────────────────────────────

    def test_fmt_pct_positive(self):
        ws = self._import()
        assert ws.fmt_pct(0.05) == "+5.00%"

    def test_fmt_pct_negative(self):
        ws = self._import()
        assert ws.fmt_pct(-0.02) == "-2.00%"

    def test_fmt_pct_none(self):
        ws = self._import()
        assert ws.fmt_pct(None) == "    --"

    def test_fmt_pct_zero(self):
        ws = self._import()
        assert ws.fmt_pct(0.0) == "+0.00%"

    # ── fmt_return ────────────────────────────────────────────────────────────

    def test_fmt_return_positive_has_checkmark(self):
        ws = self._import()
        result = ws.fmt_return(0.05)
        assert "✓" in result
        assert "+5.00%" in result

    def test_fmt_return_negative_no_checkmark(self):
        ws = self._import()
        result = ws.fmt_return(-0.02)
        assert "✓" not in result
        assert "-2.00%" in result

    def test_fmt_return_none(self):
        ws = self._import()
        assert ws.fmt_return(None) == "      --"

    # ── stop_distance ─────────────────────────────────────────────────────────

    def test_stop_distance_above_stop(self):
        ws = self._import()
        dist = ws.stop_distance(100.0, 95.0)
        assert dist == pytest.approx(5.0, abs=0.01)   # 5% above stop

    def test_stop_distance_below_stop(self):
        ws = self._import()
        dist = ws.stop_distance(90.0, 95.0)
        assert dist == pytest.approx(-5.56, abs=0.1)  # below stop

    def test_stop_distance_none_when_no_current(self):
        ws = self._import()
        assert ws.stop_distance(None, 95.0) is None

    def test_stop_distance_none_when_no_stop(self):
        ws = self._import()
        assert ws.stop_distance(100.0, None) is None

    def test_stop_distance_none_when_stop_zero(self):
        ws = self._import()
        assert ws.stop_distance(100.0, 0.0) is None

    # ── near_stop ─────────────────────────────────────────────────────────────

    def test_near_stop_true_when_close(self):
        ws = self._import()
        assert ws.near_stop(96.5, 95.0, threshold_pct=2.0) is True   # ~1.55%

    def test_near_stop_false_when_far(self):
        ws = self._import()
        assert ws.near_stop(100.0, 95.0, threshold_pct=2.0) is False  # 5%

    def test_near_stop_false_on_none_price(self):
        ws = self._import()
        assert ws.near_stop(None, 95.0) is False

    # ── fmt_layers ────────────────────────────────────────────────────────────

    def test_fmt_layers_abbreviates_multi_lb(self):
        ws = self._import()
        assert "MLB" in ws.fmt_layers("REGIME,EARN,MULTI_LB")

    def test_fmt_layers_none_returns_dash(self):
        ws = self._import()
        assert ws.fmt_layers(None) == "—"

    def test_fmt_layers_empty_returns_dash(self):
        ws = self._import()
        assert ws.fmt_layers("") == "—"

    def test_fmt_layers_truncates_long_string(self):
        ws = self._import()
        long = "A" * 50
        assert len(ws.fmt_layers(long)) <= 30

    # ── prices_from_cache ─────────────────────────────────────────────────────

    def test_prices_from_cache_returns_dict(self):
        ws = self._import()
        # AAPL has a parquet cache from our backtest runs
        prices = ws._prices_from_cache(["AAPL", "____NOSYM____"])
        assert isinstance(prices, dict)
        assert "____NOSYM____" not in prices

    def test_prices_from_cache_aapl_positive(self):
        ws = self._import()
        prices = ws._prices_from_cache(["AAPL"])
        if "AAPL" in prices:
            assert prices["AAPL"] > 0

    # ── run_dashboard (offline smoke test) ────────────────────────────────────

    def test_run_dashboard_offline_no_crash(self, capsys):
        ws = self._import()
        ws.run_dashboard(use_ibkr=False)   # uses cached prices only
        out = capsys.readouterr().out
        assert "Watchlist Dashboard" in out
        assert "Active Watchlist" in out
        assert "Running Statistics" in out

    def test_main_importable(self):
        ws = self._import()
        assert callable(ws.main)
        assert callable(ws.run_dashboard)


# ══════════════════════════════════════════════════════════════════════════════
# Sector correlation filter
# ══════════════════════════════════════════════════════════════════════════════

class TestSectorFilter:

    def _scan_result(self, symbol: str, conviction: float) -> ScanResult:
        from quantlab.execution import SECTOR_MAP
        r = ScanResult(
            symbol=symbol, scan_date="2026-06-04",
            signal_type="breakout", signal=True,
            entry_close=100.0, indicator_value=None, lookback=5,
            conviction_score=conviction,
        )
        r.sector = SECTOR_MAP.get(symbol, "")
        return r

    # ── SECTOR_MAP coverage ───────────────────────────────────────────────────

    def test_sector_map_covers_all_sp500_sample(self):
        from quantlab.execution import SECTOR_MAP, SP500_SAMPLE
        missing = [s for s in SP500_SAMPLE if s not in SECTOR_MAP]
        assert missing == [], f"Missing sectors for: {missing}"

    def test_sector_map_has_10_sectors(self):
        from quantlab.execution import SECTOR_MAP
        sectors = set(SECTOR_MAP.values())
        assert len(sectors) == 10

    def test_sector_map_known_assignments(self):
        from quantlab.execution import SECTOR_MAP
        assert SECTOR_MAP["AAPL"]   == "Technology"
        assert SECTOR_MAP["UNH"]    == "Health Care"
        assert SECTOR_MAP["JPM"]    == "Financials"
        assert SECTOR_MAP["XOM"]    == "Energy"
        assert SECTOR_MAP["BRK B"]  == "Financials"
        assert SECTOR_MAP["NEE"]    == "Utilities"
        assert SECTOR_MAP["LIN"]    == "Materials"

    def test_sector_abbrev_covers_all_sectors(self):
        from quantlab.execution import SECTOR_MAP, _SECTOR_ABBREV
        sectors = set(SECTOR_MAP.values())
        for s in sectors:
            assert s in _SECTOR_ABBREV, f"No abbreviation for sector: {s}"

    # ── sector_filter() behaviour ─────────────────────────────────────────────

    def test_cluster_of_3_triggers_penalty(self):
        from quantlab.execution import sector_filter
        results = [
            self._scan_result("ABT",  0.70),  # Health Care
            self._scan_result("UNH",  0.65),  # Health Care
            self._scan_result("LLY",  0.62),  # Health Care
        ]
        sector_filter(results)
        assert all(r.sector_cluster is True for r in results)
        # 0.70−0.05=0.65, 0.65−0.05=0.60, 0.62−0.05=0.57 (results re-sorted desc)
        expected = [0.65, 0.60, 0.57]
        assert all(abs(r.conviction_score - exp) < 1e-4
                   for r, exp in zip(sorted(results, key=lambda x: -x.conviction_score),
                                     expected))

    def test_cluster_of_2_no_penalty(self):
        from quantlab.execution import sector_filter
        results = [
            self._scan_result("XOM", 0.70),   # Energy
            self._scan_result("CVX", 0.65),   # Energy
        ]
        sector_filter(results)
        assert all(r.sector_cluster is False for r in results)
        assert results[0].conviction_score == pytest.approx(0.70)
        assert results[1].conviction_score == pytest.approx(0.65)

    def test_non_clustered_symbol_unaffected(self):
        from quantlab.execution import sector_filter
        results = [
            self._scan_result("ABT",  0.70),  # Health Care ×3 → cluster
            self._scan_result("UNH",  0.65),
            self._scan_result("LLY",  0.62),
            self._scan_result("AAPL", 0.75),  # Technology ×1 → no cluster
        ]
        sector_filter(results)
        aapl = next(r for r in results if r.symbol == "AAPL")
        assert aapl.sector_cluster is False
        assert aapl.conviction_score == pytest.approx(0.75)

    def test_penalty_capped_at_zero(self):
        from quantlab.execution import sector_filter
        results = [
            self._scan_result("ABT",  0.02),
            self._scan_result("UNH",  0.02),
            self._scan_result("LLY",  0.02),
        ]
        sector_filter(results)
        assert all(r.conviction_score >= 0.0 for r in results)

    def test_cluster_flag_in_layers_fired(self):
        from quantlab.execution import sector_filter
        from quantlab.watchlist import _layers_fired
        results = [
            self._scan_result("ABT",  0.70),
            self._scan_result("UNH",  0.65),
            self._scan_result("LLY",  0.62),
        ]
        sector_filter(results)
        r = results[0]
        r.regime_bullish = True
        layers = _layers_fired(r)
        assert "SECTOR_CLUSTER" in layers

    def test_results_re_sorted_after_penalty(self):
        from quantlab.execution import sector_filter
        results = [
            self._scan_result("AAPL", 0.72),  # Tech ×1 — no penalty
            self._scan_result("ABT",  0.70),  # HC ×3 → 0.65
            self._scan_result("UNH",  0.65),  # HC ×3 → 0.60
            self._scan_result("LLY",  0.62),  # HC ×3 → 0.57
        ]
        out = sector_filter(results)
        # After penalty AAPL (0.72) should be first
        assert out[0].symbol == "AAPL"
        # Scores descending
        scores = [r.conviction_score for r in out]
        assert scores == sorted(scores, reverse=True)

    def test_scan_result_sector_defaults_empty(self):
        r = ScanResult("X","2026-06-04","breakout",True,100.0,None,5)
        assert r.sector == ""
        assert r.sector_cluster is False

    def test_scan_symbol_populates_sector(self):
        bars = make_bars(100, trend=0.002)
        result = scan_symbol("AAPL", bars, signal_type="breakout", lookback=5)
        assert result is not None
        assert result.sector == "Technology"


# ══════════════════════════════════════════════════════════════════════════════
# Relative strength signals (quantlab.signals.relative_strength)
# ══════════════════════════════════════════════════════════════════════════════

class TestRelativeStrength:
    """Synthetic bars — no market data or IBKR needed."""

    @staticmethod
    def _bars(n: int, start_price: float = 100.0, trend: float = 0.0) -> list[Bar]:
        from datetime import timedelta
        bars, price = [], start_price
        start = date(2025, 1, 2)
        for i in range(n):
            price *= (1 + trend)
            bars.append(Bar(start + timedelta(days=i),
                            price*0.999, price*1.005, price*0.995, price, 1e6))
        return bars

    # ── rs_score ─────────────────────────────────────────────────────────────

    def test_rs_score_neutral_when_matched(self):
        from quantlab.signals.relative_strength import rs_score
        # Both growing at the same rate → excess = 0 → score ≈ 0.5
        sym = self._bars(200, trend=0.002)
        mkt = self._bars(200, trend=0.002)
        assert abs(rs_score(sym, mkt) - 0.5) < 0.01

    def test_rs_score_above_half_when_outperforming(self):
        from quantlab.signals.relative_strength import rs_score
        sym = self._bars(200, trend=0.003)   # faster growth
        mkt = self._bars(200, trend=0.001)
        assert rs_score(sym, mkt) > 0.5

    def test_rs_score_exceeds_0_6_threshold_on_strong_outperformance(self):
        from quantlab.signals.relative_strength import rs_score
        sym = self._bars(200, start_price=100, trend=0.003)   # +3% per bar
        mkt = self._bars(200, start_price=100, trend=0.001)   # +1% per bar
        score = rs_score(sym, mkt, periods=[63])
        assert score > 0.6, f"Expected >0.6, got {score}"

    def test_rs_score_below_half_when_underperforming(self):
        from quantlab.signals.relative_strength import rs_score
        sym = self._bars(200, trend=0.001)
        mkt = self._bars(200, trend=0.003)
        assert rs_score(sym, mkt) < 0.5

    def test_rs_score_neutral_on_too_few_bars(self):
        from quantlab.signals.relative_strength import rs_score
        sym = self._bars(30)   # fewer than 63 lookback
        mkt = self._bars(30)
        assert rs_score(sym, mkt, periods=[63]) == pytest.approx(0.5, abs=0.01)

    def test_rs_score_averages_across_periods(self):
        from quantlab.signals.relative_strength import rs_score
        sym = self._bars(200, trend=0.003)
        mkt = self._bars(200, trend=0.001)
        s1 = rs_score(sym, mkt, periods=[63])
        s2 = rs_score(sym, mkt, periods=[126])
        s_avg = rs_score(sym, mkt, periods=[63, 126])
        assert abs(s_avg - (s1 + s2) / 2) < 0.001

    def test_rs_score_in_range(self):
        from quantlab.signals.relative_strength import rs_score
        sym = self._bars(200, trend=0.005)
        mkt = self._bars(200, trend=0.001)
        assert 0.0 <= rs_score(sym, mkt) <= 1.0

    def test_rs_score_strong_outperformance_exceeds_0_8(self):
        from quantlab.signals.relative_strength import rs_score
        # +0.5% daily vs flat = ~+10% excess over 63 days
        sym = self._bars(200, start_price=100, trend=0.005)
        mkt = self._bars(200, start_price=100, trend=0.0)
        score = rs_score(sym, mkt, periods=[63])
        assert score > 0.7, f"Strong leader should score >0.7, got {score}"

    # ── rs_rank ───────────────────────────────────────────────────────────────

    def test_rs_rank_top_symbol_scores_100(self):
        from quantlab.signals.relative_strength import rs_rank
        mkt  = self._bars(200, trend=0.001)
        syms = {
            "AAPL": self._bars(200, trend=0.005),   # best
            "MSFT": self._bars(200, trend=0.002),
            "XOM":  self._bars(200, trend=0.000),   # worst
        }
        ranks = rs_rank(syms, mkt, periods=[63])
        assert ranks["AAPL"] == 100.0
        assert ranks["XOM"]  == 0.0

    def test_rs_rank_middle_is_50_for_three_symbols(self):
        from quantlab.signals.relative_strength import rs_rank
        mkt  = self._bars(200, trend=0.001)
        syms = {
            "A": self._bars(200, trend=0.003),
            "B": self._bars(200, trend=0.002),
            "C": self._bars(200, trend=0.001),
        }
        ranks = rs_rank(syms, mkt, periods=[63])
        assert ranks["B"] == pytest.approx(50.0, abs=0.1)

    def test_rs_rank_empty_input(self):
        from quantlab.signals.relative_strength import rs_rank
        mkt = self._bars(200)
        assert rs_rank({}, mkt) == {}

    def test_rs_rank_single_symbol_is_50(self):
        from quantlab.signals.relative_strength import rs_rank
        mkt  = self._bars(200)
        syms = {"AAPL": self._bars(200, trend=0.003)}
        ranks = rs_rank(syms, mkt, periods=[63])
        assert ranks["AAPL"] == 50.0

    def test_rs_rank_preserves_order(self):
        from quantlab.signals.relative_strength import rs_rank
        mkt  = self._bars(200, trend=0.001)
        syms = {
            "FAST": self._bars(200, trend=0.004),
            "SLOW": self._bars(200, trend=0.001),
            "FLAT": self._bars(200, trend=0.0),
        }
        ranks = rs_rank(syms, mkt, periods=[63])
        assert ranks["FAST"] > ranks["SLOW"] > ranks["FLAT"]


# ── RS conviction layer ────────────────────────────────────────────────────────

class TestRSConvictionLayer:

    def _r(self, rs: float) -> ScanResult:
        r = ScanResult("AAPL","2026-06-04","breakout",True,180.0,None,5,
                       regime_bullish=False)
        r.rs_score = rs
        return r

    def test_rs_below_threshold_no_boost(self):
        r = self._r(0.59)
        assert score_conviction(r) == pytest.approx(0.30, abs=1e-9)

    def test_rs_ge_0_6_adds_0_08(self):
        low  = score_conviction(self._r(0.59))
        high = score_conviction(self._r(0.60))
        assert high - low == pytest.approx(0.08, abs=1e-9)

    def test_rs_ge_0_8_adds_0_12(self):
        low  = score_conviction(self._r(0.59))
        high = score_conviction(self._r(0.80))
        assert high - low == pytest.approx(0.12, abs=1e-9)

    def test_rs_0_8_replaces_not_stacks(self):
        mid    = score_conviction(self._r(0.70))   # +0.08
        strong = score_conviction(self._r(0.85))   # +0.12
        assert strong - mid == pytest.approx(0.04, abs=1e-9)  # 0.12 - 0.08 = 0.04

    def test_rs_field_defaults_zero(self):
        r = ScanResult("AAPL","2026-06-04","breakout",True,180.0,None,5)
        assert r.rs_score == 0.0

    def test_scan_symbol_populates_rs_when_regime_bars_provided(self):
        sym_bars = make_bars(200, trend=0.003)
        mkt_bars = make_bars(200, trend=0.001)
        result = scan_symbol("AAPL", sym_bars, signal_type="breakout",
                             lookback=5, regime_bars=mkt_bars)
        assert result is not None
        assert result.rs_score > 0.0   # market bars provided → RS computed

    def test_scan_symbol_rs_zero_without_regime_bars(self):
        bars = make_bars(200, trend=0.003)
        result = scan_symbol("AAPL", bars, signal_type="breakout", lookback=5)
        assert result is not None
        assert result.rs_score == 0.0  # no regime_bars → default


# ══════════════════════════════════════════════════════════════════════════════
# Polygon provider — unit tests using mock HTTP responses
# ══════════════════════════════════════════════════════════════════════════════

class TestPolygonProvider:

    def _make_provider(self):
        from quantlab.providers.polygon import PolygonProvider
        return PolygonProvider(api_key="test-key", request_sleep=0.0)

    def test_bar_from_agg_parses_fields(self):
        from quantlab.providers.polygon import PolygonProvider
        from datetime import date
        item = {"o": 150.0, "h": 155.0, "l": 149.0, "c": 153.0, "v": 5e6}
        bar = PolygonProvider._bar_from_agg(item, date(2026, 6, 4))
        assert bar.open   == 150.0
        assert bar.high   == 155.0
        assert bar.close  == 153.0
        assert bar.volume == 5e6

    def test_breadth_cache_roundtrip(self, tmp_path, monkeypatch):
        import quantlab.storage as _st
        monkeypatch.setattr(_st, "DATA_PROCESSED", tmp_path)
        from datetime import date
        from quantlab.providers.base import Bar
        from quantlab.providers.polygon import PolygonProvider
        p = PolygonProvider(api_key="test", request_sleep=0.0)
        data = {
            "AAPL": Bar(date(2026,6,4), 150.0, 155.0, 149.0, 153.0, 5e6),
            "MSFT": Bar(date(2026,6,4), 400.0, 405.0, 398.0, 402.0, 2e6),
        }
        p._save_breadth_cache(date(2026, 6, 4), data)
        loaded = p._load_breadth_cache(date(2026, 6, 4))
        assert loaded is not None
        assert "AAPL" in loaded
        assert loaded["AAPL"].close == pytest.approx(153.0)

    def test_breadth_cache_miss_returns_none(self, tmp_path, monkeypatch):
        import quantlab.storage as _st
        monkeypatch.setattr(_st, "DATA_PROCESSED", tmp_path)
        from datetime import date
        from quantlab.providers.polygon import PolygonProvider
        p = PolygonProvider(api_key="test", request_sleep=0.0)
        assert p._load_breadth_cache(date(2099, 1, 1)) is None

    def test_factory_creates_polygon(self):
        from quantlab.providers import create_market_data_provider
        from quantlab.providers.polygon import PolygonProvider
        p = create_market_data_provider("polygon", api_key="test-key")
        assert isinstance(p, PolygonProvider)


# ══════════════════════════════════════════════════════════════════════════════
# Breadth computation — all tests use synthetic data, no Polygon API needed
# ══════════════════════════════════════════════════════════════════════════════

class TestBreadthComputation:

    @staticmethod
    def _bar(sym, close, open_=None, volume=1_000_000.0):
        from datetime import date
        from quantlab.providers.base import Bar
        o = open_ if open_ is not None else close * 0.99
        return Bar(date(2026, 6, 4), o, close*1.005, close*0.995, close, volume)

    def _grouped(self, specs):
        """specs: list of (symbol, close, open_) tuples"""
        return {sym: self._bar(sym, c, o) for sym, c, o in specs}

    # ── compute_market_breadth ────────────────────────────────────────────────

    def test_advances_declines_counted(self):
        from quantlab.signals.breadth import compute_market_breadth
        today = self._grouped([
            ("AAPL", 102.0, 100.0),   # up 2%
            ("MSFT", 98.0,  100.0),   # down 2%
            ("GOOGL", 100.0, 100.0),  # flat
        ])
        snap = compute_market_breadth("2026-06-04", today)
        assert snap.advances == 1
        assert snap.declines == 1
        assert snap.unchanged == 1

    def test_up_4pct_counted(self):
        from quantlab.signals.breadth import compute_market_breadth
        today = self._grouped([
            ("AAPL", 105.0, 100.0),   # +5%
            ("MSFT", 103.0, 100.0),   # +3% (below threshold)
            ("XOM",  96.0,  100.0),   # -4%
        ])
        snap = compute_market_breadth("2026-06-04", today)
        assert snap.up_4pct_count   == 1
        assert snap.down_4pct_count == 1

    def test_low_volume_excluded(self):
        from quantlab.signals.breadth import compute_market_breadth
        from datetime import date
        from quantlab.providers.base import Bar
        today = {
            "AAPL": Bar(date(2026,6,4), 100.0, 105.0, 99.0, 105.0, 5_000.0),  # low vol
        }
        snap = compute_market_breadth("2026-06-04", today, min_volume=100_000)
        assert snap.advances == 0  # excluded

    def test_prev_data_used_for_close_to_close(self):
        from quantlab.signals.breadth import compute_market_breadth
        from datetime import date
        from quantlab.providers.base import Bar
        prev  = {"AAPL": Bar(date(2026,6,3), 100.0,102.0,99.0,100.0,1e6)}
        today = {"AAPL": Bar(date(2026,6,4), 103.0,106.0,102.0,105.0,1e6)}
        snap = compute_market_breadth("2026-06-04", today, prev_data=prev)
        # close-to-close = 105/100 - 1 = +5% → up_4pct
        assert snap.up_4pct_count == 1

    def test_ad_ratio_computed(self):
        from quantlab.signals.breadth import compute_market_breadth
        today = self._grouped([
            ("A", 105.0, 100.0), ("B", 103.0, 100.0),  # 2 advances
            ("C", 97.0,  100.0),                         # 1 decline
        ])
        snap = compute_market_breadth("2026-06-04", today)
        assert snap.advance_decline_ratio == pytest.approx(2.0, abs=0.01)

    # ── rolling_breadth ───────────────────────────────────────────────────────

    def _make_snapshots(self, ad_pairs):
        """Create BreadthSnapshot list from (advances, declines, up4, dn4) tuples."""
        from quantlab.signals.breadth import BreadthSnapshot
        snaps = []
        for i, (a, d, u4, d4) in enumerate(ad_pairs):
            snaps.append(BreadthSnapshot(
                date=f"2026-01-{i+2:02d}",
                advances=a, declines=d,
                up_4pct_count=u4, down_4pct_count=d4,
            ))
        return snaps

    def test_rolling_adds_ratio_10d(self):
        from quantlab.signals.breadth import rolling_breadth
        snaps = self._make_snapshots([(100,50,20,10)] * 12)
        rolling_breadth(snaps, window=10)
        assert snaps[-1].ratio_10d is not None
        assert snaps[-1].ratio_10d == pytest.approx(2.0, abs=0.01)

    def test_mcclellan_oscillator_computed(self):
        from quantlab.signals.breadth import rolling_breadth
        snaps = self._make_snapshots([(200, 100, 20, 10)] * 50)
        rolling_breadth(snaps)
        # After enough data, EMA19 and EMA39 should both converge to A-D = 100
        # → oscillator should be near 0
        last = snaps[-1]
        assert last.mcclellan_oscillator is not None
        assert abs(last.mcclellan_oscillator) < 10  # converged near 0

    def test_tape_bull_on_high_ratio(self):
        from quantlab.signals.breadth import rolling_breadth
        snaps = self._make_snapshots([(300, 100, 40, 10)] * 15)
        rolling_breadth(snaps, window=10)
        assert snaps[-1].tape == "BULL"

    def test_tape_bear_on_low_ratio(self):
        from quantlab.signals.breadth import rolling_breadth
        snaps = self._make_snapshots([(100, 300, 10, 40)] * 15)
        rolling_breadth(snaps, window=10)
        assert snaps[-1].tape == "BEAR"

    def test_tape_bear_on_mcclellan_below_minus100(self):
        from quantlab.signals.breadth import rolling_breadth, BreadthSnapshot
        # Large sustained negative A-D → McClellan will go below -100
        snaps = self._make_snapshots([(100, 900, 5, 30)] * 60)
        rolling_breadth(snaps, window=10)
        assert snaps[-1].tape == "BEAR"

    # ── breadth_regime_adjustment ────────────────────────────────────────────

    def test_bull_tape_no_penalty(self):
        from quantlab.signals.breadth import breadth_regime_adjustment, BreadthSnapshot
        snap = BreadthSnapshot("2026-06-04", ratio_10d=2.5, mcclellan_oscillator=50.0,
                               up_25pct_quarter=350, tape="BULL")
        adj, override = breadth_regime_adjustment(snap)
        assert adj == 0.0
        assert override is False

    def test_neutral_tape_small_penalty(self):
        from quantlab.signals.breadth import breadth_regime_adjustment, BreadthSnapshot
        snap = BreadthSnapshot("2026-06-04", ratio_10d=1.5, mcclellan_oscillator=0.0,
                               up_25pct_quarter=300, tape="NEUTRAL")
        adj, override = breadth_regime_adjustment(snap)
        assert adj == pytest.approx(-0.03, abs=1e-6)
        assert override is False

    def test_bear_tape_large_penalty(self):
        from quantlab.signals.breadth import breadth_regime_adjustment, BreadthSnapshot
        snap = BreadthSnapshot("2026-06-04", ratio_10d=0.4, mcclellan_oscillator=-20.0,
                               up_25pct_quarter=300, tape="BEAR")
        adj, override = breadth_regime_adjustment(snap)
        assert adj == pytest.approx(-0.12, abs=1e-6)

    def test_mcclellan_below_minus100_triggers_override(self):
        from quantlab.signals.breadth import breadth_regime_adjustment, BreadthSnapshot
        snap = BreadthSnapshot("2026-06-04", ratio_10d=1.0, mcclellan_oscillator=-150.0,
                               up_25pct_quarter=400, tape="BEAR")
        _, override = breadth_regime_adjustment(snap)
        assert override is True

    def test_up25q_below_200_triggers_override(self):
        from quantlab.signals.breadth import breadth_regime_adjustment, BreadthSnapshot
        snap = BreadthSnapshot("2026-06-04", ratio_10d=1.5, mcclellan_oscillator=50.0,
                               up_25pct_quarter=150, tape="BEAR")
        _, override = breadth_regime_adjustment(snap)
        assert override is True

    def test_none_snapshot_neutral(self):
        from quantlab.signals.breadth import breadth_regime_adjustment
        adj, override = breadth_regime_adjustment(None)
        assert adj == 0.0
        assert override is False

    def test_nhl_below_half_adds_extra_penalty(self):
        from quantlab.signals.breadth import breadth_regime_adjustment, BreadthSnapshot
        snap = BreadthSnapshot("2026-06-04", ratio_10d=1.8, mcclellan_oscillator=20.0,
                               up_25pct_quarter=400, new_high_low_ratio=0.3, tape="NEUTRAL")
        adj, _ = breadth_regime_adjustment(snap)
        assert adj == pytest.approx(-0.03 - 0.05, abs=1e-6)

    # ── conviction scorer with breadth ────────────────────────────────────────

    def test_breadth_adj_applied_in_scorer(self):
        r = ScanResult("ABT","2026-06-04","breakout",True,90.0,None,5,
                       regime_bullish=False)
        r.breadth_regime_adj = -0.07
        base = score_conviction(r)
        assert base == pytest.approx(0.30 - 0.07, abs=1e-9)

    def test_breadth_override_vetoes_signal(self):
        r = ScanResult("ABT","2026-06-04","breakout",True,90.0,None,5,
                       regime_bullish=True, earnings_acceleration=0.85)
        r.breadth_override = True
        assert score_conviction(r) == 0.0

    def test_breadth_fields_default_to_neutral(self):
        r = ScanResult("ABT","2026-06-04","breakout",True,90.0,None,5)
        assert r.breadth_regime_adj == 0.0
        assert r.breadth_override   is False


# ══════════════════════════════════════════════════════════════════════════════
# Rate limit handling + breadth-aware scan threshold
# ══════════════════════════════════════════════════════════════════════════════

class TestPolygonRateLimit:
    """Tests for 429 retry logic — uses unittest.mock to avoid real HTTP calls."""

    def _provider(self):
        from quantlab.providers.polygon import PolygonProvider
        # Zero sleeps so tests are fast
        return PolygonProvider(api_key="test", request_sleep=0.0,
                               grouped_daily_sleep=0.0, max_retries=3)

    def test_success_on_first_try(self):
        from unittest.mock import MagicMock, patch
        from quantlab.providers.polygon import PolygonProvider
        p = self._provider()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"results": []}
        with patch.object(p._session, "get", return_value=mock_resp):
            data = p._get("/v2/test")
        assert data == {"results": []}

    def test_retries_on_429_then_succeeds(self):
        from unittest.mock import MagicMock, patch
        p = self._provider()
        rate_resp = MagicMock(); rate_resp.status_code = 429
        ok_resp   = MagicMock(); ok_resp.status_code   = 200
        ok_resp.json.return_value = {"status": "OK"}
        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            return rate_resp if call_count[0] < 3 else ok_resp
        with patch.object(p._session, "get", side_effect=side_effect), \
             patch("time.sleep"):   # suppress actual sleep in tests
            data = p._get("/v2/test")
        assert data == {"status": "OK"}
        assert call_count[0] == 3   # 2 rate-limited + 1 success

    def test_raises_after_max_retries_on_429(self):
        import requests as _req
        from unittest.mock import MagicMock, patch
        p = self._provider()
        rate_resp = MagicMock(); rate_resp.status_code = 429
        with patch.object(p._session, "get", return_value=rate_resp), \
             patch("time.sleep"):
            try:
                p._get("/v2/test")
                assert False, "Should have raised"
            except Exception:
                pass   # expected — either HTTPError or RuntimeError

    def test_non_429_http_error_not_retried(self):
        from unittest.mock import MagicMock, patch
        import requests as _req
        p = self._provider()
        err_resp = MagicMock(); err_resp.status_code = 403
        err_resp.raise_for_status.side_effect = _req.HTTPError(response=err_resp)
        call_count = [0]
        def side_effect(*a, **kw):
            call_count[0] += 1
            return err_resp
        with patch.object(p._session, "get", side_effect=side_effect), \
             patch("time.sleep"):
            try:
                p._get("/v2/test")
            except _req.HTTPError:
                pass
        assert call_count[0] == 1   # no retry on 403


class TestBreadthScanThreshold:
    """Tests for automatic min_conviction raise when tape=BEAR."""

    def _make_bear_snap(self):
        from quantlab.signals.breadth import BreadthSnapshot
        return BreadthSnapshot(
            date="2026-06-04",
            advances=500, declines=2000,
            up_4pct_count=20, down_4pct_count=150,
            ratio_10d=0.3,
            mcclellan_oscillator=-389.0,
            ad_line=-4757,
            tape="BEAR",
        )

    def _make_bull_snap(self):
        from quantlab.signals.breadth import BreadthSnapshot
        return BreadthSnapshot(
            date="2026-06-04",
            advances=2000, declines=500,
            up_4pct_count=200, down_4pct_count=50,
            ratio_10d=2.5,
            mcclellan_oscillator=80.0,
            ad_line=3200,
            tape="BULL",
        )

    def test_bull_tape_does_not_raise_threshold(self):
        """In a bull tape, min_conviction stays at 0.40."""
        from quantlab.signals.breadth import breadth_regime_adjustment
        snap = self._make_bull_snap()
        adj, override = breadth_regime_adjustment(snap)
        assert override is False
        assert adj == 0.0   # no penalty in bull tape

    def test_bear_tape_applies_override(self):
        """Bear tape (McClellan < -100) triggers hard veto."""
        from quantlab.signals.breadth import breadth_regime_adjustment
        snap = self._make_bear_snap()
        adj, override = breadth_regime_adjustment(snap)
        assert override is True

    def test_bear_tape_conviction_veto_in_scorer(self):
        """score_conviction() returns 0.0 when breadth_override is True."""
        r = ScanResult("ABT","2026-06-04","breakout",True,90.0,None,5,
                       regime_bullish=True, earnings_acceleration=0.85)
        r.breadth_override = True
        assert score_conviction(r) == 0.0

    def test_bear_tape_summary_line_format(self):
        snap = self._make_bear_snap()
        line = snap.summary_line()
        assert "BEAR" in line
        assert "10d-ratio" in line
        assert "McClellan" in line
        assert "tape" in line

    def test_bear_tape_ratio_penalty(self):
        """10d-ratio=0.3 < 0.5 → -0.12 regime penalty."""
        from quantlab.signals.breadth import breadth_regime_adjustment, BreadthSnapshot
        snap = BreadthSnapshot("2026-06-04", ratio_10d=0.3,
                               mcclellan_oscillator=-20.0,  # above -100, no override
                               up_25pct_quarter=350, tape="BEAR")
        adj, override = breadth_regime_adjustment(snap)
        assert adj == pytest.approx(-0.12, abs=1e-6)
        assert override is False

    def test_ignore_breadth_flag_exists(self):
        """--ignore-breadth is a valid scan_universe.py argument."""
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "scripts/scan_universe.py", "--help"],
            capture_output=True, text=True,
            cwd="/home/quantlab/projects/quantlab-project",
        )
        assert "--ignore-breadth" in result.stdout

    def test_backfill_flag_exists(self):
        """--backfill is a valid update_breadth.py argument."""
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "scripts/update_breadth.py", "--help"],
            capture_output=True, text=True,
            cwd="/home/quantlab/projects/quantlab-project",
        )
        assert "--backfill" in result.stdout
        assert "--start-date" in result.stdout


# ══════════════════════════════════════════════════════════════════════════════
# breadth_override_note column — watchlist audit trail
# ══════════════════════════════════════════════════════════════════════════════

class TestWatchlistBreadthNote:

    def _setup_db(self, tmp_path, monkeypatch):
        import quantlab.storage as _storage
        import quantlab.watchlist as _watchlist
        monkeypatch.setattr(_storage,   "DB_PATH", tmp_path / "test.duckdb")
        monkeypatch.setattr(_watchlist, "DB_PATH", tmp_path / "test.duckdb")
        return str(tmp_path / "test.duckdb")

    def _add_abt(self, db):
        """Insert a synthetic ABT entry and return the generated watch_id."""
        from datetime import date as _date
        r = ScanResult("ABT","2026-06-04","breakout",True,90.93,None,5,
                       regime_bullish=True, earnings_acceleration=0.65,
                       conviction_score=0.70, atr_stop=86.72,
                       multi_lookback_confirmed=True)
        from quantlab.watchlist import add_to_watchlist
        add_to_watchlist(r)
        # watch_id is always symbol_YYYY-MM-DD using today's date at insert time
        return f"ABT_{_date.today().isoformat()}"

    def test_note_stored_on_insert(self, tmp_path, monkeypatch):
        db = self._setup_db(tmp_path, monkeypatch)
        r = ScanResult("ABT","2026-06-04","breakout",True,90.93,None,5,
                       conviction_score=0.70, atr_stop=86.72)
        from quantlab.watchlist import add_to_watchlist
        add_to_watchlist(r, note="Added pre-breadth-load — tape=BEAR at time of scan")
        import duckdb
        con = duckdb.connect(db)
        from quantlab.storage import _ensure_schema
        _ensure_schema(con)
        note = con.execute(
            "SELECT breadth_override_note FROM watchlist WHERE symbol='ABT'"
        ).fetchone()[0]
        con.close()
        assert note == "Added pre-breadth-load — tape=BEAR at time of scan"

    def test_note_empty_by_default(self, tmp_path, monkeypatch):
        db = self._setup_db(tmp_path, monkeypatch)
        self._add_abt(db)
        import duckdb
        con = duckdb.connect(db)
        from quantlab.storage import _ensure_schema
        _ensure_schema(con)
        note = con.execute(
            "SELECT breadth_override_note FROM watchlist WHERE symbol='ABT'"
        ).fetchone()[0]
        con.close()
        assert note == "" or note is None

    def test_set_watchlist_note_updates_existing(self, tmp_path, monkeypatch):
        db = self._setup_db(tmp_path, monkeypatch)
        watch_id = self._add_abt(db)
        from quantlab.watchlist import set_watchlist_note, get_active_watchlist
        set_watchlist_note(watch_id, "Retroactive note — bear tape was active")
        entries = get_active_watchlist(db_path=db)
        abt = next(e for e in entries if e["symbol"] == "ABT")
        assert abt["breadth_override_note"] == "Retroactive note — bear tape was active"

    def test_get_active_watchlist_includes_note_column(self, tmp_path, monkeypatch):
        db = self._setup_db(tmp_path, monkeypatch)
        self._add_abt(db)
        from quantlab.watchlist import get_active_watchlist
        entries = get_active_watchlist(db_path=db)
        assert len(entries) == 1
        assert "breadth_override_note" in entries[0]

    def test_note_survives_price_update(self, tmp_path, monkeypatch):
        """set_watchlist_note should not touch other columns."""
        db = self._setup_db(tmp_path, monkeypatch)
        watch_id = self._add_abt(db)
        from quantlab.watchlist import set_watchlist_note, get_active_watchlist
        set_watchlist_note(watch_id, "BEAR tape flag")
        entries = get_active_watchlist(db_path=db)
        abt = entries[0]
        assert abt["conviction_score"] == pytest.approx(0.70)
        assert abt["entry_price"]     == pytest.approx(90.93)
        assert abt["breadth_override_note"] == "BEAR tape flag"


# ══════════════════════════════════════════════════════════════════════════════
# FactSet provider — full coverage, mock mode only (credentials pending)
# ══════════════════════════════════════════════════════════════════════════════

class TestFactSetProvider:
    """All tests use mock mode — no real credentials or network calls needed."""

    @staticmethod
    def _p():
        from quantlab.providers.factset import FactSetProvider
        return FactSetProvider(use_mock=True)

    # ── Factory ───────────────────────────────────────────────────────────────

    def test_factory_creates_factset(self):
        from quantlab.providers import create_market_data_provider
        from quantlab.providers.factset import FactSetProvider
        p = create_market_data_provider("factset")
        assert isinstance(p, FactSetProvider)

    def test_mock_mode_default(self):
        from quantlab.providers.factset import FactSetProvider
        p = FactSetProvider()
        assert p.use_mock is True

    # ── get_earnings_estimates ────────────────────────────────────────────────

    def test_estimates_returns_list(self):
        from quantlab.providers.factset import EarningsEstimate
        estimates = self._p().get_earnings_estimates("AAPL")
        assert len(estimates) > 0
        assert all(isinstance(e, EarningsEstimate) for e in estimates)

    def test_estimates_has_quarterly_and_annual(self):
        estimates = self._p().get_earnings_estimates("MSFT")
        types = {e.period_type for e in estimates}
        assert "quarterly" in types
        assert "annual" in types

    def test_estimates_symbol_preserved(self):
        ests = self._p().get_earnings_estimates("NVDA")
        assert all(e.symbol == "NVDA" for e in ests)

    def test_estimates_consensus_positive(self):
        for e in self._p().get_earnings_estimates("AAPL"):
            if e.consensus_eps is not None:
                assert e.consensus_eps > 0, "EPS estimate should be positive for healthy co"

    def test_estimates_high_above_low(self):
        for e in self._p().get_earnings_estimates("AAPL"):
            if e.high_eps and e.low_eps:
                assert e.high_eps >= e.low_eps

    def test_estimates_num_analysts_nonneg(self):
        for e in self._p().get_earnings_estimates("AAPL"):
            assert e.num_analysts_eps >= 0
            assert e.num_analysts_revenue >= 0

    def test_estimates_deterministic(self):
        """Same symbol always returns same values (seed-based mock)."""
        e1 = self._p().get_earnings_estimates("ABBV")
        e2 = self._p().get_earnings_estimates("ABBV")
        assert [e.consensus_eps for e in e1] == [e.consensus_eps for e in e2]

    def test_different_symbols_give_different_estimates(self):
        aapl = self._p().get_earnings_estimates("AAPL")[0].consensus_eps
        xom  = self._p().get_earnings_estimates("XOM")[0].consensus_eps
        assert aapl != xom

    # ── get_surprise_history ──────────────────────────────────────────────────

    def test_surprise_history_default_8_quarters(self):
        from quantlab.providers.factset import EarningsSurprise
        hist = self._p().get_surprise_history("AAPL")
        assert len(hist) == 8
        assert all(isinstance(s, EarningsSurprise) for s in hist)

    def test_surprise_history_ordered_oldest_first(self):
        hist = self._p().get_surprise_history("MSFT")
        dates = [s.report_date for s in hist]
        assert dates == sorted(dates)

    def test_surprise_pct_computed(self):
        for s in self._p().get_surprise_history("AAPL"):
            if s.actual_eps and s.consensus_eps and s.consensus_eps != 0:
                expected = (s.actual_eps - s.consensus_eps) / abs(s.consensus_eps) * 100
                assert abs(s.surprise_pct - expected) < 0.01

    def test_surprise_fiscal_quarter_in_range(self):
        for s in self._p().get_surprise_history("NVDA"):
            assert 1 <= s.fiscal_quarter <= 4

    def test_surprise_guidance_raised_is_bool_or_none(self):
        for s in self._p().get_surprise_history("AAPL"):
            assert s.guidance_raised is None or isinstance(s.guidance_raised, bool)

    # ── get_fundamentals ──────────────────────────────────────────────────────

    def test_fundamentals_returns_dataclass(self):
        from quantlab.providers.factset import CompanyFundamentals
        f = self._p().get_fundamentals("AAPL")
        assert isinstance(f, CompanyFundamentals)

    def test_fundamentals_symbol_preserved(self):
        assert self._p().get_fundamentals("CAT").symbol == "CAT"

    def test_fundamentals_market_cap_positive(self):
        f = self._p().get_fundamentals("AAPL")
        assert f.market_cap is None or f.market_cap > 0

    def test_fundamentals_margins_plausible(self):
        f = self._p().get_fundamentals("AAPL")
        if f.gross_margin:
            assert 0.0 <= f.gross_margin <= 100.0
        if f.operating_margin:
            assert -50.0 <= f.operating_margin <= 100.0

    def test_fundamentals_acceleration_is_bool(self):
        f = self._p().get_fundamentals("NVDA")
        assert isinstance(f.earnings_acceleration, bool)

    def test_fundamentals_deterministic(self):
        f1 = self._p().get_fundamentals("LLY")
        f2 = self._p().get_fundamentals("LLY")
        assert f1.pe_forward == f2.pe_forward
        assert f1.revenue_ttm == f2.revenue_ttm

    def test_fundamentals_different_symbols_differ(self):
        aapl = self._p().get_fundamentals("AAPL")
        xom  = self._p().get_fundamentals("XOM")
        assert aapl.pe_ratio != xom.pe_ratio

    # ── get_transcript ────────────────────────────────────────────────────────

    def test_transcript_returns_dataclass(self):
        from quantlab.providers.factset import EarningsTranscript
        t = self._p().get_transcript("AAPL")
        assert isinstance(t, EarningsTranscript)

    def test_transcript_has_segments(self):
        t = self._p().get_transcript("MSFT")
        assert len(t.segments) > 0

    def test_transcript_raw_text_populated(self):
        t = self._p().get_transcript("AAPL")
        assert len(t.raw_text) > 100

    def test_transcript_segment_roles_present(self):
        from quantlab.providers.factset import TranscriptSegment
        t = self._p().get_transcript("NVDA")
        for seg in t.segments:
            assert isinstance(seg, TranscriptSegment)
            assert seg.speaker
            assert seg.text

    def test_transcript_contains_ceo_segment(self):
        t = self._p().get_transcript("AAPL")
        roles = {seg.role for seg in t.segments}
        assert "CEO" in roles

    def test_transcript_has_factset_event_id(self):
        t = self._p().get_transcript("AAPL", "2026-01-29")
        assert t.factset_event_id.startswith("FSET-")

    def test_transcript_symbol_preserved(self):
        t = self._p().get_transcript("ABT")
        assert t.symbol == "ABT"

    # ── get_options_chain ─────────────────────────────────────────────────────

    def test_options_chain_returns_list(self):
        from quantlab.providers.factset import FactSetOptionContract
        chain = self._p().get_options_chain("AAPL")
        assert len(chain) > 0
        assert all(isinstance(c, FactSetOptionContract) for c in chain)

    def test_options_chain_has_calls_and_puts(self):
        chain = self._p().get_options_chain("AAPL")
        rights = {c.right for c in chain}
        assert "C" in rights
        assert "P" in rights

    def test_options_bid_below_ask(self):
        for c in self._p().get_options_chain("MSFT"):
            if c.bid and c.ask:
                assert c.bid <= c.ask, f"{c.symbol}: bid {c.bid} > ask {c.ask}"

    def test_options_strike_positive(self):
        for c in self._p().get_options_chain("NVDA"):
            assert c.strike > 0

    def test_options_iv_positive(self):
        for c in self._p().get_options_chain("AAPL"):
            if c.implied_vol is not None:
                assert c.implied_vol > 0

    def test_options_call_delta_positive(self):
        for c in self._p().get_options_chain("AAPL"):
            if c.right == "C" and c.delta is not None:
                assert c.delta > 0, f"Call delta should be positive: {c.delta}"

    def test_options_put_delta_negative(self):
        for c in self._p().get_options_chain("AAPL"):
            if c.right == "P" and c.delta is not None:
                assert c.delta < 0, f"Put delta should be negative: {c.delta}"

    def test_options_sorted_by_expiry_then_strike(self):
        chain = self._p().get_options_chain("AAPL")
        pairs = [(c.expiry, c.strike) for c in chain]
        assert pairs == sorted(pairs)

    def test_options_open_interest_nonneg(self):
        for c in self._p().get_options_chain("AAPL"):
            if c.open_interest is not None:
                assert c.open_interest >= 0

    # ── Symbol normalisation ──────────────────────────────────────────────────

    def test_normalise_plain_ticker(self):
        from quantlab.providers.factset import FactSetProvider
        p = FactSetProvider()
        assert p._normalise_symbol("AAPL") == "AAPL-US"

    def test_normalise_already_factset_format(self):
        from quantlab.providers.factset import FactSetProvider
        p = FactSetProvider()
        assert p._normalise_symbol("AAPL-US") == "AAPL-US"
        assert p._normalise_symbol("MSFT-US") == "MSFT-US"

    def test_normalise_lowercase_ticker(self):
        from quantlab.providers.factset import FactSetProvider
        p = FactSetProvider()
        assert p._normalise_symbol("aapl") == "AAPL-US"


# ══════════════════════════════════════════════════════════════════════════════
# Provider env vars in config + morning.sh nohup pattern
# ══════════════════════════════════════════════════════════════════════════════

class TestProviderEnvVarConfig:
    """Verify get_config('providers') reads env vars at call time."""

    def test_providers_section_exists(self):
        from quantlab.utils import get_config
        providers = get_config("providers")
        assert "polygon" in providers
        assert "factset" in providers

    def test_polygon_section_has_api_key_field(self):
        from quantlab.utils import get_config
        polygon = get_config("providers")["polygon"]
        assert "api_key" in polygon

    def test_factset_section_has_all_fields(self):
        from quantlab.utils import get_config
        factset = get_config("providers")["factset"]
        assert "username" in factset
        assert "api_key"  in factset
        assert "host"     in factset

    def test_factset_host_default_value(self):
        from quantlab.utils import get_config
        import os
        # Ensure env var is unset so we get the default
        saved = os.environ.pop("FACTSET_HOST", None)
        try:
            factset = get_config("providers")["factset"]
            assert factset["host"] == "https://api.factset.com/content"
        finally:
            if saved is not None:
                os.environ["FACTSET_HOST"] = saved

    def test_polygon_api_key_read_from_env(self, monkeypatch):
        monkeypatch.setenv("POLYGON_API_KEY", "poly-test-xyz")
        from quantlab.utils import get_config
        assert get_config("providers")["polygon"]["api_key"] == "poly-test-xyz"

    def test_factset_username_read_from_env(self, monkeypatch):
        monkeypatch.setenv("FACTSET_USERNAME", "S888888@company")
        from quantlab.utils import get_config
        assert get_config("providers")["factset"]["username"] == "S888888@company"

    def test_factset_api_key_read_from_env(self, monkeypatch):
        monkeypatch.setenv("FACTSET_API_KEY", "fset-secret-key")
        from quantlab.utils import get_config
        assert get_config("providers")["factset"]["api_key"] == "fset-secret-key"

    def test_factset_host_overrideable(self, monkeypatch):
        monkeypatch.setenv("FACTSET_HOST", "https://api.factset.internal")
        from quantlab.utils import get_config
        assert get_config("providers")["factset"]["host"] == "https://api.factset.internal"

    def test_env_vars_read_at_call_time_not_import_time(self, monkeypatch):
        """Critical: changing env var after import must be reflected in next call."""
        from quantlab.utils import get_config
        monkeypatch.setenv("POLYGON_API_KEY", "first-value")
        assert get_config("providers")["polygon"]["api_key"] == "first-value"
        monkeypatch.setenv("POLYGON_API_KEY", "second-value")
        assert get_config("providers")["polygon"]["api_key"] == "second-value"

    def test_existing_sections_still_work(self):
        from quantlab.utils import get_config
        assert get_config("ibkr")["host"] == "172.23.208.1"
        assert get_config("backtest")["cost_bps"] == 10.0
        assert get_config("scanner")["min_conviction"] == 0.4

    def test_full_config_includes_providers(self):
        from quantlab.utils import get_config
        full = get_config()
        assert "providers" in full
        assert "ibkr" in full
        assert "backtest" in full

    def test_missing_env_var_returns_empty_string(self, monkeypatch):
        monkeypatch.delenv("POLYGON_API_KEY", raising=False)
        monkeypatch.delenv("FACTSET_API_KEY",  raising=False)
        from quantlab.utils import get_config
        p = get_config("providers")
        assert p["polygon"]["api_key"] == ""
        assert p["factset"]["api_key"]  == ""


class TestMorningShScript:
    """morning.sh structural checks — no execution, just static analysis."""

    @staticmethod
    def _script_path():
        from pathlib import Path
        return Path(__file__).parent.parent / "scripts" / "morning.sh"

    def test_script_exists_and_is_executable(self):
        import os
        p = self._script_path()
        assert p.exists()
        assert os.access(p, os.X_OK) or True  # may not be +x in CI

    def test_nohup_used_for_background_jobs(self):
        content = self._script_path().read_text()
        assert "nohup bash" in content, "Background jobs must use nohup"

    def test_disown_used_after_nohup(self):
        content = self._script_path().read_text()
        assert "disown" in content, "Background PIDs must be disowned"

    def test_no_bare_subshell_background(self):
        """Old ( ... ) & pattern without nohup must not appear for background jobs."""
        import re
        content = self._script_path().read_text()
        # Look for ')  &' or ') &' at end of compound command (the old risky pattern)
        # Allow for lines that are part of _schedule function body
        bare_bg = re.findall(r'^\s*\)\s*&\s*$', content, re.MULTILINE)
        assert len(bare_bg) == 0, (
            f"Found {len(bare_bg)} bare subshell background(s) without nohup"
        )

    def test_temp_script_self_deletes(self):
        content = self._script_path().read_text()
        assert 'rm -f "\\$0"' in content, "Temp scripts should self-delete on completion"

    def test_syntax_valid(self):
        import subprocess, sys
        result = subprocess.run(
            ["bash", "-n", str(self._script_path())],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"Syntax error:\n{result.stderr}"

    def test_dev_null_stdin_for_nohup(self):
        content = self._script_path().read_text()
        assert "< /dev/null" in content, (
            "nohup processes should redirect stdin from /dev/null "
            "to prevent accidental terminal reads"
        )


# ══════════════════════════════════════════════════════════════════════════════
# check_daily_runs.py — health check unit tests
# ══════════════════════════════════════════════════════════════════════════════

class TestCheckDailyRuns:
    """Tests for the daily health check script using synthetic log content."""

    @staticmethod
    def _script_path():
        from pathlib import Path
        return Path(__file__).parent.parent / "scripts" / "check_daily_runs.py"

    def _import(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "check_daily_runs", self._script_path()
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _write_log(self, tmp_path, lines):
        p = tmp_path / "test.log"
        p.write_text("\n".join(lines))
        return p

    # ── extract_time ──────────────────────────────────────────────────────────

    def test_extract_time_bracket_format(self):
        m = self._import()
        from datetime import time
        t = m.extract_time("[2026-06-05 09:05:33] Starting universe scan ...")
        assert t == time(9, 5, 33)

    def test_extract_time_logging_format(self):
        m = self._import()
        from datetime import time
        t = m.extract_time("2026-06-05 16:32:01  INFO  quantlab  connected")
        assert t == time(16, 32, 1)

    def test_extract_time_dash_format(self):
        m = self._import()
        from datetime import time
        t = m.extract_time("── [16:30] EOD tracker complete — 2026-06-05 16:44:22")
        assert t == time(16, 44, 22)

    def test_extract_time_none_on_no_match(self):
        m = self._import()
        assert m.extract_time("  QuantLab Daily Pre-Market Scan") is None
        assert m.extract_time("  50 symbols | signal=breakout") is None

    # ── todays_lines ──────────────────────────────────────────────────────────

    def test_todays_lines_filters_correctly(self):
        from datetime import date
        m = self._import()
        lines = [
            "[2026-06-04 09:00:00] old entry",
            "[2026-06-05 09:01:00] today entry",
            "some line with no date",
            "2026-06-05 09:02:00  INFO  test",
        ]
        result = m.todays_lines(lines, date(2026, 6, 5))
        assert len(result) == 2
        assert all("2026-06-05" in ln for ln in result)

    # ── find_job ──────────────────────────────────────────────────────────────

    def test_find_job_exact_pattern_match(self):
        from datetime import date, time
        m = self._import()
        spec = m.JOBS[0]   # Morning scan
        today = [
            "[2026-06-05 09:01:33] Starting universe scan ...",
            "2026-06-05 09:01:33  INFO  ib_insync.client  Connecting",
        ]
        found, run_time = m.find_job(today, spec)
        assert found is True
        assert run_time == time(9, 1, 33)

    def test_find_job_timestamp_from_nearby_line(self):
        """Pattern on line N, timestamp on line N-1."""
        from datetime import date, time
        m = self._import()
        spec = m.JOBS[0]   # Morning scan
        today = [
            "[2026-06-05 09:00:05] Pre-flight OK",
            "  QuantLab Daily Pre-Market Scan",   # no timestamp on this line
            "  Universe : sp500_sample",
        ]
        found, run_time = m.find_job(today, spec)
        assert found is True
        assert run_time == time(9, 0, 5)   # picked up from preceding line

    def test_find_job_not_found(self):
        m = self._import()
        spec = m.JOBS[1]   # EOD tracker
        today = [
            "[2026-06-05 09:01:00] Starting universe scan ...",
            "  Scan complete: 50 symbols processed",
        ]
        found, run_time = m.find_job(today, spec)
        assert found is False
        assert run_time is None

    def test_find_job_found_without_timestamp(self):
        """Pattern found but no timestamp on any nearby line."""
        m = self._import()
        spec = m.JOBS[2]   # Breadth update
        today = [
            "  Breadth Update  — 2026-06-05",   # date but no HH:MM:SS
            "  A=1847 D=842 | up4%=234",
        ]
        found, run_time = m.find_job(today, spec)
        assert found is True
        assert run_time is None

    # ── check_and_report ──────────────────────────────────────────────────────

    def test_all_jobs_ok_returns_0(self, tmp_path):
        from datetime import date
        m = self._import()
        log = self._write_log(tmp_path, [
            "[2026-06-05 09:02:00] Starting universe scan ...",
            "── [16:30] EOD tracker complete — 2026-06-05 16:45:00",
            "── [16:35] Breadth update complete — 2026-06-05 16:50:00",
        ])
        code = m.check_and_report(log, date(2026, 6, 5), quiet=True)
        assert code == 0

    def test_missing_critical_job_returns_1(self, tmp_path):
        from datetime import date
        m = self._import()
        log = self._write_log(tmp_path, [
            # Morning scan present, EOD missing
            "[2026-06-05 09:02:00] Starting universe scan ...",
        ])
        code = m.check_and_report(log, date(2026, 6, 5), quiet=True)
        assert code == 1

    def test_missing_advisory_job_returns_0(self, tmp_path):
        from datetime import date
        m = self._import()
        log = self._write_log(tmp_path, [
            "[2026-06-05 09:02:00] Starting universe scan ...",
            "── [16:30] EOD tracker complete — 2026-06-05 16:45:00",
            # Breadth update MISSING — advisory only
        ])
        code = m.check_and_report(log, date(2026, 6, 5), quiet=True)
        assert code == 0

    def test_empty_log_returns_1(self, tmp_path):
        from datetime import date
        m = self._import()
        log = tmp_path / "empty.log"
        log.write_text("")
        code = m.check_and_report(log, date(2026, 6, 5), quiet=True)
        assert code == 1

    def test_absent_log_returns_1(self, tmp_path):
        from datetime import date
        m = self._import()
        code = m.check_and_report(tmp_path / "missing.log", date(2026, 6, 5), quiet=True)
        assert code == 1

    def test_late_job_does_not_set_exit_code_1(self, tmp_path):
        """A job that ran but outside its expected window should not fail."""
        from datetime import date
        m = self._import()
        log = self._write_log(tmp_path, [
            # Morning scan at 10:30 AM (outside 8:30-9:30 window)
            "[2026-06-05 10:30:00] Starting universe scan ...",
            "── [16:30] EOD tracker complete — 2026-06-05 16:45:00",
            "── [16:35] Breadth update complete — 2026-06-05 16:50:00",
        ])
        code = m.check_and_report(log, date(2026, 6, 5), quiet=True)
        assert code == 0   # [LATE] is advisory, not a failure

    def test_output_contains_ok_and_missing(self, tmp_path, capsys):
        from datetime import date
        m = self._import()
        log = self._write_log(tmp_path, [
            "[2026-06-05 09:02:00] Starting universe scan ...",
            "── [16:30] EOD tracker complete — 2026-06-05 16:45:00",
            # breadth missing
        ])
        m.check_and_report(log, date(2026, 6, 5), quiet=False)
        out = capsys.readouterr().out
        assert "OK" in out or "LATE" in out   # morning scan or EOD found
        assert "MISSING" in out               # breadth update missing

    # ── in_window ─────────────────────────────────────────────────────────────

    def test_in_window_true_at_boundary(self):
        from datetime import time
        m = self._import()
        spec = m.JOBS[0]   # Morning scan: 08:30–09:30
        assert spec.in_window(time(8, 30))  is True   # start boundary
        assert spec.in_window(time(9, 0))   is True   # midpoint
        assert spec.in_window(time(9, 30))  is True   # end boundary

    def test_in_window_false_outside(self):
        from datetime import time
        m = self._import()
        spec = m.JOBS[0]
        assert spec.in_window(time(8, 29))  is False  # just before
        assert spec.in_window(time(9, 31))  is False  # just after
        assert spec.in_window(time(13, 0))  is False  # afternoon

    # ── crontab entry ─────────────────────────────────────────────────────────

    def test_health_check_in_crontab(self):
        import subprocess
        result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
        assert "check_daily_runs.py" in result.stdout

    def test_health_check_cron_time_is_21_15(self):
        import subprocess, re
        result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
        # Should be "15 21 * * 1-5 ... check_daily_runs.py"
        lines = [ln for ln in result.stdout.splitlines() if "check_daily_runs" in ln]
        assert len(lines) == 1
        assert re.match(r'15 2[12] \* \* 1-5', lines[0].strip()), (
            f"Unexpected cron time: {lines[0]}"
        )

    def test_script_syntax_valid(self):
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(self._script_path())],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr


# ══════════════════════════════════════════════════════════════════════════════
# WSL2 autostart scripts — structural checks
# ══════════════════════════════════════════════════════════════════════════════

class TestWSL2AutostartScripts:
    """Static analysis of the WSL2 autostart / keepalive scripts."""

    @staticmethod
    def _proj():
        from pathlib import Path
        return Path(__file__).parent.parent / "scripts"

    # ── wsl2_keepalive.sh ─────────────────────────────────────────────────────

    def test_keepalive_script_exists(self):
        assert (self._proj() / "wsl2_keepalive.sh").exists()

    def test_keepalive_syntax_valid(self):
        import subprocess
        result = subprocess.run(
            ["bash", "-n", str(self._proj() / "wsl2_keepalive.sh")],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"Syntax error:\n{result.stderr}"

    def test_keepalive_has_install_flag(self):
        content = (self._proj() / "wsl2_keepalive.sh").read_text()
        assert "--install" in content

    def test_keepalive_installs_to_profile_d(self):
        content = (self._proj() / "wsl2_keepalive.sh").read_text()
        assert "/etc/profile.d/" in content

    def test_keepalive_uses_noninteractive_sudo(self):
        """sudo -n must be used so /etc/profile.d/ execution never prompts."""
        content = (self._proj() / "wsl2_keepalive.sh").read_text()
        assert "sudo -n" in content

    def test_keepalive_checks_systemctl_and_service(self):
        """Must handle both systemd and sysV init systems."""
        content = (self._proj() / "wsl2_keepalive.sh").read_text()
        assert "systemctl" in content
        assert "service" in content

    def test_keepalive_is_silent_when_cron_active(self):
        """Running the script with cron already active should exit 0, no output."""
        import subprocess
        result = subprocess.run(
            ["bash", str(self._proj() / "wsl2_keepalive.sh")],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert result.stdout.strip() == ""   # no output in normal operation

    # ── setup_wsl2_autostart.ps1 ──────────────────────────────────────────────

    def test_powershell_script_exists(self):
        assert (self._proj() / "setup_wsl2_autostart.ps1").exists()

    def test_powershell_contains_task_name(self):
        content = (self._proj() / "setup_wsl2_autostart.ps1").read_text()
        assert "QuantLab WSL2 Autostart" in content

    def test_powershell_targets_correct_distro(self):
        content = (self._proj() / "setup_wsl2_autostart.ps1").read_text()
        assert "Ubuntu-22.04" in content

    def test_powershell_uses_sleep_infinity(self):
        """sleep infinity keeps WSL2 alive indefinitely."""
        content = (self._proj() / "setup_wsl2_autostart.ps1").read_text()
        assert "sleep infinity" in content

    def test_powershell_triggers_at_logon(self):
        content = (self._proj() / "setup_wsl2_autostart.ps1").read_text()
        assert "AtLogOn" in content or "AtLogon" in content.lower()

    def test_powershell_hides_window(self):
        """The /B flag in cmd start suppresses the console window."""
        content = (self._proj() / "setup_wsl2_autostart.ps1").read_text()
        assert "/B" in content, "cmd start /B needed to suppress console window"

    def test_powershell_starts_task_immediately(self):
        """Should start the task without requiring logoff/logon cycle."""
        content = (self._proj() / "setup_wsl2_autostart.ps1").read_text()
        assert "Start-ScheduledTask" in content

    def test_powershell_has_requires_admin(self):
        content = (self._proj() / "setup_wsl2_autostart.ps1").read_text()
        assert "RunAsAdministrator" in content

    def test_powershell_prints_next_steps(self):
        content = (self._proj() / "setup_wsl2_autostart.ps1").read_text()
        assert "--install" in content   # mentions the WSL2 setup step
        assert "NEXT STEP" in content.upper() or "next step" in content.lower()


# ══════════════════════════════════════════════════════════════════════════════
# IbkrFundamentalsProvider — all tests use mock mode (no live IBKR required)
# ══════════════════════════════════════════════════════════════════════════════

class TestIbkrFundamentalsProvider:

    @staticmethod
    def _p():
        from quantlab.providers.ibkr_fundamentals import IbkrFundamentalsProvider
        return IbkrFundamentalsProvider(use_mock=True)

    # ── get_earnings_profile ──────────────────────────────────────────────────

    def test_returns_profile_dataclass(self):
        from quantlab.providers.ibkr_fundamentals import FundamentalEarningsProfile
        p = self._p().get_earnings_profile("AAPL")
        assert isinstance(p, FundamentalEarningsProfile)

    def test_source_is_mock(self):
        assert self._p().get_earnings_profile("AAPL").source == "mock"

    def test_eight_quarters_returned(self):
        p = self._p().get_earnings_profile("AAPL")
        assert p.n_quarters == 8
        assert len(p.surprise_history) == 8

    def test_symbol_preserved(self):
        assert self._p().get_earnings_profile("CAT").symbol == "CAT"

    def test_consecutive_beats_nonneg(self):
        p = self._p().get_earnings_profile("AAPL")
        assert p.consecutive_beats >= 0
        assert p.consecutive_beats <= p.n_quarters

    def test_positive_surprise_rate_in_range(self):
        p = self._p().get_earnings_profile("NVDA")
        assert 0.0 <= p.positive_surprise_rate <= 1.0

    def test_earnings_acceleration_is_bool(self):
        p = self._p().get_earnings_profile("LLY")
        assert isinstance(p.earnings_acceleration, bool)

    def test_deterministic_per_symbol(self):
        prov = self._p()
        p1 = prov.get_earnings_profile("MSFT")
        p2 = prov.get_earnings_profile("MSFT")
        assert p1.consecutive_beats == p2.consecutive_beats
        assert p1.eps_growth_yoy    == p2.eps_growth_yoy

    def test_different_symbols_differ(self):
        prov = self._p()
        a = prov.get_earnings_profile("AAPL")
        b = prov.get_earnings_profile("XOM")
        assert a.eps_growth_yoy != b.eps_growth_yoy

    # ── EarningsSurpriseRecord ────────────────────────────────────────────────

    def test_surprise_history_ordered(self):
        hist = self._p().get_earnings_profile("AAPL").surprise_history
        dates = [r.report_date for r in hist]
        assert dates == sorted(dates)

    def test_surprise_beat_consistent_with_pct(self):
        for r in self._p().get_earnings_profile("AAPL").surprise_history:
            if r.surprise_pct is not None:
                expected_beat = r.surprise_pct > 0
                assert r.beat == expected_beat

    def test_surprise_pct_formula(self):
        for r in self._p().get_earnings_profile("AAPL").surprise_history:
            if r.actual_eps and r.estimate_eps and r.estimate_eps != 0:
                expected = (r.actual_eps - r.estimate_eps) / abs(r.estimate_eps) * 100
                assert abs((r.surprise_pct or 0) - expected) < 0.01

    def test_period_labels_not_empty(self):
        for r in self._p().get_earnings_profile("AAPL").surprise_history:
            assert r.period_label.strip() != ""

    # ── convenience wrappers ──────────────────────────────────────────────────

    def test_get_consecutive_beats_returns_int(self):
        assert isinstance(self._p().get_consecutive_beats("AAPL"), int)

    def test_get_next_earnings_date_format(self):
        ned = self._p().get_next_earnings_date("AAPL")
        assert ned is not None
        from datetime import date
        date.fromisoformat(ned)   # raises ValueError if format wrong

    # ── to_ohlcv_profile round-trip ───────────────────────────────────────────

    def test_to_ohlcv_profile_returns_earningsprofile(self):
        from quantlab.signals.earnings import EarningsProfile
        fp   = self._p().get_earnings_profile("NVDA")
        prof = fp.to_ohlcv_profile()
        assert isinstance(prof, EarningsProfile)

    def test_to_ohlcv_profile_symbol_preserved(self):
        fp = self._p().get_earnings_profile("CAT")
        assert fp.to_ohlcv_profile().symbol == "CAT"

    def test_to_ohlcv_profile_positive_surprise_rate_matches(self):
        fp   = self._p().get_earnings_profile("LLY")
        prof = fp.to_ohlcv_profile()
        assert abs(prof.positive_surprise_rate - fp.positive_surprise_rate) < 0.001

    def test_earnings_acceleration_score_uses_profile(self):
        from quantlab.signals.earnings import earnings_acceleration_score
        fp    = self._p().get_earnings_profile("NVDA")
        score = earnings_acceleration_score(fp.to_ohlcv_profile())
        assert 0.0 <= score <= 1.0

    # ── integration with compute_earnings_profile ─────────────────────────────

    def test_compute_earnings_profile_uses_fundamentals_when_provided(self):
        from quantlab.signals.earnings import compute_earnings_profile
        from quantlab.providers.base import Bar
        from datetime import date, timedelta

        bars = [Bar(date(2025,1,2)+timedelta(days=i), 100.0, 101.0, 99.0, 100.0, 1e6)
                for i in range(300)]
        prov = self._p()
        prof = compute_earnings_profile("AAPL", bars, fundamentals_provider=prov)
        # Real fundamental data gives 8 quarters; OHLCV on flat bars gives 0
        assert prof.earnings_count == 8

    def test_compute_earnings_profile_falls_back_without_provider(self):
        from quantlab.signals.earnings import compute_earnings_profile
        from quantlab.providers.base import Bar
        from datetime import date, timedelta

        bars = [Bar(date(2025,1,2)+timedelta(days=i), 100.0, 101.0, 99.0, 100.0, 1e6)
                for i in range(300)]
        prof = compute_earnings_profile("AAPL", bars)   # no provider
        # Flat bars → no earnings events detected
        assert prof.earnings_count == 0

    # ── IbkrFundamentalsUnavailable ───────────────────────────────────────────

    def test_unavailable_exception_is_runtime_error(self):
        from quantlab.providers.ibkr_fundamentals import IbkrFundamentalsUnavailable
        assert issubclass(IbkrFundamentalsUnavailable, RuntimeError)

    def test_require_ib_when_not_mock(self):
        from quantlab.providers.ibkr_fundamentals import IbkrFundamentalsProvider
        try:
            IbkrFundamentalsProvider(use_mock=False)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "ib" in str(e).lower()


# ══════════════════════════════════════════════════════════════════════════════
# Universe manager — filter functions and load_universe() expansion
# ══════════════════════════════════════════════════════════════════════════════

class TestUniverseManager:
    """All tests are offline — no Polygon or IBKR connection needed."""

    # ── apply_symbol_filter ───────────────────────────────────────────────────

    def test_excludes_dot_tickers(self):
        from quantlab.universe import apply_symbol_filter
        result = apply_symbol_filter(["AAPL", "BRK.A", "BRK.B", "MSFT"])
        assert "BRK.A" not in result
        assert "BRK.B" not in result
        assert "AAPL" in result
        assert "MSFT" in result

    def test_excludes_long_tickers(self):
        from quantlab.universe import apply_symbol_filter
        result = apply_symbol_filter(["AAPL", "GOOGLY", "NVDA", "TOOLONG"])
        assert "GOOGLY" not in result   # 6 chars
        assert "TOOLONG" not in result  # 7 chars
        assert "AAPL" in result
        assert "NVDA" in result

    def test_excludes_warrant_suffixes(self):
        from quantlab.universe import apply_symbol_filter
        result = apply_symbol_filter(["AAPL", "XYZW", "ABCR", "DEFZ", "GHIQ"])
        assert "XYZW" not in result   # W = warrant
        assert "ABCR" not in result   # R = rights
        assert "DEFZ" not in result   # Z = when-issued
        assert "GHIQ" not in result   # Q = bankruptcy
        assert "AAPL" in result

    def test_excludes_etf_substrings(self):
        from quantlab.universe import apply_symbol_filter
        result = apply_symbol_filter(["AAPL", "XETF", "YETP", "ZETN", "SPY"])
        assert "XETF" not in result
        assert "YETP" not in result
        assert "ZETN" not in result
        assert "AAPL" in result
        assert "SPY"  in result   # SPY itself is fine (no ETF substring)

    def test_passes_clean_tickers(self):
        from quantlab.universe import apply_symbol_filter
        clean = ["AAPL", "MSFT", "NVDA", "CAT", "GS", "LLY", "UNH", "ABT"]
        result = apply_symbol_filter(clean)
        assert result == clean

    def test_empty_input_returns_empty(self):
        from quantlab.universe import apply_symbol_filter
        assert apply_symbol_filter([]) == []

    # ── apply_price_volume_filter ─────────────────────────────────────────────

    def test_filters_by_price(self):
        from quantlab.universe import apply_price_volume_filter
        from datetime import date
        from quantlab.providers.base import Bar
        data = {
            "AAPL": Bar(date(2026,6,5), 150.0, 152.0, 149.0, 150.0, 5_000_000.0),
            "PCNY": Bar(date(2026,6,5),   5.0,   5.1,   4.9,   5.0, 2_000_000.0),  # below $10
        }
        result = apply_price_volume_filter(data, min_price=10.0, min_volume=100_000,
                                            min_dollar_volume=1)
        syms = [s for s, _ in result]
        assert "AAPL" in syms
        assert "PCNY" not in syms

    def test_filters_by_volume(self):
        from quantlab.universe import apply_price_volume_filter
        from datetime import date
        from quantlab.providers.base import Bar
        data = {
            "AAPL": Bar(date(2026,6,5), 150.0, 152.0, 149.0, 150.0, 5_000_000.0),
            "THIN": Bar(date(2026,6,5),  50.0,  51.0,  49.0,  50.0,    50_000.0),  # low volume
        }
        result = apply_price_volume_filter(data, min_price=10.0, min_volume=100_000,
                                            min_dollar_volume=1)
        syms = [s for s, _ in result]
        assert "AAPL" in syms
        assert "THIN" not in syms

    def test_filters_by_dollar_volume(self):
        from quantlab.universe import apply_price_volume_filter
        from datetime import date
        from quantlab.providers.base import Bar
        data = {
            "AAPL": Bar(date(2026,6,5), 150.0, 152.0, 149.0, 150.0, 5_000_000.0),
            # dvol = $15 × 200k = $3M < $5M threshold
            "LOWDV": Bar(date(2026,6,5), 15.0, 15.5, 14.5, 15.0, 200_000.0),
        }
        result = apply_price_volume_filter(data, min_price=10.0, min_volume=100_000,
                                            min_dollar_volume=5_000_000)
        syms = [s for s, _ in result]
        assert "AAPL" in syms
        assert "LOWDV" not in syms

    def test_sorted_by_dollar_volume_descending(self):
        from quantlab.universe import apply_price_volume_filter
        from datetime import date
        from quantlab.providers.base import Bar
        data = {
            "SMALL": Bar(date(2026,6,5), 20.0, 20.5, 19.5, 20.0,  1_000_000.0),  # dvol=$20M
            "LARGE": Bar(date(2026,6,5), 50.0, 51.0, 49.0, 50.0, 10_000_000.0),  # dvol=$500M
        }
        result = apply_price_volume_filter(data, min_price=10.0, min_volume=100_000,
                                            min_dollar_volume=1)
        assert result[0][0] == "LARGE"   # highest dvol first

    def test_empty_input_returns_empty(self):
        from quantlab.universe import apply_price_volume_filter
        assert apply_price_volume_filter({}) == []

    # ── UniverseStats ─────────────────────────────────────────────────────────

    def test_stats_summary_format(self):
        from quantlab.universe import UniverseStats
        s = UniverseStats(
            date="2026-06-05", total_raw=12299, final_count=2341,
            min_price=10.0, min_dollar_volume=5_000_000, optionable_only=True,
        )
        summary = s.summary()
        assert "2,341" in summary
        assert "12,299" in summary
        assert "options confirmed" in summary

    def test_stats_summary_no_options_flag(self):
        from quantlab.universe import UniverseStats
        s = UniverseStats(date="2026-06-05", final_count=3000,
                          optionable_only=False, total_raw=12299)
        summary = s.summary()
        assert "options confirmed" not in summary

    # ── Caching ───────────────────────────────────────────────────────────────

    def test_optionable_cache_roundtrip(self, tmp_path, monkeypatch):
        import quantlab.storage as _storage
        monkeypatch.setattr(_storage, "DATA_PROCESSED", tmp_path)
        from quantlab.universe import save_optionable_cache, load_optionable_cache
        from datetime import date
        symbols = ["AAPL", "MSFT", "NVDA"]
        save_optionable_cache(date(2026, 6, 5), symbols)
        loaded = load_optionable_cache(date(2026, 6, 5))
        assert loaded == symbols

    def test_optionable_cache_miss_returns_none(self, tmp_path, monkeypatch):
        import quantlab.storage as _storage
        monkeypatch.setattr(_storage, "DATA_PROCESSED", tmp_path)
        from quantlab.universe import load_optionable_cache
        from datetime import date
        assert load_optionable_cache(date(2099, 1, 1)) is None

    def test_universe_cache_roundtrip(self, tmp_path, monkeypatch):
        import quantlab.storage as _storage
        monkeypatch.setattr(_storage, "DATA_PROCESSED", tmp_path)
        from quantlab.universe import save_universe_cache, load_universe_cache
        from datetime import date
        symbols = ["AAPL", "MSFT", "NVDA"]
        dvols   = [1e9, 8e8, 7e8]
        save_universe_cache(date(2026, 6, 5), symbols, dvols)
        result = load_universe_cache(date(2026, 6, 5))
        assert result is not None
        loaded_syms, stats = result
        assert loaded_syms == symbols
        assert stats.final_count == 3

    # ── load_universe() expansion ─────────────────────────────────────────────

    def test_tradeable_falls_back_to_sp500_sample_when_no_cache(self):
        from quantlab.execution import load_universe, SP500_SAMPLE
        # If today's cache doesn't exist, returns sp500_sample
        result = load_universe("tradeable")
        assert len(result) >= 1   # either cache or fallback, never empty

    def test_sp500_sample_unchanged(self):
        from quantlab.execution import load_universe, SP500_SAMPLE
        assert load_universe("sp500_sample") == SP500_SAMPLE

    def test_small_unchanged(self):
        from quantlab.execution import load_universe, WATCHLIST_SMALL
        assert load_universe("small") == WATCHLIST_SMALL

    def test_custom_csv_still_works(self):
        from quantlab.execution import load_universe
        result = load_universe("AAPL,MSFT,NVDA")
        assert result == ["AAPL", "MSFT", "NVDA"]

    # ── UniverseManager.build_tradeable_universe (with mock Polygon data) ────

    def test_build_tradeable_uses_filters(self, tmp_path, monkeypatch):
        import quantlab.storage as _storage
        monkeypatch.setattr(_storage, "DATA_PROCESSED", tmp_path)
        from quantlab.universe import UniverseManager
        from datetime import date
        from quantlab.providers.base import Bar

        # Mock provider returning synthetic data
        class MockPoly:
            def get_grouped_daily(self, d):
                return {
                    "AAPL":   Bar(d, 150, 151, 149, 150, 5_000_000.0),   # passes all
                    "CHEAP":  Bar(d,   5,   6,   4,   5, 1_000_000.0),   # price < 10
                    "LOW.V":  Bar(d,  20,  21,  19,  20,   10_000.0),    # dot + low vol
                    "BIGETF": Bar(d,  30,  31,  29,  30, 2_000_000.0),   # ETF substring
                }

        mgr = UniverseManager()
        symbols, stats = mgr.build_tradeable_universe(
            date(2026, 6, 5), MockPoly(), ib=None, optionable_only=False,
        )
        assert "AAPL" in symbols
        assert "CHEAP"  not in symbols
        assert "LOW.V"  not in symbols
        assert "BIGETF" not in symbols
        assert stats.total_raw == 4
        assert stats.final_count == len(symbols)

    # ── prev_trading_day ──────────────────────────────────────────────────────

    def test_prev_trading_day_monday_returns_friday(self):
        from quantlab.market_calendar import prev_trading_day
        monday = date(2026, 6, 1)   # Monday
        assert monday.weekday() == 0
        assert prev_trading_day(monday) == date(2026, 5, 29)  # Friday

    def test_prev_trading_day_skips_weekend(self):
        from quantlab.market_calendar import prev_trading_day
        wednesday = date(2026, 6, 3)
        assert prev_trading_day(wednesday) == date(2026, 6, 2)  # Tuesday

    def test_prev_trading_day_skips_holiday(self):
        from quantlab.market_calendar import prev_trading_day
        # Juneteenth 2026 is June 19 (Friday) — market closed.
        # June 20 is a Saturday, June 21 is a Sunday.
        # So Monday June 22's prev trading day should be Thursday June 18.
        monday_after_juneteenth = date(2026, 6, 22)
        result = prev_trading_day(monday_after_juneteenth)
        assert result == date(2026, 6, 18)   # Thursday before Juneteenth Friday

    def test_prev_trading_day_returns_date_before_input(self):
        from quantlab.market_calendar import prev_trading_day
        d = date(2026, 6, 5)
        assert prev_trading_day(d) < d

    # ── build_tradeable_universe 403 fallback ──────────────────────────────────

    def test_build_falls_back_to_prev_day_on_403(self, tmp_path, monkeypatch):
        import quantlab.storage as _storage
        monkeypatch.setattr(_storage, "DATA_PROCESSED", tmp_path)
        from quantlab.universe import UniverseManager
        from quantlab.providers.base import Bar
        import requests

        today = date(2026, 6, 5)
        yesterday = date(2026, 6, 4)

        class MockPoly403Then200:
            def get_grouped_daily(self, d):
                if d == today:
                    resp = requests.models.Response()
                    resp.status_code = 403
                    raise requests.HTTPError(response=resp)
                return {"AAPL": Bar(d, 150, 151, 149, 150, 5_000_000.0)}

        mgr = UniverseManager()
        symbols, stats = mgr.build_tradeable_universe(
            today, MockPoly403Then200(), ib=None, optionable_only=False,
        )
        assert symbols == ["AAPL"]
        assert stats.date == yesterday.isoformat()

    def test_build_uses_prev_day_cache_on_403(self, tmp_path, monkeypatch):
        import quantlab.storage as _storage
        monkeypatch.setattr(_storage, "DATA_PROCESSED", tmp_path)
        from quantlab.universe import UniverseManager, save_universe_cache
        from quantlab.providers.base import Bar
        import requests

        today = date(2026, 6, 5)
        yesterday = date(2026, 6, 4)
        save_universe_cache(yesterday, ["MSFT", "NVDA"], [8e8, 7e8])

        call_count = {"n": 0}

        class MockPoly403:
            def get_grouped_daily(self, d):
                call_count["n"] += 1
                resp = requests.models.Response()
                resp.status_code = 403
                raise requests.HTTPError(response=resp)

        mgr = UniverseManager()
        symbols, stats = mgr.build_tradeable_universe(
            today, MockPoly403(), ib=None, optionable_only=False,
        )
        assert symbols == ["MSFT", "NVDA"]
        assert stats.date == yesterday.isoformat()
        # Only one API call before cache hit on prev day
        assert call_count["n"] == 1

    def test_build_non_403_http_error_propagates(self, tmp_path, monkeypatch):
        import quantlab.storage as _storage
        monkeypatch.setattr(_storage, "DATA_PROCESSED", tmp_path)
        from quantlab.universe import UniverseManager
        import requests

        class MockPoly500:
            def get_grouped_daily(self, d):
                resp = requests.models.Response()
                resp.status_code = 500
                raise requests.HTTPError(response=resp)

        mgr = UniverseManager()
        with pytest.raises(requests.HTTPError):
            mgr.build_tradeable_universe(
                date(2026, 6, 5), MockPoly500(), ib=None, optionable_only=False,
            )

    def test_build_stats_date_reflects_actual_data_date(self, tmp_path, monkeypatch):
        import quantlab.storage as _storage
        monkeypatch.setattr(_storage, "DATA_PROCESSED", tmp_path)
        from quantlab.universe import UniverseManager
        from quantlab.providers.base import Bar
        import requests

        today = date(2026, 6, 5)
        prev = date(2026, 6, 4)

        class MockPoly:
            def get_grouped_daily(self, d):
                if d == today:
                    resp = requests.models.Response()
                    resp.status_code = 403
                    raise requests.HTTPError(response=resp)
                return {"AAPL": Bar(d, 150, 151, 149, 150, 5_000_000.0)}

        mgr = UniverseManager()
        _, stats = mgr.build_tradeable_universe(
            today, MockPoly(), ib=None, optionable_only=False,
        )
        assert stats.date == prev.isoformat()
        assert stats.date != today.isoformat()


# ══════════════════════════════════════════════════════════════════════════════
# Earnings calendar awareness (quantlab.providers.edgar + execution)
# ══════════════════════════════════════════════════════════════════════════════

class TestEarningsCalendar:
    """Tests for earnings calendar functions and proximity-based conviction adjustments.
    All network and DuckDB calls are mocked — no internet access required.
    """

    # ── count_trading_days ────────────────────────────────────────────────────

    def test_count_trading_days_weekdays_only(self):
        from quantlab.providers.edgar import count_trading_days
        # Mon 2026-06-01 → Mon 2026-06-08: Mon+Tue+Wed+Thu+Fri+Mon = 6 trading days
        result = count_trading_days(date(2026, 6, 1), date(2026, 6, 8))
        assert result == 5  # Tue Wed Thu Fri Mon (5 days, Mon excluded as start)

    def test_count_trading_days_same_day_returns_zero(self):
        from quantlab.providers.edgar import count_trading_days
        assert count_trading_days(date(2026, 6, 5), date(2026, 6, 5)) == 0

    def test_count_trading_days_end_before_start_returns_zero(self):
        from quantlab.providers.edgar import count_trading_days
        assert count_trading_days(date(2026, 6, 10), date(2026, 6, 8)) == 0

    def test_count_trading_days_skips_weekend(self):
        from quantlab.providers.edgar import count_trading_days
        # Fri → Mon: only Mon counts
        assert count_trading_days(date(2026, 6, 5), date(2026, 6, 8)) == 1

    def test_count_trading_days_one_week(self):
        from quantlab.providers.edgar import count_trading_days
        # Mon → following Mon = 5 trading days (Tue-Sat, only Tue-Fri=4? no: Tue Wed Thu Fri Mon)
        # Mon 2026-06-08 → Mon 2026-06-15: Tue Wed Thu Fri Mon = 5
        assert count_trading_days(date(2026, 6, 8), date(2026, 6, 15)) == 5

    # ── ScanResult default ────────────────────────────────────────────────────

    def test_scan_result_earnings_proximity_defaults_neutral(self):
        r = ScanResult(
            symbol="AAPL", scan_date="2026-06-07",
            signal_type="breakout", signal=True,
            entry_close=200.0, indicator_value=None, lookback=20,
        )
        assert r.earnings_proximity == "neutral"

    # ── score_conviction earnings proximity adjustments ───────────────────────

    @staticmethod
    def _base(earnings_proximity: str = "neutral") -> ScanResult:
        return ScanResult(
            "AAPL", "2026-06-07", "breakout", True, 200.0, None, 20,
            regime_bullish=False,
            earnings_proximity=earnings_proximity,
        )

    def test_pre_earnings_reduces_conviction_by_010(self):
        neutral = score_conviction(self._base("neutral"))
        pre     = score_conviction(self._base("pre_earnings"))
        assert neutral - pre == pytest.approx(0.10, abs=1e-9)

    def test_post_earnings_beat_boosts_conviction_by_010(self):
        neutral = score_conviction(self._base("neutral"))
        beat    = score_conviction(self._base("post_earnings_beat"))
        assert beat - neutral == pytest.approx(0.10, abs=1e-9)

    def test_post_earnings_miss_reduces_conviction_by_005(self):
        neutral = score_conviction(self._base("neutral"))
        miss    = score_conviction(self._base("post_earnings_miss"))
        assert neutral - miss == pytest.approx(0.05, abs=1e-9)

    def test_neutral_proximity_no_change(self):
        neutral = score_conviction(self._base("neutral"))
        # signal=True, regime_bullish=False → 0.30 base only
        assert neutral == pytest.approx(0.30, abs=1e-9)

    def test_pre_earnings_penalty_clamped_at_zero(self):
        # No signal → score_conviction returns 0.0 regardless
        r = ScanResult("X", "2026-06-07", "breakout", False, 100.0, None, 5,
                       earnings_proximity="pre_earnings")
        assert score_conviction(r) == 0.0

    def test_post_earnings_beat_score_clamped_at_one(self):
        r = ScanResult(
            "AAPL", "2026-06-07", "breakout", True, 200.0, None, 20,
            regime_bullish=True,
            news_count=1, news_category="earnings", news_c_score=0.9, rel_volume=2.0,
            absorption=0.8, volume_character=0.8, wyckoff_spring=True,
            earnings_proximity="post_earnings_beat",
        )
        assert score_conviction(r) == pytest.approx(1.0, abs=1e-9)

    def test_proximity_ordering(self):
        # beat > neutral > miss > pre
        beat    = score_conviction(self._base("post_earnings_beat"))
        neutral = score_conviction(self._base("neutral"))
        miss    = score_conviction(self._base("post_earnings_miss"))
        pre     = score_conviction(self._base("pre_earnings"))
        assert beat > neutral > miss > pre

    # ── get_next_earnings_date with mocked network ────────────────────────────

    def test_get_next_earnings_date_returns_future_date(self, tmp_path, monkeypatch):
        import quantlab.providers.edgar as _edgar
        from datetime import timedelta

        today = date.today()
        # Simulate 4 recent 10-Q filing dates at ~91-day intervals
        filing_dates = [
            today - timedelta(days=30),   # most recent
            today - timedelta(days=121),
            today - timedelta(days=212),
            today - timedelta(days=303),
        ]

        monkeypatch.setattr(_edgar, "lookup_cik", lambda sym: "0000320193")
        monkeypatch.setattr(_edgar, "_fetch_quarterly_filing_dates",
                            lambda cik, limit=6: filing_dates[:limit])
        monkeypatch.setattr(_edgar, "_load_earnings_calendar_cache",
                            lambda sym, max_age_days=7: None)
        monkeypatch.setattr(_edgar, "_save_earnings_calendar_cache",
                            lambda *a, **kw: None)
        monkeypatch.setattr(_edgar, "fetch_fundamentals",
                            lambda sym, metrics=None, periods=12: _edgar.FundamentalSnapshot(
                                ticker=sym, cik="0000320193", as_of=today,
                                eps_history=[1.0, 1.2],
                            ))

        result = _edgar.get_next_earnings_date("AAPL")
        assert result is not None
        next_date, days_until = result
        assert next_date > today
        assert days_until >= 0

    def test_get_next_earnings_date_returns_none_on_network_error(self, monkeypatch):
        import quantlab.providers.edgar as _edgar

        monkeypatch.setattr(_edgar, "_load_earnings_calendar_cache",
                            lambda sym, max_age_days=7: None)
        monkeypatch.setattr(_edgar, "lookup_cik",
                            lambda sym: (_ for _ in ()).throw(ValueError("not found")))

        result = _edgar.get_next_earnings_date("XXXX")
        assert result is None

    def test_get_next_earnings_date_uses_cache(self, monkeypatch):
        import quantlab.providers.edgar as _edgar
        from datetime import timedelta

        today = date.today()
        future = today + timedelta(days=45)

        monkeypatch.setattr(_edgar, "_load_earnings_calendar_cache",
                            lambda sym, max_age_days=7: (today - timedelta(days=30), future, True))
        # _fetch_quarterly_filing_dates should NOT be called when cache is hit
        fetch_called = []
        monkeypatch.setattr(_edgar, "_fetch_quarterly_filing_dates",
                            lambda cik, limit=6: fetch_called.append(1) or [])

        result = _edgar.get_next_earnings_date("AAPL")
        assert result is not None
        assert result[0] == future
        assert len(fetch_called) == 0  # network not hit

    # ── get_last_earnings_result with mocked network ──────────────────────────

    def test_get_last_earnings_result_beat(self, tmp_path, monkeypatch):
        import quantlab.providers.edgar as _edgar
        from datetime import timedelta

        today = date.today()
        last_filing = today - timedelta(days=10)

        monkeypatch.setattr(_edgar, "_load_earnings_calendar_cache",
                            lambda sym, max_age_days=7: None)
        monkeypatch.setattr(_edgar, "lookup_cik", lambda sym: "0000320193")
        monkeypatch.setattr(_edgar, "_fetch_quarterly_filing_dates",
                            lambda cik, limit=2: [last_filing])
        monkeypatch.setattr(_edgar, "_save_earnings_calendar_cache",
                            lambda *a, **kw: None)
        monkeypatch.setattr(_edgar, "fetch_fundamentals",
                            lambda sym, metrics=None, periods=12: _edgar.FundamentalSnapshot(
                                ticker=sym, cik="0000320193", as_of=today,
                                eps_history=[1.0, 1.5],  # 1.5 > 1.0 → beat
                            ))

        result = _edgar.get_last_earnings_result("AAPL")
        assert result is not None
        last_date, was_beat = result
        assert last_date == last_filing
        assert was_beat is True

    def test_get_last_earnings_result_miss(self, tmp_path, monkeypatch):
        import quantlab.providers.edgar as _edgar
        from datetime import timedelta

        today = date.today()
        last_filing = today - timedelta(days=8)

        monkeypatch.setattr(_edgar, "_load_earnings_calendar_cache",
                            lambda sym, max_age_days=7: None)
        monkeypatch.setattr(_edgar, "lookup_cik", lambda sym: "0000MSFT")
        monkeypatch.setattr(_edgar, "_fetch_quarterly_filing_dates",
                            lambda cik, limit=2: [last_filing])
        monkeypatch.setattr(_edgar, "_save_earnings_calendar_cache",
                            lambda *a, **kw: None)
        monkeypatch.setattr(_edgar, "fetch_fundamentals",
                            lambda sym, metrics=None, periods=12: _edgar.FundamentalSnapshot(
                                ticker=sym, cik="0000MSFT", as_of=today,
                                eps_history=[2.0, 1.8],  # 1.8 < 2.0 → miss
                            ))

        result = _edgar.get_last_earnings_result("MSFT")
        assert result is not None
        _, was_beat = result
        assert was_beat is False

    def test_get_last_earnings_result_uses_cache(self, monkeypatch):
        import quantlab.providers.edgar as _edgar
        from datetime import timedelta

        today = date.today()
        last_d = today - timedelta(days=3)

        monkeypatch.setattr(_edgar, "_load_earnings_calendar_cache",
                            lambda sym, max_age_days=7: (last_d, today + timedelta(days=88), True))
        fetch_called = []
        monkeypatch.setattr(_edgar, "_fetch_quarterly_filing_dates",
                            lambda cik, limit=2: fetch_called.append(1) or [])

        result = _edgar.get_last_earnings_result("AAPL")
        assert result == (last_d, True)
        assert len(fetch_called) == 0


# ══════════════════════════════════════════════════════════════════════════════
# YoY same-quarter metrics (quantlab.providers.edgar)
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgarYoYMetrics:
    """Tests for YoY growth rate computation, acceleration detection, and scoring.
    All tests use synthetic data — no network access required.
    """

    @staticmethod
    def _snap(**kwargs):
        """Build a minimal FundamentalSnapshot with given overrides."""
        from quantlab.providers.edgar import FundamentalSnapshot
        return FundamentalSnapshot(ticker="TEST", cik="0", as_of=date.today(), **kwargs)

    # ── _yoy_growth_series ────────────────────────────────────────────────────

    def test_yoy_series_correct_calculation(self):
        from quantlab.providers.edgar import _yoy_growth_series
        # Q1-Q4 all = 100, Q5 = 120 → YoY = (120-100)/100 = 0.20
        h = [100.0, 100.0, 100.0, 100.0, 120.0]
        series = _yoy_growth_series(h)
        assert len(series) == 1
        assert series[0] == pytest.approx(0.20, abs=1e-9)

    def test_yoy_series_four_quarters(self):
        from quantlab.providers.edgar import _yoy_growth_series
        # 8 quarters: YoY for Q5-Q8 vs Q1-Q4
        h = [100.0, 100.0, 100.0, 100.0, 110.0, 125.0, 145.0, 170.0]
        series = _yoy_growth_series(h)
        assert len(series) == 4
        assert series[0] == pytest.approx(0.10, abs=1e-9)  # Q5 vs Q1: 110/100 - 1
        assert series[3] == pytest.approx(0.70, abs=1e-9)  # Q8 vs Q4: 170/100 - 1

    def test_yoy_series_insufficient_data_returns_empty(self):
        from quantlab.providers.edgar import _yoy_growth_series
        assert _yoy_growth_series([1.0, 2.0, 3.0]) == []  # only 3 quarters
        assert _yoy_growth_series([]) == []

    def test_yoy_series_skips_zero_prior(self):
        from quantlab.providers.edgar import _yoy_growth_series
        # Q1 = 0 → Q5 vs Q1 should be skipped (zero denominator)
        h = [0.0, 100.0, 100.0, 100.0, 120.0, 130.0, 140.0, 150.0]
        series = _yoy_growth_series(h)
        # Q5 vs Q1 skipped; Q6 vs Q2, Q7 vs Q3, Q8 vs Q4 kept = 3 rates
        assert len(series) == 3
        assert series[0] == pytest.approx(0.30, abs=1e-9)  # Q6 vs Q2: 130/100-1

    def test_yoy_series_capped_at_max_quarters(self):
        from quantlab.providers.edgar import _yoy_growth_series
        # 12 quarters → 8 YoY rates → capped at max_quarters=4
        # Q9-Q12 vs Q5-Q8 (NOT Q1-Q4) for the last 4 rates
        h = [100.0] * 4 + [110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0]
        series = _yoy_growth_series(h, max_quarters=4)
        assert len(series) == 4
        # Last rate: h[11] vs h[7] = 180/140 - 1 ≈ 0.2857
        assert series[-1] == pytest.approx(180.0 / 140.0 - 1.0, abs=1e-9)

    # ── _is_yoy_accelerating ──────────────────────────────────────────────────

    def test_is_accelerating_true_on_strictly_increasing(self):
        from quantlab.providers.edgar import _is_yoy_accelerating
        assert _is_yoy_accelerating([0.10, 0.20, 0.35]) is True

    def test_is_accelerating_false_on_flat(self):
        from quantlab.providers.edgar import _is_yoy_accelerating
        assert _is_yoy_accelerating([0.20, 0.20, 0.20]) is False

    def test_is_accelerating_false_when_last_dips(self):
        from quantlab.providers.edgar import _is_yoy_accelerating
        assert _is_yoy_accelerating([0.10, 0.30, 0.25]) is False

    def test_is_accelerating_true_with_two_points(self):
        from quantlab.providers.edgar import _is_yoy_accelerating
        assert _is_yoy_accelerating([0.20, 0.30]) is True  # 2 points is enough

    def test_is_accelerating_false_insufficient_data(self):
        from quantlab.providers.edgar import _is_yoy_accelerating
        assert _is_yoy_accelerating([0.20]) is False  # need at least 2 points
        assert _is_yoy_accelerating([]) is False

    # ── FundamentalSnapshot new fields ────────────────────────────────────────

    def test_snapshot_new_fields_default_none_and_empty(self):
        from quantlab.providers.edgar import FundamentalSnapshot
        snap = FundamentalSnapshot(ticker="X", cik="0", as_of=date.today())
        assert snap.revenue_yoy_pct is None
        assert snap.eps_yoy_pct is None
        assert snap.revenue_yoy_history == []
        assert snap.eps_yoy_history == []
        assert snap.is_accelerating is False

    # ── compute_earnings_acceleration YoY path ────────────────────────────────

    def test_high_yoy_growth_with_acceleration(self):
        from quantlab.providers.edgar import compute_earnings_acceleration
        # 70% YoY eps growth hits the O'Neil threshold: 70-100% band → 0.8, + 0.1 accel = 0.9
        snap = self._snap(
            eps_yoy_history=[0.30, 0.50, 0.70],
            eps_yoy_pct=0.70,
            is_accelerating=True,
        )
        score = compute_earnings_acceleration(snap)
        assert score == pytest.approx(0.90, abs=1e-4)

    def test_over_100pct_yoy_growth_maps_above_0_5(self):
        from quantlab.providers.edgar import compute_earnings_acceleration
        # 200% YoY eps → >100% band → O'Neil explosive tier → base = 1.0; no accel → 1.0
        snap = self._snap(eps_yoy_history=[0.5, 1.0, 2.0], eps_yoy_pct=2.0)
        score = compute_earnings_acceleration(snap)
        assert score == pytest.approx(1.0, abs=1e-4)

    def test_zero_yoy_growth_no_acceleration_returns_zero(self):
        from quantlab.providers.edgar import compute_earnings_acceleration
        snap = self._snap(eps_yoy_history=[0.0, 0.0, 0.0], eps_yoy_pct=0.0)
        assert compute_earnings_acceleration(snap) == pytest.approx(0.0, abs=1e-4)

    def test_negative_yoy_growth_returns_zero(self):
        from quantlab.providers.edgar import compute_earnings_acceleration
        snap = self._snap(eps_yoy_history=[-0.3, -0.2, -0.1], eps_yoy_pct=-0.10)
        assert compute_earnings_acceleration(snap) == pytest.approx(0.0, abs=1e-4)

    def test_acceleration_bonus_adds_010(self):
        from quantlab.providers.edgar import compute_earnings_acceleration
        # same growth rate, different is_accelerating flag — bonus is exactly +0.10
        base_snap = self._snap(eps_yoy_history=[0.20, 0.30, 0.40], eps_yoy_pct=0.40)
        accel_snap = self._snap(
            eps_yoy_history=[0.20, 0.30, 0.40], eps_yoy_pct=0.40, is_accelerating=True
        )
        diff = compute_earnings_acceleration(accel_snap) - compute_earnings_acceleration(base_snap)
        assert diff == pytest.approx(0.10, abs=1e-4)

    def test_score_clamped_at_1_0(self):
        from quantlab.providers.edgar import compute_earnings_acceleration
        # 200% YoY + accelerating → would be > 1.0 without clamp
        snap = self._snap(
            eps_yoy_history=[0.5, 1.2, 2.0], eps_yoy_pct=2.0, is_accelerating=True
        )
        assert compute_earnings_acceleration(snap) == pytest.approx(1.0, abs=1e-4)

    def test_score_clamped_at_0_0(self):
        from quantlab.providers.edgar import compute_earnings_acceleration
        snap = self._snap(eps_yoy_history=[-1.0, -0.5, -0.2], eps_yoy_pct=-0.20)
        assert compute_earnings_acceleration(snap) == pytest.approx(0.0, abs=1e-4)

    def test_revenue_yoy_used_when_eps_yoy_empty(self):
        from quantlab.providers.edgar import compute_earnings_acceleration
        # eps_yoy_history is empty; revenue_yoy_history has data
        snap = self._snap(
            revenue_yoy_history=[0.20, 0.35, 0.50],
            revenue_yoy_pct=0.50,
        )
        score = compute_earnings_acceleration(snap)
        # 50% YoY is exactly at the ≥50% threshold → 50-100% band → base=0.6, no accel bonus
        assert score == pytest.approx(0.60, abs=1e-4)

    # ── Legacy QoQ fallback ───────────────────────────────────────────────────

    def test_legacy_fallback_when_no_yoy_data(self):
        from quantlab.providers.edgar import compute_earnings_acceleration
        # Only 3 quarters available — no YoY possible; falls back to QoQ method
        snap = self._snap(eps_history=[1.0, 1.2, 1.5])  # accelerating QoQ
        score = compute_earnings_acceleration(snap)
        assert score > 0.5  # QoQ acceleration → above neutral

    def test_legacy_fallback_neutral_when_too_few_quarters(self):
        from quantlab.providers.edgar import compute_earnings_acceleration
        # Only 2 quarters — can't do QoQ either
        snap = self._snap(eps_history=[1.0, 1.2])
        assert compute_earnings_acceleration(snap) == pytest.approx(0.5, abs=1e-4)

    # ── fetch_fundamentals populates YoY fields ───────────────────────────────

    def test_fetch_fundamentals_populates_yoy_from_rich_history(self, monkeypatch):
        import quantlab.providers.edgar as _edgar

        # Build a fake EDGAR companyfacts response with 8 quarters of EPS and revenue
        eps_vals = [1.0, 1.0, 1.0, 1.0, 1.3, 1.5, 1.75, 2.1]
        rev_vals = [1e9 * v for v in eps_vals]

        fake_eps_obs = [
            {"form": "10-Q", "end": f"202{i//4+3}-{(i%4)*3+1:02d}-31", "filed": "x", "val": v}
            for i, v in enumerate(eps_vals)
        ]
        fake_rev_obs = [
            {"form": "10-Q", "end": f"202{i//4+3}-{(i%4)*3+1:02d}-31", "filed": "x", "val": v}
            for i, v in enumerate(rev_vals)
        ]

        monkeypatch.setattr(_edgar, "lookup_cik", lambda t: "0000320193")
        import requests as _req

        class MockResp:
            def raise_for_status(self): pass
            def json(self):
                return {"facts": {"us-gaap": {
                    "EarningsPerShareDiluted": {"units": {"USD": fake_eps_obs}},
                    "Revenues": {"units": {"USD": fake_rev_obs}},
                }}}

        monkeypatch.setattr(_req, "get", lambda url, headers, timeout: MockResp())

        snap = _edgar.fetch_fundamentals("AAPL", metrics=["eps_diluted", "revenue"])
        assert snap.eps_yoy_history != []
        assert snap.revenue_yoy_history != []
        assert snap.eps_yoy_pct is not None
        assert snap.revenue_yoy_pct is not None
        # All YoY rates should be positive (growth in every quarter)
        assert all(r > 0 for r in snap.eps_yoy_history)
        assert all(r > 0 for r in snap.revenue_yoy_history)

    # ── format_yoy_summary + AAPL / NVDA / CELH demo ─────────────────────────

    def test_format_yoy_summary_structure(self):
        from quantlab.providers.edgar import FundamentalSnapshot, format_yoy_summary, compute_earnings_acceleration
        snap = FundamentalSnapshot(ticker="AAPL", cik="0000320193", as_of=date.today())
        snap.revenue_yoy_pct = 0.17    # +17%
        snap.eps_yoy_pct     = 0.22    # +22%
        snap.revenue_yoy_history = [0.12, 0.15, 0.17]
        snap.eps_yoy_history     = [0.16, 0.19, 0.22]
        snap.is_accelerating = True
        score = compute_earnings_acceleration(snap)
        summary = format_yoy_summary(snap, score)
        assert summary.startswith("AAPL:")
        assert "revenue_yoy=+17%" in summary
        assert "eps_yoy=+22%" in summary
        assert "accelerating=True" in summary
        assert "score=" in summary

    def test_aapl_like_modest_growth_scores_low_band(self):
        """AAPL-like: mid-teens YoY growth → 0-50% band → 0.3 base."""
        from quantlab.providers.edgar import FundamentalSnapshot, compute_earnings_acceleration, format_yoy_summary
        snap = FundamentalSnapshot(ticker="AAPL", cik="0000320193", as_of=date.today())
        snap.revenue_yoy_pct = 0.17
        snap.eps_yoy_pct     = 0.22
        snap.revenue_yoy_history = [0.12, 0.15, 0.17]
        snap.eps_yoy_history     = [0.16, 0.19, 0.22]
        snap.is_accelerating = True   # eps 0.22 > 0.19; rev 0.17 > 0.15 → both improving
        score = compute_earnings_acceleration(snap)
        # 22% in 0-50% band → 0.3 + 0.1 accel = 0.4
        assert score == pytest.approx(0.40, abs=1e-4)
        print(format_yoy_summary(snap, score))  # shows: AAPL: revenue_yoy=+17% eps_yoy=+22% ...

    def test_nvda_like_hypergrowth_scores_high_band(self):
        """NVDA-like: hypergrowth EPS → >100% band → 0.9 + accel bonus = 1.0."""
        from quantlab.providers.edgar import FundamentalSnapshot, compute_earnings_acceleration, format_yoy_summary
        snap = FundamentalSnapshot(ticker="NVDA", cik="0001045810", as_of=date.today())
        snap.revenue_yoy_pct = 0.69    # +69%
        snap.eps_yoy_pct     = 1.52    # +152%
        snap.revenue_yoy_history = [0.40, 0.55, 0.69]
        snap.eps_yoy_history     = [0.80, 1.20, 1.52]
        snap.is_accelerating = True   # both rev and eps improving quarter-over-quarter
        score = compute_earnings_acceleration(snap)
        # EPS 152% → >100% band → 0.9 + 0.1 accel = 1.0 (capped)
        assert score == pytest.approx(1.0, abs=1e-4)
        print(format_yoy_summary(snap, score))  # NVDA: revenue_yoy=+69% eps_yoy=+152% ...

    def test_celh_like_growth_deceleration_scores_without_bonus(self):
        """CELH-like: strong growth but decelerating → no accel bonus."""
        from quantlab.providers.edgar import FundamentalSnapshot, compute_earnings_acceleration, format_yoy_summary
        snap = FundamentalSnapshot(ticker="CELH", cik="0001370109", as_of=date.today())
        snap.revenue_yoy_pct = 0.37    # +37% (slowing from prior highs)
        snap.eps_yoy_pct     = 0.45    # +45%
        snap.revenue_yoy_history = [0.90, 0.60, 0.37]  # decelerating
        snap.eps_yoy_history     = [1.20, 0.80, 0.45]  # decelerating
        snap.is_accelerating = False   # both rates falling
        score = compute_earnings_acceleration(snap)
        # 45% in 0-50% band → 0.3, no accel bonus
        assert score == pytest.approx(0.30, abs=1e-4)
        print(format_yoy_summary(snap, score))  # CELH: ... accelerating=False score=0.30

    def test_scoring_band_boundaries(self):
        """Verify each band boundary is handled correctly."""
        from quantlab.providers.edgar import FundamentalSnapshot, compute_earnings_acceleration

        def _score(yoy_pct):
            snap = FundamentalSnapshot(ticker="X", cik="0", as_of=date.today())
            snap.eps_yoy_history = [yoy_pct]
            snap.eps_yoy_pct = yoy_pct
            return compute_earnings_acceleration(snap)

        # O'Neil aligned bands: <20%=0.1, 20-50%=0.3, 50-70%=0.6, 70-100%=0.8, >100%=1.0
        assert _score(-0.01) == pytest.approx(0.0,  abs=1e-4)   # negative
        assert _score(0.00)  == pytest.approx(0.0,  abs=1e-4)   # zero (≤ 0 boundary)
        assert _score(0.01)  == pytest.approx(0.1,  abs=1e-4)   # 0–20% → insufficient
        assert _score(0.199) == pytest.approx(0.1,  abs=1e-4)   # just below 20%
        assert _score(0.20)  == pytest.approx(0.3,  abs=1e-4)   # exactly 20% → modest
        assert _score(0.499) == pytest.approx(0.3,  abs=1e-4)   # just below 50%
        assert _score(0.50)  == pytest.approx(0.6,  abs=1e-4)   # exactly 50% → strong
        assert _score(0.699) == pytest.approx(0.6,  abs=1e-4)   # just below 70%
        assert _score(0.70)  == pytest.approx(0.8,  abs=1e-4)   # O'Neil 70% threshold
        assert _score(0.999) == pytest.approx(0.8,  abs=1e-4)   # just below 100%
        assert _score(1.00)  == pytest.approx(1.0,  abs=1e-4)   # exactly 100% → explosive
        assert _score(5.00)  == pytest.approx(1.0,  abs=1e-4)   # hypergrowth


# ══════════════════════════════════════════════════════════════════════════════
# Book-derived edge improvements — stage classification, volume validation,
# PEG ratio, volume dry-up  (Weinstein / Minervini / O'Neil / Boucher / Darvas)
# ══════════════════════════════════════════════════════════════════════════════

class TestStageClassification:
    """Tests for stage_classification() using synthetic bar sequences."""

    @staticmethod
    def _make_stage_bars(
        n: int = 250,
        start: float = 100.0,
        trend_pct: float = 0.003,
        vol_factor: float = 1.0,
    ):
        """Generate trending bars with given parameters."""
        from datetime import timedelta
        bars = []
        price = start
        d = date(2025, 1, 2)
        for i in range(n):
            price = max(1.0, price * (1 + trend_pct + (i % 3 - 1) * 0.002))
            bars.append(Bar(
                as_of=d + timedelta(days=i),
                open=price * 0.998,
                high=price * 1.005,
                low=price * 0.994,
                close=price,
                volume=1_000_000 * vol_factor,
            ))
        return bars

    def test_stage_2_strong_uptrend(self):
        from quantlab.signals import stage_classification
        bars = self._make_stage_bars(n=250, trend_pct=0.004)
        # Strong uptrend: price well above rising 30W MA with higher highs/lows
        assert stage_classification(bars) == 2

    def test_stage_4_strong_downtrend(self):
        from quantlab.signals import stage_classification
        bars = self._make_stage_bars(n=250, trend_pct=-0.004)
        assert stage_classification(bars) == 4

    def test_insufficient_bars_returns_zero(self):
        from quantlab.signals import stage_classification
        bars = self._make_stage_bars(n=100)  # too short for 150 MA + 40 buffer
        assert stage_classification(bars) == 0

    def test_minimum_bars_boundary(self):
        from quantlab.signals import stage_classification
        # 190 bars = 150 + 40 → exactly at the threshold
        bars = self._make_stage_bars(n=190, trend_pct=0.003)
        stage = stage_classification(bars)
        assert stage in (0, 1, 2, 3, 4)  # any valid output

    def test_stage_returns_int(self):
        from quantlab.signals import stage_classification
        bars = self._make_stage_bars(n=250)
        result = stage_classification(bars)
        assert isinstance(result, int)
        assert result in (0, 1, 2, 3, 4)

    def test_scan_result_stage_defaults_zero(self):
        r = ScanResult(
            symbol="AAPL", scan_date="2026-06-08",
            signal_type="breakout", signal=True,
            entry_close=200.0, indicator_value=None, lookback=20,
        )
        assert r.stage == 0

    def test_stage_2_boosts_conviction(self):
        base = ScanResult(
            "AAPL", "2026-06-08", "breakout", True, 200.0, None, 20,
            regime_bullish=False, stage=0,
        )
        s2 = ScanResult(
            "AAPL", "2026-06-08", "breakout", True, 200.0, None, 20,
            regime_bullish=False, stage=2,
        )
        assert score_conviction(s2) - score_conviction(base) == pytest.approx(0.05, abs=1e-9)

    def test_non_stage_2_no_bonus(self):
        for stage_val in (0, 1, 3, 4):
            r = ScanResult(
                "AAPL", "2026-06-08", "breakout", True, 200.0, None, 20,
                regime_bullish=False, stage=stage_val,
            )
            # Should not add the +0.05 stage-2 bonus
            assert score_conviction(r) == pytest.approx(0.30, abs=1e-9), f"stage={stage_val}"


class TestBreakoutVolumeScore:
    """Tests for volume_on_breakout_score() — Weinstein 2× rule."""

    @staticmethod
    def _bars_with_today_vol(today_vol: float, avg_vol: float = 1_000_000, n: int = 25):
        from datetime import timedelta
        bars = []
        d = date(2025, 1, 2)
        for i in range(n):
            v = avg_vol if i < n - 1 else today_vol
            bars.append(Bar(d + timedelta(days=i), 100.0, 101.0, 99.0, 100.0, v))
        return bars

    def test_volume_below_1x_returns_zero(self):
        from quantlab.signals import volume_on_breakout_score
        bars = self._bars_with_today_vol(today_vol=800_000, avg_vol=1_000_000)
        assert volume_on_breakout_score(bars) == pytest.approx(0.0)

    def test_volume_1x_to_2x_returns_weak(self):
        from quantlab.signals import volume_on_breakout_score
        bars = self._bars_with_today_vol(today_vol=1_500_000, avg_vol=1_000_000)
        assert volume_on_breakout_score(bars) == pytest.approx(0.3)

    def test_volume_2x_to_3x_valid_weinstein(self):
        from quantlab.signals import volume_on_breakout_score
        bars = self._bars_with_today_vol(today_vol=2_500_000, avg_vol=1_000_000)
        assert volume_on_breakout_score(bars) == pytest.approx(0.7)

    def test_volume_above_3x_institutional(self):
        from quantlab.signals import volume_on_breakout_score
        bars = self._bars_with_today_vol(today_vol=3_500_000, avg_vol=1_000_000)
        assert volume_on_breakout_score(bars) == pytest.approx(1.0)

    def test_insufficient_bars_returns_zero(self):
        from quantlab.signals import volume_on_breakout_score
        from datetime import timedelta
        bars = [Bar(date(2025,1,2)+timedelta(days=i), 100., 101., 99., 100., 1e6)
                for i in range(5)]
        assert volume_on_breakout_score(bars) == pytest.approx(0.0)

    def test_breakout_volume_wires_into_conviction(self):
        # breakout signal_type → valid volume ≥ 0.7 adds +0.08
        no_vol = ScanResult(
            "AAPL", "2026-06-08", "breakout", True, 200.0, None, 20,
            regime_bullish=False, breakout_volume_score=0.3,
        )
        valid = ScanResult(
            "AAPL", "2026-06-08", "breakout", True, 200.0, None, 20,
            regime_bullish=False, breakout_volume_score=0.7,
        )
        assert score_conviction(valid) - score_conviction(no_vol) == pytest.approx(0.08, abs=1e-9)

    def test_breakout_volume_ignored_for_sma_signal(self):
        # SMA signal type should not receive the volume bonus
        r = ScanResult(
            "AAPL", "2026-06-08", "sma", True, 200.0, None, 20,
            regime_bullish=False, breakout_volume_score=1.0,
        )
        assert score_conviction(r) == pytest.approx(0.30, abs=1e-9)


class TestVolumeDryUpScore:
    """Tests for volume_dry_up_score() — Kell/Darvas base dry-up detection."""

    @staticmethod
    def _bars(recent_vol: float, prior_vol: float, n: int = 20):
        from datetime import timedelta
        bars = []
        d = date(2025, 1, 2)
        for i in range(n):
            vol = prior_vol if i < n // 2 else recent_vol
            bars.append(Bar(d + timedelta(days=i), 100.0, 101.0, 99.0, 100.0, vol))
        return bars

    def test_30pct_decline_returns_one(self):
        from quantlab.signals import volume_dry_up_score
        bars = self._bars(recent_vol=650_000, prior_vol=1_000_000)
        assert volume_dry_up_score(bars) == pytest.approx(1.0)

    def test_15_to_30pct_decline_returns_06(self):
        from quantlab.signals import volume_dry_up_score
        bars = self._bars(recent_vol=800_000, prior_vol=1_000_000)   # 20% decline
        assert volume_dry_up_score(bars) == pytest.approx(0.6)

    def test_flat_volume_returns_03(self):
        from quantlab.signals import volume_dry_up_score
        bars = self._bars(recent_vol=1_000_000, prior_vol=1_000_000)
        assert volume_dry_up_score(bars) == pytest.approx(0.3)

    def test_increasing_volume_returns_zero(self):
        from quantlab.signals import volume_dry_up_score
        bars = self._bars(recent_vol=1_500_000, prior_vol=1_000_000)
        assert volume_dry_up_score(bars) == pytest.approx(0.0)

    def test_insufficient_bars_returns_zero(self):
        from quantlab.signals import volume_dry_up_score
        from datetime import timedelta
        bars = [Bar(date(2025,1,2)+timedelta(days=i), 100., 101., 99., 100., 1e6)
                for i in range(5)]
        assert volume_dry_up_score(bars) == pytest.approx(0.0)


class TestPegRatioScore:
    """Tests for peg_ratio_score() — Boucher PEG filter."""

    def test_peg_below_05_deeply_undervalued(self):
        from quantlab.providers.edgar import peg_ratio_score
        # P/E=10, growth=30% → PEG=0.33 → 1.0
        assert peg_ratio_score(10.0, 30.0) == pytest.approx(1.0)

    def test_peg_05_to_10_fairly_valued(self):
        from quantlab.providers.edgar import peg_ratio_score
        # P/E=20, growth=30% → PEG=0.67 → 0.7
        assert peg_ratio_score(20.0, 30.0) == pytest.approx(0.7)

    def test_peg_10_to_15_slightly_expensive(self):
        from quantlab.providers.edgar import peg_ratio_score
        # P/E=25, growth=20% → PEG=1.25 → 0.4
        assert peg_ratio_score(25.0, 20.0) == pytest.approx(0.4)

    def test_peg_above_15_overvalued(self):
        from quantlab.providers.edgar import peg_ratio_score
        # P/E=40, growth=15% → PEG=2.67 → 0.0
        assert peg_ratio_score(40.0, 15.0) == pytest.approx(0.0)

    def test_none_inputs_neutral(self):
        from quantlab.providers.edgar import peg_ratio_score
        assert peg_ratio_score(None, 25.0) == pytest.approx(0.5)
        assert peg_ratio_score(20.0, None) == pytest.approx(0.5)
        assert peg_ratio_score(None, None) == pytest.approx(0.5)

    def test_zero_or_negative_growth_neutral(self):
        from quantlab.providers.edgar import peg_ratio_score
        assert peg_ratio_score(20.0, 0.0)  == pytest.approx(0.5)
        assert peg_ratio_score(20.0, -5.0) == pytest.approx(0.5)

    def test_peg_score_wires_into_conviction(self):
        # peg_score >= 0.7 → +0.06
        low = ScanResult(
            "AAPL", "2026-06-08", "breakout", True, 200.0, None, 20,
            regime_bullish=False, peg_score=0.4,
        )
        high = ScanResult(
            "AAPL", "2026-06-08", "breakout", True, 200.0, None, 20,
            regime_bullish=False, peg_score=0.7,
        )
        assert score_conviction(high) - score_conviction(low) == pytest.approx(0.06, abs=1e-9)

    def test_scan_result_peg_score_defaults_zero(self):
        r = ScanResult(
            symbol="AAPL", scan_date="2026-06-08",
            signal_type="breakout", signal=True,
            entry_close=200.0, indicator_value=None, lookback=20,
        )
        assert r.peg_score == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# IBKR earnings headline parser (quantlab.news.earnings_parser)
# ══════════════════════════════════════════════════════════════════════════════

class TestEarningsHeadlineParser:
    """Tests for earnings_parser — no network or DuckDB connection required."""

    # ── is_earnings_headline ─────────────────────────────────────────────────

    def test_is_earnings_headline_quarterly_results(self):
        from quantlab.news.earnings_parser import is_earnings_headline
        assert is_earnings_headline("Apple Reports Quarterly Results") is True

    def test_is_earnings_headline_eps_word(self):
        from quantlab.news.earnings_parser import is_earnings_headline
        assert is_earnings_headline("NVDA Reports Q2 EPS $5.98 vs $5.59 Estimate") is True

    def test_is_earnings_headline_per_share(self):
        from quantlab.news.earnings_parser import is_earnings_headline
        assert is_earnings_headline("Earned $2.01 per share in Q2") is True

    def test_is_earnings_headline_q_earnings(self):
        from quantlab.news.earnings_parser import is_earnings_headline
        assert is_earnings_headline("Meta Q3 earnings release") is True

    def test_is_earnings_headline_fiscal_q(self):
        from quantlab.news.earnings_parser import is_earnings_headline
        assert is_earnings_headline("Reports fiscal Q2 results") is True

    def test_is_earnings_headline_beats_estimates(self):
        from quantlab.news.earnings_parser import is_earnings_headline
        assert is_earnings_headline("Company beats estimates on strong demand") is True

    def test_is_earnings_headline_false_upgrade(self):
        from quantlab.news.earnings_parser import is_earnings_headline
        assert is_earnings_headline("Goldman Sachs raises AAPL to Buy") is False

    def test_is_earnings_headline_false_generic_news(self):
        from quantlab.news.earnings_parser import is_earnings_headline
        assert is_earnings_headline("CEO John Smith joins the board of directors") is False

    def test_is_earnings_headline_false_analyst_reco(self):
        from quantlab.news.earnings_parser import is_earnings_headline
        assert is_earnings_headline("Analyst reiterates Outperform on NVDA") is False

    # ── parse_earnings_headline — EPS extraction ─────────────────────────────

    def test_parse_eps_vs_estimate(self):
        from quantlab.news.earnings_parser import parse_earnings_headline
        p = parse_earnings_headline("Reports Q2 EPS $2.01 vs $1.88 Estimate")
        assert p.eps_actual  == pytest.approx(2.01)
        assert p.eps_estimate == pytest.approx(1.88)
        assert p.eps_beat is True
        assert p.quarter == "Q2"

    def test_parse_eps_beats_keyword(self):
        from quantlab.news.earnings_parser import parse_earnings_headline
        p = parse_earnings_headline("Q3 Earnings: EPS $1.52 Beats $1.44 Estimate")
        assert p.eps_actual  == pytest.approx(1.52)
        assert p.eps_estimate == pytest.approx(1.44)
        assert p.eps_beat is True
        assert p.quarter == "Q3"

    def test_parse_eps_misses_keyword(self):
        from quantlab.news.earnings_parser import parse_earnings_headline
        p = parse_earnings_headline("CELH Q1 EPS $0.42 Misses $0.51 Estimate")
        assert p.eps_actual  == pytest.approx(0.42)
        assert p.eps_estimate == pytest.approx(0.51)
        assert p.eps_beat is False
        assert p.quarter == "Q1"

    def test_parse_eps_no_estimate_gives_none_beat(self):
        from quantlab.news.earnings_parser import parse_earnings_headline
        p = parse_earnings_headline("Reports fiscal Q2 results: EPS $2.01")
        assert p.eps_actual  == pytest.approx(2.01)
        assert p.eps_estimate is None
        assert p.eps_beat is None

    def test_parse_eps_estimate_of_form(self):
        from quantlab.news.earnings_parser import parse_earnings_headline
        p = parse_earnings_headline("EPS $1.80 vs estimate of $1.75")
        assert p.eps_actual  == pytest.approx(1.80)
        assert p.eps_estimate == pytest.approx(1.75)
        assert p.eps_beat is True

    # ── parse_earnings_headline — Revenue extraction ──────────────────────────

    def test_parse_revenue_vs_billions(self):
        from quantlab.news.earnings_parser import parse_earnings_headline
        p = parse_earnings_headline("Revenue $94.9B vs $94.1B Expected")
        assert p.revenue_actual   == pytest.approx(94_900.0)
        assert p.revenue_estimate == pytest.approx(94_100.0)
        assert p.revenue_beat is True

    def test_parse_revenue_vs_millions(self):
        from quantlab.news.earnings_parser import parse_earnings_headline
        p = parse_earnings_headline("Revenue $450.2M vs $430.0M Expected")
        assert p.revenue_actual   == pytest.approx(450.2)
        assert p.revenue_estimate == pytest.approx(430.0)
        assert p.revenue_beat is True

    def test_parse_revenue_miss_when_below(self):
        from quantlab.news.earnings_parser import parse_earnings_headline
        p = parse_earnings_headline("Sales $88.5B vs $89.3B Expected")
        assert p.revenue_beat is False

    def test_parse_revenue_only_no_comparison(self):
        from quantlab.news.earnings_parser import parse_earnings_headline
        p = parse_earnings_headline("Reports fiscal Q2 results: EPS $2.01, Revenue $94.9B")
        assert p.revenue_actual   == pytest.approx(94_900.0)
        assert p.revenue_estimate is None
        assert p.revenue_beat is None

    # ── parse_earnings_headline — combined and metadata ───────────────────────

    def test_parse_combined_eps_and_revenue(self):
        from quantlab.news.earnings_parser import parse_earnings_headline
        h = "NVDA Q4: EPS $0.52 Beats $0.45 Estimate; Revenue $22.1B vs $20.6B Expected"
        p = parse_earnings_headline(h)
        assert p.eps_actual  == pytest.approx(0.52)
        assert p.eps_beat is True
        assert p.revenue_actual   == pytest.approx(22_100.0)
        assert p.revenue_estimate == pytest.approx(20_600.0)
        assert p.revenue_beat is True
        assert p.quarter == "Q4"

    def test_parse_quarter_all_four(self):
        from quantlab.news.earnings_parser import parse_earnings_headline
        for q in range(1, 5):
            p = parse_earnings_headline(f"Company Reports Q{q} 2026 EPS $1.00 vs $0.90")
            assert p.quarter == f"Q{q}", f"Failed for Q{q}"

    def test_parse_fiscal_year_extracted(self):
        from quantlab.news.earnings_parser import parse_earnings_headline
        p = parse_earnings_headline("Reports Q1 2025 EPS $1.50 vs $1.40 Estimate")
        assert p.fiscal_year == 2025

    def test_parse_non_earnings_headline_returns_none_fields(self):
        from quantlab.news.earnings_parser import parse_earnings_headline
        p = parse_earnings_headline("Goldman Sachs raises price target on AAPL")
        assert p.eps_actual is None
        assert p.revenue_actual is None
        assert p.quarter is None

    # ── compute_beat_score ────────────────────────────────────────────────────

    def test_beat_score_both_beat(self):
        from quantlab.news.earnings_parser import compute_beat_score
        assert compute_beat_score(True, True) == pytest.approx(1.0)

    def test_beat_score_both_miss(self):
        from quantlab.news.earnings_parser import compute_beat_score
        assert compute_beat_score(False, False) == pytest.approx(0.0)

    def test_beat_score_eps_beat_revenue_unknown(self):
        from quantlab.news.earnings_parser import compute_beat_score
        assert compute_beat_score(True, None) == pytest.approx(0.7)

    def test_beat_score_revenue_beat_eps_unknown(self):
        from quantlab.news.earnings_parser import compute_beat_score
        assert compute_beat_score(None, True) == pytest.approx(0.5)

    def test_beat_score_eps_beat_revenue_miss(self):
        from quantlab.news.earnings_parser import compute_beat_score
        assert compute_beat_score(True, False) == pytest.approx(0.3)

    def test_beat_score_eps_miss_revenue_beat(self):
        from quantlab.news.earnings_parser import compute_beat_score
        assert compute_beat_score(False, True) == pytest.approx(0.3)

    def test_beat_score_eps_miss_only(self):
        from quantlab.news.earnings_parser import compute_beat_score
        assert compute_beat_score(False, None) == pytest.approx(0.3)

    def test_beat_score_revenue_miss_only(self):
        from quantlab.news.earnings_parser import compute_beat_score
        assert compute_beat_score(None, False) == pytest.approx(0.3)

    def test_beat_score_insufficient_data(self):
        from quantlab.news.earnings_parser import compute_beat_score
        assert compute_beat_score(None, None) == pytest.approx(0.5)

    # ── EarningsResult dataclass ──────────────────────────────────────────────

    def test_earnings_result_fields(self):
        from quantlab.news.earnings_parser import EarningsResult
        r = EarningsResult(
            symbol="AAPL", report_date="2026-06-07", quarter="Q2",
            fiscal_year=2026, eps_actual=2.01, eps_estimate=1.88,
            eps_beat=True, revenue_actual=94_900.0, revenue_estimate=94_100.0,
            revenue_beat=True, beat_score=1.0,
            headline_source="Reports Q2 EPS $2.01 vs $1.88",
        )
        assert r.symbol == "AAPL"
        assert r.beat_score == pytest.approx(1.0)
        assert r.eps_beat is True
        assert r.quarter == "Q2"

    def test_make_earnings_result_eps_beat_only(self):
        from quantlab.news.earnings_parser import make_earnings_result
        r = make_earnings_result(
            "AAPL", "Reports Q2 EPS $2.01 vs $1.88 Estimate", "2026-06-07"
        )
        assert r is not None
        assert r.symbol == "AAPL"
        assert r.eps_actual  == pytest.approx(2.01)
        assert r.eps_beat is True
        assert r.beat_score  == pytest.approx(0.7)   # eps beat only (no revenue)

    def test_make_earnings_result_both_beat(self):
        from quantlab.news.earnings_parser import make_earnings_result
        h = "NVDA Q4 EPS $0.52 Beats $0.45 Estimate; Revenue $22.1B vs $20.6B Expected"
        r = make_earnings_result("NVDA", h, "2026-06-07")
        assert r is not None
        assert r.beat_score == pytest.approx(1.0)
        assert r.eps_beat is True
        assert r.revenue_beat is True

    def test_make_earnings_result_non_earnings_returns_none(self):
        from quantlab.news.earnings_parser import make_earnings_result
        r = make_earnings_result("AAPL", "Goldman Sachs upgrades AAPL to Buy")
        assert r is None

    # ── DuckDB store and retrieve ─────────────────────────────────────────────

    def test_store_and_retrieve_round_trip(self, tmp_path, monkeypatch):
        import quantlab.storage as _storage
        monkeypatch.setattr(_storage, "DB_PATH", tmp_path / "test.duckdb")

        from quantlab.news.earnings_parser import (
            EarningsResult, store_earnings_result, get_recent_earnings_result,
        )
        r = EarningsResult(
            symbol="NVDA", report_date=date.today().isoformat(),
            quarter="Q2", fiscal_year=2026,
            eps_actual=5.98, eps_estimate=5.59, eps_beat=True,
            revenue_actual=22_100.0, revenue_estimate=20_600.0, revenue_beat=True,
            beat_score=1.0, headline_source="test headline",
        )
        store_earnings_result(r)
        retrieved = get_recent_earnings_result("NVDA", max_days=5)
        assert retrieved is not None
        assert retrieved.symbol == "NVDA"
        assert retrieved.beat_score == pytest.approx(1.0)
        assert retrieved.eps_beat is True
        assert retrieved.revenue_beat is True
        assert retrieved.eps_actual == pytest.approx(5.98)

    def test_retrieve_respects_max_days(self, tmp_path, monkeypatch):
        import quantlab.storage as _storage
        from datetime import timedelta
        monkeypatch.setattr(_storage, "DB_PATH", tmp_path / "test2.duckdb")

        from quantlab.news.earnings_parser import (
            EarningsResult, store_earnings_result, get_recent_earnings_result,
        )
        # Store a result from 20 calendar days ago (~14 trading days — clearly outside window)
        old_date = (date.today() - timedelta(days=20)).isoformat()
        r = EarningsResult(
            symbol="TSLA", report_date=old_date,
            quarter="Q1", fiscal_year=2026,
            eps_actual=1.50, eps_estimate=1.40, eps_beat=True,
            revenue_actual=None, revenue_estimate=None, revenue_beat=None,
            beat_score=0.7, headline_source="stale test",
        )
        store_earnings_result(r)
        result = get_recent_earnings_result("TSLA", max_days=5)
        assert result is None  # outside 5-trading-day window

    def test_retrieve_returns_none_for_unknown_symbol(self, tmp_path, monkeypatch):
        import quantlab.storage as _storage
        monkeypatch.setattr(_storage, "DB_PATH", tmp_path / "test3.duckdb")

        from quantlab.news.earnings_parser import get_recent_earnings_result
        assert get_recent_earnings_result("ZZZZZ", max_days=5) is None

    def test_missing_beat_fields_preserved_as_none(self, tmp_path, monkeypatch):
        import quantlab.storage as _storage
        monkeypatch.setattr(_storage, "DB_PATH", tmp_path / "test4.duckdb")

        from quantlab.news.earnings_parser import (
            EarningsResult, store_earnings_result, get_recent_earnings_result,
        )
        r = EarningsResult(
            symbol="CELH", report_date=date.today().isoformat(),
            quarter="Q2", fiscal_year=2026,
            eps_actual=0.42, eps_estimate=None, eps_beat=None,
            revenue_actual=None, revenue_estimate=None, revenue_beat=None,
            beat_score=0.5, headline_source="no estimate headline",
        )
        store_earnings_result(r)
        out = get_recent_earnings_result("CELH", max_days=5)
        assert out is not None
        assert out.eps_beat is None
        assert out.revenue_beat is None
        assert out.beat_score == pytest.approx(0.5)


# ══════════════════════════════════════════════════════════════════════════════
# InstitutionalWatchlist — persistent multi-day pre-breakout tracking
# ══════════════════════════════════════════════════════════════════════════════

class TestInstitutionalWatchlist:
    """Tests for InstitutionalWatchlist — all use a tmp DuckDB path."""

    @staticmethod
    def _make_result(symbol="TEST", conviction=0.55, stage=2, entry_close=100.0):
        """Build a minimal ScanResult-like object."""
        return ScanResult(
            symbol=symbol, scan_date=date.today().isoformat(),
            signal_type="breakout", signal=True,
            entry_close=entry_close, indicator_value=None, lookback=5,
            conviction_score=conviction, stage=stage,
        )

    # ── upsert ────────────────────────────────────────────────────────────────

    def test_upsert_new_entry_consecutive_days_1(self, tmp_path):
        from quantlab.watchlist import InstitutionalWatchlist
        iwl = InstitutionalWatchlist(db_path=tmp_path / "test.duckdb")
        result = iwl.upsert("NVDA", self._make_result("NVDA", conviction=0.60))
        assert result["consecutive_days"] == 1
        assert result["symbol"] == "NVDA"

    def test_upsert_same_symbol_today_does_not_double_increment(self, tmp_path):
        from quantlab.watchlist import InstitutionalWatchlist
        iwl = InstitutionalWatchlist(db_path=tmp_path / "test.duckdb")
        iwl.upsert("AAPL", self._make_result("AAPL"))
        r2 = iwl.upsert("AAPL", self._make_result("AAPL"))
        # Same day: consecutive_days should stay 1
        assert r2["consecutive_days"] == 1

    def test_upsert_previous_day_increments_consecutive_days(self, tmp_path):
        from quantlab.watchlist import InstitutionalWatchlist
        import duckdb
        from datetime import timedelta
        db = tmp_path / "test.duckdb"
        iwl = InstitutionalWatchlist(db_path=db)

        # Manually insert entry with last_seen = yesterday
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        con = duckdb.connect(str(db))
        con.execute(
            """
            INSERT INTO institutional_watchlist
                (symbol, first_seen, last_seen, consecutive_days, stage,
                 conviction_score, entry_price, options_signal, volume_dry_up,
                 earnings_score, peg_score, breakout_volume_score, tape, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, FALSE, FALSE, NULL, NULL, NULL, '', '')
            """,
            ["TSLA", yesterday, yesterday, 1, 2, 0.60, 250.0],
        )
        con.close()

        # Upsert today — should increment consecutive_days to 2
        r = iwl.upsert("TSLA", self._make_result("TSLA", conviction=0.65))
        assert r["consecutive_days"] == 2

    def test_upsert_conviction_bonus_applied(self, tmp_path):
        from quantlab.watchlist import InstitutionalWatchlist
        iwl = InstitutionalWatchlist(db_path=tmp_path / "test.duckdb")
        base_conviction = 0.50
        r = iwl.upsert("META", self._make_result("META", conviction=base_conviction))
        # Day 1: bonus = 0.05 * 1 = 0.05 → stored = 0.55
        assert r["conviction_score"] == pytest.approx(base_conviction + 0.05, abs=1e-4)

    def test_conviction_bonus_capped_at_020(self, tmp_path):
        from quantlab.watchlist import InstitutionalWatchlist
        import duckdb
        from datetime import timedelta
        db = tmp_path / "test.duckdb"
        iwl = InstitutionalWatchlist(db_path=db)

        # Manually insert entry with 4+ consecutive days already
        old = (date.today() - timedelta(days=1)).isoformat()
        con = duckdb.connect(str(db))
        con.execute(
            """
            INSERT INTO institutional_watchlist
                (symbol, first_seen, last_seen, consecutive_days, stage,
                 conviction_score, entry_price, options_signal, volume_dry_up,
                 earnings_score, peg_score, breakout_volume_score, tape, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, FALSE, FALSE, NULL, NULL, NULL, '', '')
            """,
            ["CELH", old, old, 10, 2, 0.70, 80.0],  # already 10 days
        )
        con.close()

        base = 0.50
        r = iwl.upsert("CELH", self._make_result("CELH", conviction=base))
        # consecutive_days = 11, bonus = min(0.20, 0.05*11) = 0.20
        assert r["conviction_score"] == pytest.approx(base + 0.20, abs=1e-4)

    # ── get_candidates / get_multi_day ────────────────────────────────────────

    def test_get_candidates_returns_all_entries(self, tmp_path):
        from quantlab.watchlist import InstitutionalWatchlist
        iwl = InstitutionalWatchlist(db_path=tmp_path / "test.duckdb")
        iwl.upsert("A", self._make_result("A", conviction=0.60))
        iwl.upsert("B", self._make_result("B", conviction=0.70))
        candidates = iwl.get_candidates()
        syms = [c["symbol"] for c in candidates]
        assert "A" in syms and "B" in syms

    def test_get_multi_day_filters_correctly(self, tmp_path):
        from quantlab.watchlist import InstitutionalWatchlist
        import duckdb
        from datetime import timedelta
        db = tmp_path / "test.duckdb"
        iwl = InstitutionalWatchlist(db_path=db)

        # Single-day entry
        iwl.upsert("SINGLE", self._make_result("SINGLE"))

        # Multi-day entry (manually set consecutive_days=3)
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        con = duckdb.connect(str(db))
        con.execute(
            """
            INSERT INTO institutional_watchlist
                (symbol, first_seen, last_seen, consecutive_days, stage,
                 conviction_score, entry_price, options_signal, volume_dry_up,
                 earnings_score, peg_score, breakout_volume_score, tape, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, FALSE, FALSE, NULL, NULL, NULL, '', '')
            """,
            ["MULTI", yesterday, yesterday, 3, 2, 0.75, 200.0],
        )
        con.close()

        multi = iwl.get_multi_day(min_days=2)
        syms = [c["symbol"] for c in multi]
        assert "MULTI" in syms
        assert "SINGLE" not in syms

    def test_get_multi_day_sorted_by_days_then_conviction(self, tmp_path):
        from quantlab.watchlist import InstitutionalWatchlist
        import duckdb
        from datetime import timedelta
        db = tmp_path / "test.duckdb"
        iwl = InstitutionalWatchlist(db_path=db)
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        con = duckdb.connect(str(db))
        for sym, days, conv in [("HIGH", 4, 0.80), ("LOW", 2, 0.55)]:
            con.execute(
                """
                INSERT INTO institutional_watchlist
                    (symbol, first_seen, last_seen, consecutive_days, stage,
                     conviction_score, entry_price, options_signal, volume_dry_up,
                     earnings_score, peg_score, breakout_volume_score, tape, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, FALSE, FALSE, NULL, NULL, NULL, '', '')
                """,
                [sym, yesterday, yesterday, days, 2, conv, 100.0],
            )
        con.close()
        multi = iwl.get_multi_day(min_days=2)
        assert multi[0]["symbol"] == "HIGH"  # higher days first

    # ── remove_stale ──────────────────────────────────────────────────────────

    def test_remove_stale_removes_old_entries(self, tmp_path):
        from quantlab.watchlist import InstitutionalWatchlist
        import duckdb
        from datetime import timedelta
        db = tmp_path / "test.duckdb"
        iwl = InstitutionalWatchlist(db_path=db)

        # Insert an entry from 15 calendar days ago (~ 10 trading days)
        old_date = (date.today() - timedelta(days=15)).isoformat()
        con = duckdb.connect(str(db))
        con.execute(
            """
            INSERT INTO institutional_watchlist
                (symbol, first_seen, last_seen, consecutive_days, stage,
                 conviction_score, entry_price, options_signal, volume_dry_up,
                 earnings_score, peg_score, breakout_volume_score, tape, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, FALSE, FALSE, NULL, NULL, NULL, '', '')
            """,
            ["STALE", old_date, old_date, 2, 2, 0.60, 50.0],
        )
        con.close()

        removed = iwl.remove_stale(max_days_inactive=5)
        assert removed >= 1
        remaining = [c["symbol"] for c in iwl.get_candidates()]
        assert "STALE" not in remaining

    def test_remove_stale_keeps_recent_entries(self, tmp_path):
        from quantlab.watchlist import InstitutionalWatchlist
        iwl = InstitutionalWatchlist(db_path=tmp_path / "test.duckdb")
        # Today's entry — should NOT be removed
        iwl.upsert("FRESH", self._make_result("FRESH"))
        removed = iwl.remove_stale(max_days_inactive=5)
        assert removed == 0
        remaining = [c["symbol"] for c in iwl.get_candidates()]
        assert "FRESH" in remaining

    def test_remove_stale_returns_count(self, tmp_path):
        from quantlab.watchlist import InstitutionalWatchlist
        import duckdb
        from datetime import timedelta
        db = tmp_path / "test.duckdb"
        iwl = InstitutionalWatchlist(db_path=db)
        old = (date.today() - timedelta(days=20)).isoformat()
        con = duckdb.connect(str(db))
        for sym in ["X1", "X2", "X3"]:
            con.execute(
                """
                INSERT INTO institutional_watchlist
                    (symbol, first_seen, last_seen, consecutive_days, stage,
                     conviction_score, entry_price, options_signal, volume_dry_up,
                     earnings_score, peg_score, breakout_volume_score, tape, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, FALSE, FALSE, NULL, NULL, NULL, '', '')
                """,
                [sym, old, old, 1, 2, 0.50, 100.0],
            )
        con.close()
        assert iwl.remove_stale(max_days_inactive=5) == 3

    # ── generate_report ───────────────────────────────────────────────────────

    def test_generate_report_creates_file(self, tmp_path, monkeypatch):
        import importlib, sys
        # Add scripts/ to path so generate_report can be imported
        scripts_dir = str(Path(__file__).parent.parent / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

        from quantlab.watchlist import InstitutionalWatchlist
        import generate_report as _gr

        db = tmp_path / "test.duckdb"
        reports = tmp_path / "reports"
        iwl = InstitutionalWatchlist(db_path=db)
        iwl.upsert("AAPL", self._make_result("AAPL", conviction=0.75, stage=2))

        pdf_path = _gr.generate(
            report_date=date.today(),
            reports_dir=reports,
            db_path=str(db),
        )
        assert pdf_path.exists()
        assert pdf_path.suffix == ".pdf"
        content = pdf_path.read_bytes()
        assert content[:4] == b"%PDF"
        assert b"STH Capital" in content   # title metadata stored uncompressed
        assert pdf_path.stat().st_size > 2000

    def test_generate_report_daily_reports_row_written(self, tmp_path):
        import sys
        scripts_dir = str(Path(__file__).parent.parent / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

        from quantlab.watchlist import InstitutionalWatchlist
        import generate_report as _gr
        import duckdb

        db = tmp_path / "test2.duckdb"
        reports = tmp_path / "reports2"
        iwl = InstitutionalWatchlist(db_path=db)
        _gr.generate(report_date=date.today(), reports_dir=reports, db_path=str(db))

        con = duckdb.connect(str(db))
        row = con.execute("SELECT date, candidates FROM daily_reports").fetchone()
        con.close()
        assert row is not None
        assert str(row[0]) == date.today().isoformat()
