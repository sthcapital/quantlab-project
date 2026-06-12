"""
Regression tests for the Weinstein breakout-volume ratio.

Background (2026-06-12): the nightly report showed "Brkout Vol: 0.30" for all
five highlighted candidates.  0.30 was the banded volume_on_breakout_score
(the entire "1–2× average" band — where nearly every pre-breakout candidate
sits) rendered in a field labeled like a ratio.  The raw ratio is now computed
by breakout_volume_ratio(), carried on ScanResult / the watchlist record, and
rendered with an explicit "x" suffix; None (not measurable) renders as "—".
"""

from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import pytest

from quantlab.providers.base import Bar
from quantlab.signals import breakout_volume_ratio, volume_on_breakout_score

_ROOT = Path(__file__).parent.parent


def _bars(volumes: list[float], price: float = 100.0) -> list[Bar]:
    """Flat-price bar series with the given per-bar volumes."""
    start = date(2026, 1, 2)
    return [
        Bar(
            as_of=start + timedelta(days=i),
            open=price, high=price * 1.01, low=price * 0.99, close=price,
            volume=v,
        )
        for i, v in enumerate(volumes)
    ]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ══════════════════════════════════════════════════════════════════════════════
# breakout_volume_ratio — raw Weinstein ratio
# ══════════════════════════════════════════════════════════════════════════════

class TestBreakoutVolumeRatio:

    def test_breakout_at_exactly_2x_base_average(self):
        """Breakout bar volume exactly 2× the 20-bar base average → ratio 2.0."""
        bars = _bars([1_000_000.0] * 20 + [2_000_000.0])
        assert breakout_volume_ratio(bars) == pytest.approx(2.0)

    def test_quiet_stock_ratio_near_1(self):
        """No volume expansion → ratio 1.0 (this is the 0.3-band regime that
        rendered as the misleading universal 'Brkout Vol: 0.30')."""
        bars = _bars([1_000_000.0] * 21)
        assert breakout_volume_ratio(bars) == pytest.approx(1.0)

    def test_insufficient_bars_returns_none(self):
        """Fewer than period+1 bars → not measurable → None, never a number."""
        bars = _bars([1_000_000.0] * 20)   # need 21 for period=20
        assert breakout_volume_ratio(bars) is None

    def test_empty_bars_returns_none(self):
        assert breakout_volume_ratio([]) is None

    def test_zero_base_volume_returns_none(self):
        bars = _bars([0.0] * 20 + [500_000.0])
        assert breakout_volume_ratio(bars) is None

    def test_breakout_bar_excluded_from_baseline(self):
        """The breakout bar itself must not inflate the base average."""
        bars = _bars([1_000_000.0] * 20 + [10_000_000.0])
        # baseline = 1M (not (20M+10M)/21) → ratio 10.0
        assert breakout_volume_ratio(bars) == pytest.approx(10.0)

    def test_score_bands_match_ratio(self):
        """The banded score and the raw ratio must stay in sync."""
        cases = [
            ([1_000_000.0] * 20 + [500_000.0], 0.0),    # 0.5x  — false breakout
            ([1_000_000.0] * 20 + [1_500_000.0], 0.3),  # 1.5x  — below minimum
            ([1_000_000.0] * 20 + [2_000_000.0], 0.7),  # 2.0x  — Weinstein valid
            ([1_000_000.0] * 20 + [3_500_000.0], 1.0),  # 3.5x  — institutional
        ]
        for volumes, expected in cases:
            assert volume_on_breakout_score(_bars(volumes)) == expected

    def test_score_zero_when_ratio_unmeasurable(self):
        assert volume_on_breakout_score(_bars([1_000_000.0] * 5)) == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Propagation — ScanResult → watchlist record
# ══════════════════════════════════════════════════════════════════════════════

class TestRatioPropagation:

    def test_scan_symbol_sets_ratio_on_result(self):
        from quantlab.execution import scan_symbol
        # 80 bars rising so the breakout signal machinery has enough history;
        # final bar carries a 2× volume spike
        price, bars = 100.0, []
        for i in range(80):
            price *= 1.004
            bars.append(Bar(
                as_of=date(2026, 1, 2) + timedelta(days=i),
                open=price * 0.999, high=price * 1.01,
                low=price * 0.99, close=price,
                volume=1_000_000.0,
            ))
        bars[-1] = Bar(
            as_of=bars[-1].as_of, open=bars[-1].open, high=bars[-1].high,
            low=bars[-1].low, close=bars[-1].close, volume=2_000_000.0,
        )
        result = scan_symbol("TEST", bars, "breakout", lookback=5)
        assert result is not None
        assert result.breakout_volume_ratio == pytest.approx(2.0)

    def test_upsert_persists_ratio_to_candidate_record(self, tmp_path):
        from quantlab.execution import ScanResult
        from quantlab.watchlist import InstitutionalWatchlist

        r = ScanResult(
            symbol="CELH", scan_date=date.today().isoformat(),
            signal_type="breakout", signal=True,
            entry_close=100.0, indicator_value=None, lookback=5,
            conviction_score=0.75, stage=2,
        )
        r.breakout_volume_ratio = 2.4

        iwl = InstitutionalWatchlist(db_path=tmp_path / "test.duckdb")
        iwl.upsert("CELH", r)
        entry = next(e for e in iwl.get_candidates() if e["symbol"] == "CELH")
        assert entry["breakout_volume_ratio"] == pytest.approx(2.4)

    def test_upsert_persists_null_when_not_measurable(self, tmp_path):
        from quantlab.execution import ScanResult
        from quantlab.watchlist import InstitutionalWatchlist

        r = ScanResult(
            symbol="NOBARS", scan_date=date.today().isoformat(),
            signal_type="breakout", signal=True,
            entry_close=100.0, indicator_value=None, lookback=5,
            conviction_score=0.75, stage=2,
        )
        # breakout_volume_ratio left at default None

        iwl = InstitutionalWatchlist(db_path=tmp_path / "test.duckdb")
        iwl.upsert("NOBARS", r)
        entry = next(e for e in iwl.get_candidates() if e["symbol"] == "NOBARS")
        assert entry["breakout_volume_ratio"] is None


# ══════════════════════════════════════════════════════════════════════════════
# Report rendering — no bar data must render "—", never a number
# ══════════════════════════════════════════════════════════════════════════════

class TestReportRendering:

    def test_none_renders_em_dash(self):
        gr = _load_script("generate_report")
        assert gr._bv_ratio(None) == "—"

    def test_ratio_renders_with_x_suffix(self):
        gr = _load_script("generate_report")
        assert gr._bv_ratio(2.0) == "2.0x"
        assert gr._bv_ratio(2.14) == "2.1x"
        assert gr._bv_ratio(0.97) == "1.0x"

    def test_garbage_renders_em_dash(self):
        gr = _load_script("generate_report")
        assert gr._bv_ratio("not-a-number") == "—"
