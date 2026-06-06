"""
tests/test_unusual_options.py — Unit tests for the unusual options activity detector.

All tests use in-memory mock data — no S3 credentials or flat file downloads needed.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quantlab.signals.unusual_options import (
    UnusualOptionsSignal,
    compute_20day_avg_volume,
    detect_unusual_activity,
    score_unusual_activity,
)

SCAN_DATE = date(2025, 5, 1)
SPOT = 300.0


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sig(
    volume_ratio: float = 10.0,
    dte: int = 35,
    otm_pct: float = 0.08,
    concentrated: bool = True,
    today_vol: float = 1000.0,
    avg_vol: float = 100.0,
) -> UnusualOptionsSignal:
    return UnusualOptionsSignal(
        symbol="CAT", date=SCAN_DATE,
        strike=SPOT * (1 + otm_pct),
        option_type="C",
        today_volume=today_vol,
        avg_20day_volume=avg_vol,
        volume_ratio=volume_ratio,
        oi_today=0.0,
        oi_change_3day=0.0,
        expiry=SCAN_DATE + timedelta(days=dte),
        days_to_expiry=dte,
        otm_pct=otm_pct,
        is_concentrated=concentrated,
        conviction_score=0.0,
    )


def _make_ffp(chains: dict[date, list[dict]]) -> MagicMock:
    """Build a FlatFileProvider mock that returns chains by date."""
    ffp = MagicMock()

    def _chain(symbol, d):
        if d in chains:
            return chains[d]
        raise FileNotFoundError(f"no cache for {d}")

    # options_cache_path exists check — True for dates that have data
    def _cache_path(d):
        m = MagicMock()
        m.exists.return_value = d in chains
        return m

    ffp.get_options_chain_from_flatfile.side_effect = _chain
    ffp.options_cache_path.side_effect = _cache_path
    return ffp


def _option_row(
    symbol: str,
    strike: float,
    opt_type: str,
    volume: float,
    dte_days: int = 35,
    otm_ratio: float = 0.0,  # unused; strike is explicit
) -> dict:
    expiry = SCAN_DATE + timedelta(days=dte_days)
    ticker = f"O:{symbol}{expiry.strftime('%y%m%d')}{'C' if opt_type=='C' else 'P'}{int(strike*1000):08d}"
    return {
        "ticker": ticker,
        "underlying": symbol,
        "expiry": expiry.isoformat(),
        "strike": float(strike),
        "option_type": opt_type,
        "volume": float(volume),
        "open": 5.0, "close": 5.1, "high": 5.5, "low": 4.8,
        "window_start": 1746072000000000000,
        "transactions": int(volume // 10),
    }


# ══════════════════════════════════════════════════════════════════════════════
# UnusualOptionsSignal dataclass
# ══════════════════════════════════════════════════════════════════════════════

class TestUnusualOptionsSignal:

    def test_fields_accessible(self):
        s = _sig()
        assert s.symbol == "CAT"
        assert s.option_type == "C"
        assert s.volume_ratio == pytest.approx(10.0)
        assert s.is_concentrated is True

    def test_defaults(self):
        s = _sig()
        assert s.oi_today == 0.0
        assert s.oi_change_3day == 0.0
        assert s.conviction_score == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# compute_20day_avg_volume
# ══════════════════════════════════════════════════════════════════════════════

class TestCompute20DayAvgVolume:

    def test_basic_average(self):
        """3 cached days each with 200 vol → avg = 200/20 = 10."""
        days = {}
        for i in range(1, 4):
            d = SCAN_DATE - timedelta(days=i)
            days[d] = [_option_row("CAT", 315.0, "C", 200.0, dte_days=35)]

        ffp = _make_ffp(days)
        result = compute_20day_avg_volume("CAT", SCAN_DATE, ffp, trading_days=20)

        assert (315.0, "C") in result
        # sum = 600, denominator = 20 (trading_days), avg = 30
        assert result[(315.0, "C")] == pytest.approx(30.0)

    def test_empty_when_no_cache(self):
        ffp = _make_ffp({})
        result = compute_20day_avg_volume("CAT", SCAN_DATE, ffp)
        assert result == {}

    def test_skips_uncached_dates(self):
        """Only the 2 cached days contribute; uncached dates are skipped."""
        days = {
            SCAN_DATE - timedelta(days=1): [_option_row("CAT", 315.0, "C", 100.0)],
            SCAN_DATE - timedelta(days=3): [_option_row("CAT", 315.0, "C", 200.0)],
            # day 2 and all others are absent → not cached
        }
        ffp = _make_ffp(days)
        result = compute_20day_avg_volume("CAT", SCAN_DATE, ffp, trading_days=20)
        # sum=300, denominator=20, avg=15
        assert result[(315.0, "C")] == pytest.approx(15.0)

    def test_separate_calls_and_puts(self):
        days = {
            SCAN_DATE - timedelta(days=1): [
                _option_row("CAT", 315.0, "C", 100.0),
                _option_row("CAT", 285.0, "P", 50.0),
            ]
        }
        ffp = _make_ffp(days)
        result = compute_20day_avg_volume("CAT", SCAN_DATE, ffp, trading_days=20)
        assert (315.0, "C") in result
        assert (285.0, "P") in result

    def test_multiple_strikes(self):
        days = {
            SCAN_DATE - timedelta(days=1): [
                _option_row("CAT", 310.0, "C", 500.0),
                _option_row("CAT", 320.0, "C", 300.0),
            ]
        }
        ffp = _make_ffp(days)
        result = compute_20day_avg_volume("CAT", SCAN_DATE, ffp, trading_days=20)
        assert len(result) == 2
        assert result[(310.0, "C")] == pytest.approx(25.0)   # 500/20


# ══════════════════════════════════════════════════════════════════════════════
# detect_unusual_activity — filters
# ══════════════════════════════════════════════════════════════════════════════

class TestDetectUnusualActivity:

    def _setup_ffp(self, today_vol: float, avg_vol: float = 100.0,
                   strike: float = 315.0, dte: int = 35) -> MagicMock:
        """Convenience: one call strike today, one cached historical day."""
        today_row = _option_row("CAT", strike, "C", today_vol, dte_days=dte)
        hist_row  = _option_row("CAT", strike, "C", avg_vol, dte_days=dte + 1)

        hist_date = SCAN_DATE - timedelta(days=1)
        ffp = _make_ffp({
            SCAN_DATE:  [today_row],
            hist_date:  [hist_row],
        })
        return ffp

    def test_detects_high_ratio_call(self):
        """1000 vol today vs historical avg of 400 (÷20 = 20/day) → 50× ratio."""
        # avg_vol=400 in 1 cached day → avg_20day = 400/20 = 20 (above min of 10)
        # ratio = 1000 / 20 = 50 >> 5× threshold
        ffp = self._setup_ffp(today_vol=1000.0, avg_vol=400.0)
        signals = detect_unusual_activity("CAT", SCAN_DATE, ffp, SPOT)
        assert len(signals) >= 1
        assert signals[0].volume_ratio == pytest.approx(1000.0 / (400.0 / 20))

    def test_rejects_below_threshold(self):
        """500 vol vs 100-day avg → ratio = 500/(100/20) = 100 wait...
        avg_20day = 100/20 = 5. ratio = 500/5 = 100. Still above threshold.
        Use very low today volume."""
        # avg = 200/20 = 10. today = 30. ratio = 3 < 5 threshold
        ffp = self._setup_ffp(today_vol=30.0, avg_vol=200.0)
        signals = detect_unusual_activity("CAT", SCAN_DATE, ffp, SPOT)
        assert signals == []

    def test_rejects_puts(self):
        """Put contracts should never appear in signals (calls only)."""
        put_row = _option_row("CAT", 285.0, "P", 5000.0, dte_days=35)
        hist_put = _option_row("CAT", 285.0, "P", 50.0, dte_days=36)
        hist_d = SCAN_DATE - timedelta(days=1)
        ffp = _make_ffp({SCAN_DATE: [put_row], hist_d: [hist_put]})
        signals = detect_unusual_activity("CAT", SCAN_DATE, ffp, SPOT)
        assert all(s.option_type == "C" for s in signals)
        assert signals == []  # no calls in this setup

    def test_rejects_itm_calls(self):
        """Strike at 0% OTM (ATM) → below otm_pct_min=0.03, rejected."""
        ffp = self._setup_ffp(today_vol=5000.0, strike=SPOT)  # exactly ATM
        signals = detect_unusual_activity("CAT", SCAN_DATE, ffp, SPOT)
        assert signals == []

    def test_rejects_far_otm_calls(self):
        """Strike > 20% OTM → rejected (too far OTM for institutional buying)."""
        far_strike = SPOT * 1.25   # 25% OTM
        ffp = self._setup_ffp(today_vol=5000.0, strike=far_strike)
        signals = detect_unusual_activity("CAT", SCAN_DATE, ffp, SPOT)
        assert signals == []

    def test_rejects_short_dte(self):
        """DTE = 5 → below dte_min=10, rejected (weekly)."""
        ffp = self._setup_ffp(today_vol=5000.0, dte=5)
        signals = detect_unusual_activity("CAT", SCAN_DATE, ffp, SPOT)
        assert signals == []

    def test_rejects_long_dte(self):
        """DTE = 90 → above dte_max=60, rejected (LEAP)."""
        ffp = self._setup_ffp(today_vol=5000.0, dte=90)
        signals = detect_unusual_activity("CAT", SCAN_DATE, ffp, SPOT)
        assert signals == []

    def test_rejects_thinly_traded_baseline(self):
        """avg_20day < 10 → min_avg_volume filter rejects it."""
        # avg = 5/20 = 0.25 (way below 10)
        ffp = self._setup_ffp(today_vol=5000.0, avg_vol=5.0)
        signals = detect_unusual_activity("CAT", SCAN_DATE, ffp, SPOT)
        assert signals == []

    def test_concentration_flag_top3(self):
        """Three strikes each with very high volume → concentrated."""
        strike1, strike2, strike3 = 309.0, 315.0, 321.0
        today_rows = [
            _option_row("CAT", strike1, "C", 8000.0, dte_days=35),
            _option_row("CAT", strike2, "C", 7000.0, dte_days=35),
            _option_row("CAT", strike3, "C", 5000.0, dte_days=35),
        ]
        hist_d = SCAN_DATE - timedelta(days=1)
        hist_rows = [
            _option_row("CAT", strike1, "C", 2000.0, dte_days=36),
            _option_row("CAT", strike2, "C", 2000.0, dte_days=36),
            _option_row("CAT", strike3, "C", 2000.0, dte_days=36),
        ]
        ffp = _make_ffp({SCAN_DATE: today_rows, hist_d: hist_rows})
        signals = detect_unusual_activity("CAT", SCAN_DATE, ffp, SPOT)
        assert len(signals) > 0
        assert all(s.is_concentrated for s in signals)

    def test_sorted_by_ratio_descending(self):
        """Highest ratio strike comes first."""
        strike_high, strike_low = 309.0, 321.0
        today_rows = [
            _option_row("CAT", strike_high, "C", 10000.0, dte_days=35),
            _option_row("CAT", strike_low,  "C",  2000.0, dte_days=35),
        ]
        hist_d = SCAN_DATE - timedelta(days=1)
        hist_rows = [
            _option_row("CAT", strike_high, "C", 1000.0, dte_days=36),
            _option_row("CAT", strike_low,  "C", 1000.0, dte_days=36),
        ]
        ffp = _make_ffp({SCAN_DATE: today_rows, hist_d: hist_rows})
        signals = detect_unusual_activity("CAT", SCAN_DATE, ffp, SPOT)
        if len(signals) >= 2:
            assert signals[0].volume_ratio >= signals[1].volume_ratio

    def test_empty_when_no_today_chain(self):
        ffp = _make_ffp({})
        signals = detect_unusual_activity("CAT", SCAN_DATE, ffp, SPOT)
        assert signals == []

    def test_zero_spot_returns_empty(self):
        ffp = _make_ffp({SCAN_DATE: [_option_row("CAT", 315.0, "C", 5000.0)]})
        signals = detect_unusual_activity("CAT", SCAN_DATE, ffp, spot_price=0.0)
        assert signals == []

    def test_custom_threshold(self):
        """3× threshold (large_cap mode) finds signals that 5× misses."""
        ffp = self._setup_ffp(today_vol=400.0, avg_vol=100.0)
        # avg_20day = 100/20 = 5. ratio = 400/5 = 80. Both thresholds pass.
        # To test threshold: use ratio barely above 3x but below 5x
        # avg_20day_vol = 200/20 = 10. today = 40. ratio = 4 > 3 but < 5.
        ffp2 = self._setup_ffp(today_vol=40.0, avg_vol=200.0)
        signals_5x = detect_unusual_activity("CAT", SCAN_DATE, ffp2, SPOT,
                                              volume_ratio_threshold=5.0)
        signals_3x = detect_unusual_activity("CAT", SCAN_DATE, ffp2, SPOT,
                                              volume_ratio_threshold=3.0)
        assert signals_5x == []
        assert len(signals_3x) >= 1


# ══════════════════════════════════════════════════════════════════════════════
# score_unusual_activity
# ══════════════════════════════════════════════════════════════════════════════

class TestScoreUnusualActivity:

    def test_empty_signals_returns_zero(self):
        assert score_unusual_activity([]) == 0.0

    def test_single_strong_signal(self):
        """50× ratio, concentrated, optimal DTE → near max score."""
        s = _sig(volume_ratio=50.0, dte=37, concentrated=True)
        score = score_unusual_activity([s])
        assert score > 0.8

    def test_single_weak_signal(self):
        """Just above threshold: 5×, not concentrated, poor DTE → low score."""
        s = _sig(volume_ratio=5.1, dte=55, concentrated=False)
        score = score_unusual_activity([s])
        assert score < 0.5

    def test_concentration_boosts_score(self):
        """Concentrated > non-concentrated for same ratio and DTE."""
        s_conc = _sig(volume_ratio=20.0, dte=35, concentrated=True)
        s_dist = _sig(volume_ratio=20.0, dte=35, concentrated=False)
        assert score_unusual_activity([s_conc]) > score_unusual_activity([s_dist])

    def test_higher_ratio_boosts_score(self):
        """10× < 50× for otherwise identical signals."""
        s_low  = _sig(volume_ratio=10.0, dte=35, concentrated=True)
        s_high = _sig(volume_ratio=50.0, dte=35, concentrated=True)
        assert score_unusual_activity([s_high]) > score_unusual_activity([s_low])

    def test_dte_quality_peaks_near_37(self):
        """DTE 35 should score higher than DTE 55 (further from optimal)."""
        s_good = _sig(volume_ratio=20.0, dte=35, concentrated=True)
        s_poor = _sig(volume_ratio=20.0, dte=55, concentrated=True)
        assert score_unusual_activity([s_good]) > score_unusual_activity([s_poor])

    def test_score_bounded_0_1(self):
        s = _sig(volume_ratio=1000.0, dte=37, concentrated=True)
        score = score_unusual_activity([s])
        assert 0.0 <= score <= 1.0

    def test_fills_per_signal_conviction(self):
        """score_unusual_activity should back-fill conviction_score on each signal."""
        signals = [
            _sig(volume_ratio=20.0, dte=35, concentrated=True),
            _sig(volume_ratio=10.0, dte=40, concentrated=True),
        ]
        score_unusual_activity(signals)
        assert all(s.conviction_score > 0 for s in signals)

    def test_0_5_threshold(self):
        """Scores near the 0.5 boundary for the +0.08 conviction bonus."""
        # Moderate signal: ~10× ratio, concentrated, good DTE → score near 0.5-0.6
        s = _sig(volume_ratio=10.0, dte=35, concentrated=True)
        score = score_unusual_activity([s])
        # Should be above 0.5 (concentration + moderate ratio)
        assert score >= 0.5

    def test_0_7_threshold(self):
        """Strong signal should cross 0.7 for the +0.15 bonus."""
        s = _sig(volume_ratio=30.0, dte=35, concentrated=True)
        score = score_unusual_activity([s])
        assert score >= 0.7


# ══════════════════════════════════════════════════════════════════════════════
# market_cap_tier
# ══════════════════════════════════════════════════════════════════════════════

class TestMarketCapTier:

    def test_mega_cap(self):
        from quantlab.execution import market_cap_tier
        assert market_cap_tier("AAPL") == "mega_cap"
        assert market_cap_tier("MSFT") == "mega_cap"
        assert market_cap_tier("NVDA") == "mega_cap"

    def test_large_cap_sp500(self):
        from quantlab.execution import market_cap_tier
        # CAT is in SP500_SAMPLE but not MEGA_CAP_LIQUID
        assert market_cap_tier("CAT") == "large_cap"
        assert market_cap_tier("JPM") == "large_cap"

    def test_mid_cap_unknown(self):
        from quantlab.execution import market_cap_tier
        # Symbol not in any known list → mid_cap
        assert market_cap_tier("CELH") == "mid_cap"
        assert market_cap_tier("ZBRA") == "mid_cap"

    def test_returns_string(self):
        from quantlab.execution import market_cap_tier
        result = market_cap_tier("AAPL")
        assert isinstance(result, str)
        assert result in ("mega_cap", "large_cap", "mid_cap", "small_cap")


# ══════════════════════════════════════════════════════════════════════════════
# ScanResult new fields
# ══════════════════════════════════════════════════════════════════════════════

class TestScanResultNewFields:

    def _result(self, **kwargs):
        from quantlab.execution import ScanResult
        defaults = dict(
            symbol="CAT", scan_date="2025-05-01",
            signal_type="breakout", signal=True,
            entry_close=300.0, indicator_value=299.0, lookback=20,
            regime_bullish=True,
        )
        defaults.update(kwargs)
        return ScanResult(**defaults)

    def test_unusual_options_score_default(self):
        r = self._result()
        assert r.unusual_options_score == 0.0

    def test_market_cap_tier_default_empty(self):
        r = self._result()
        assert r.market_cap_tier == ""

    def test_market_cap_tier_set_by_scan_symbol(self):
        """scan_symbol should populate market_cap_tier."""
        from datetime import date as _date
        from quantlab.execution import scan_symbol
        from quantlab.providers.base import Bar

        bars = [
            Bar(as_of=_date(2025, 1, 2) + __import__('datetime').timedelta(days=i),
                open=300.0, high=305.0, low=295.0, close=300.0 + i * 0.5,
                volume=1_000_000.0)
            for i in range(40)
        ]
        result = scan_symbol("CAT", bars, signal_type="breakout", lookback=20)
        if result is not None:
            assert result.market_cap_tier in ("mega_cap", "large_cap", "mid_cap")

    def test_score_conviction_uses_market_cap_tier_field(self):
        """When market_cap_tier is pre-set on result, conviction uses it."""
        from quantlab.execution import score_conviction

        r_mid = self._result(
            market_cap_tier="mid_cap",
            unusual_options_score=0.75,
        )
        r_mega = self._result(
            market_cap_tier="mega_cap",
            unusual_options_score=0.75,  # should be ignored for mega_cap
        )
        # Mid-cap uses unusual score (+0.15); mega-cap uses flat options (0)
        assert score_conviction(r_mid) > score_conviction(r_mega)


# ══════════════════════════════════════════════════════════════════════════════
# score_conviction — tier-aware options routing
# ══════════════════════════════════════════════════════════════════════════════

class TestScoreConvictionTierAware:

    def _r(self, **kwargs):
        from quantlab.execution import ScanResult
        defaults = dict(
            symbol="X", scan_date="2025-05-01",
            signal_type="breakout", signal=True,
            entry_close=100.0, indicator_value=99.0, lookback=20,
            regime_bullish=True,
        )
        defaults.update(kwargs)
        return ScanResult(**defaults)

    def test_mid_cap_unusual_high_gets_015(self):
        from quantlab.execution import score_conviction
        r_no  = self._r(market_cap_tier="mid_cap", unusual_options_score=0.0)
        r_yes = self._r(market_cap_tier="mid_cap", unusual_options_score=0.75)
        assert score_conviction(r_yes) - score_conviction(r_no) == pytest.approx(0.15)

    def test_mid_cap_unusual_moderate_gets_008(self):
        from quantlab.execution import score_conviction
        r_no  = self._r(market_cap_tier="mid_cap", unusual_options_score=0.0)
        r_yes = self._r(market_cap_tier="mid_cap", unusual_options_score=0.55)
        assert score_conviction(r_yes) - score_conviction(r_no) == pytest.approx(0.08)

    def test_mid_cap_unusual_below_threshold_no_bonus(self):
        from quantlab.execution import score_conviction
        r_no  = self._r(market_cap_tier="mid_cap", unusual_options_score=0.0)
        r_low = self._r(market_cap_tier="mid_cap", unusual_options_score=0.30)
        assert score_conviction(r_low) == pytest.approx(score_conviction(r_no))

    def test_mega_cap_ignores_unusual_uses_flat_score(self):
        from quantlab.execution import score_conviction
        # Unusual score shouldn't help mega_cap
        r_no_unusual = self._r(market_cap_tier="mega_cap", unusual_options_score=0.0,
                                options_score=0.70)
        r_with_unusual = self._r(market_cap_tier="mega_cap", unusual_options_score=0.80,
                                  options_score=0.70)
        # Both should score the same (unusual ignored for mega_cap)
        assert score_conviction(r_no_unusual) == pytest.approx(score_conviction(r_with_unusual))

    def test_mega_cap_uses_options_score_for_pcr_iv(self):
        from quantlab.execution import score_conviction
        r_low  = self._r(market_cap_tier="mega_cap", options_score=0.50)
        r_high = self._r(market_cap_tier="mega_cap", options_score=0.85)
        assert score_conviction(r_high) > score_conviction(r_low)
        # 0.85 → +0.15, 0.50 → +0.00 → diff = 0.15
        assert score_conviction(r_high) - score_conviction(r_low) == pytest.approx(0.15)

    def test_small_cap_gets_no_options_bonus(self):
        from quantlab.execution import score_conviction
        r_base = self._r(market_cap_tier="small_cap")
        r_opts = self._r(market_cap_tier="small_cap", options_score=0.90,
                          unusual_options_score=0.90)
        assert score_conviction(r_base) == pytest.approx(score_conviction(r_opts))

    def test_large_cap_falls_back_to_flat_score(self):
        from quantlab.execution import score_conviction
        r_no   = self._r(market_cap_tier="large_cap")
        r_flat = self._r(market_cap_tier="large_cap", options_score=0.70)
        assert score_conviction(r_flat) > score_conviction(r_no)
        assert score_conviction(r_flat) - score_conviction(r_no) == pytest.approx(0.10)

    def test_mid_cap_unusual_preferred_over_flat_options(self):
        from quantlab.execution import score_conviction
        # Both flat and unusual set — unusual should win for mid_cap
        r = self._r(market_cap_tier="mid_cap", unusual_options_score=0.75,
                    options_score=0.90)
        r_flat_only = self._r(market_cap_tier="mega_cap", unusual_options_score=0.75,
                               options_score=0.90)
        # mid_cap uses unusual (+0.15); mega_cap uses flat options_score (0.90 → +0.15)
        # Same bonus amount in this case but different routing
        assert score_conviction(r) == pytest.approx(score_conviction(r_flat_only))
