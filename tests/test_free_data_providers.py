"""
tests/test_free_data_providers.py — Tests for CBOE, FRED, and EDGAR providers.

All tests use mocked HTTP responses — no network access required.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ══════════════════════════════════════════════════════════════════════════════
# CBOE VIX
# ══════════════════════════════════════════════════════════════════════════════

class TestCboe:

    def test_vix_bar_dataclass(self):
        from quantlab.providers.cboe import VixBar
        bar = VixBar(date=date(2026, 1, 2), open=15.0, high=16.5, low=14.5, close=16.0)
        assert bar.date == date(2026, 1, 2)
        assert bar.close == 16.0
        assert bar.high >= bar.low

    def test_classify_vix_regime_low(self):
        from quantlab.providers.cboe import classify_vix_regime
        label, score = classify_vix_regime(12.5)
        assert label == "low"
        assert score == 0

    def test_classify_vix_regime_elevated(self):
        from quantlab.providers.cboe import classify_vix_regime
        label, score = classify_vix_regime(20.0)
        assert label == "elevated"
        assert score == 1

    def test_classify_vix_regime_high(self):
        from quantlab.providers.cboe import classify_vix_regime
        label, score = classify_vix_regime(30.0)
        assert label == "high"
        assert score == 2

    def test_classify_vix_regime_extreme(self):
        from quantlab.providers.cboe import classify_vix_regime
        label, score = classify_vix_regime(40.0)
        assert label == "extreme"
        assert score == 3

    def test_classify_vix_regime_boundaries(self):
        from quantlab.providers.cboe import classify_vix_regime
        assert classify_vix_regime(14.99)[0] == "low"
        assert classify_vix_regime(15.0)[0] == "elevated"
        assert classify_vix_regime(24.99)[0] == "elevated"
        assert classify_vix_regime(25.0)[0] == "high"
        assert classify_vix_regime(34.99)[0] == "high"
        assert classify_vix_regime(35.0)[0] == "extreme"

    def test_fetch_vix_history_mocked(self):
        csv_content = (
            "DATE,OPEN,HIGH,LOW,CLOSE\n"
            "01/02/2026,14.00,15.50,13.80,15.00\n"
            "01/05/2026,15.10,16.00,14.90,15.80\n"
            "01/06/2026,15.80,17.00,15.50,16.50\n"
        )
        mock_resp = MagicMock()
        mock_resp.text = csv_content
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_resp):
            from quantlab.providers.cboe import fetch_vix_history
            bars = fetch_vix_history(date(2026, 1, 1), date(2026, 1, 31))

        assert len(bars) == 3
        assert bars[0].date == date(2026, 1, 2)
        assert bars[-1].close == 16.50

    def test_fetch_vix_history_filters_by_date(self):
        csv_content = (
            "DATE,OPEN,HIGH,LOW,CLOSE\n"
            "12/31/2025,13.00,13.50,12.80,13.20\n"
            "01/02/2026,14.00,15.50,13.80,15.00\n"
            "01/05/2026,15.10,16.00,14.90,15.80\n"
        )
        mock_resp = MagicMock()
        mock_resp.text = csv_content
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_resp):
            from quantlab.providers.cboe import fetch_vix_history
            bars = fetch_vix_history(date(2026, 1, 1), date(2026, 1, 31))

        assert len(bars) == 2
        assert bars[0].date == date(2026, 1, 2)

    def test_fetch_vix_history_sorted_ascending(self):
        # CSV rows out of order
        csv_content = (
            "DATE,OPEN,HIGH,LOW,CLOSE\n"
            "01/05/2026,15.10,16.00,14.90,15.80\n"
            "01/02/2026,14.00,15.50,13.80,15.00\n"
        )
        mock_resp = MagicMock()
        mock_resp.text = csv_content
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_resp):
            from quantlab.providers.cboe import fetch_vix_history
            bars = fetch_vix_history(date(2026, 1, 1), date(2026, 1, 31))

        assert bars[0].date < bars[1].date


# ══════════════════════════════════════════════════════════════════════════════
# FRED
# ══════════════════════════════════════════════════════════════════════════════

class TestFred:

    def test_fred_series_dict_has_all_expected_keys(self):
        from quantlab.providers.fred import FRED_SERIES
        expected = {"T10Y2Y", "T10Y3M", "BAMLH0A0HYM2", "DGS10", "FEDFUNDS", "DCOILWTICO"}
        assert expected == set(FRED_SERIES.keys())

    def test_macro_snapshot_defaults(self):
        from quantlab.providers.fred import MacroSnapshot
        snap = MacroSnapshot(as_of=date(2026, 1, 1))
        assert snap.macro_regime == "risk_on"
        assert snap.yield_spread_10y2y is None
        assert snap.vix_close is None

    def test_classify_macro_regime_risk_on_no_warnings(self):
        from quantlab.providers.fred import MacroSnapshot, classify_macro_regime
        snap = MacroSnapshot(
            as_of=date(2026, 1, 1),
            yield_spread_10y2y=0.50,   # positive — no warning
            hy_credit_spread=3.00,      # below 5.0 — no warning
            vix_close=12.0,             # below 25 — no warning
        )
        assert classify_macro_regime(snap) == "risk_on"

    def test_classify_macro_regime_risk_off_one_warning(self):
        from quantlab.providers.fred import MacroSnapshot, classify_macro_regime
        snap = MacroSnapshot(
            as_of=date(2026, 1, 1),
            yield_spread_10y2y=-0.10,  # inverted — one warning
            hy_credit_spread=3.00,
            vix_close=12.0,
        )
        assert classify_macro_regime(snap) == "risk_off"

    def test_classify_macro_regime_stress_two_warnings(self):
        from quantlab.providers.fred import MacroSnapshot, classify_macro_regime
        snap = MacroSnapshot(
            as_of=date(2026, 1, 1),
            yield_spread_10y2y=-0.20,  # inverted
            hy_credit_spread=6.50,      # above 5.0
            vix_close=20.0,
        )
        assert classify_macro_regime(snap) == "stress"

    def test_classify_macro_regime_stress_with_vix(self):
        from quantlab.providers.fred import MacroSnapshot, classify_macro_regime
        snap = MacroSnapshot(
            as_of=date(2026, 1, 1),
            yield_spread_10y2y=0.10,   # normal
            hy_credit_spread=6.00,      # above 5.0 — warning
            vix_close=30.0,             # above 25 — warning
        )
        assert classify_macro_regime(snap) == "stress"

    def test_classify_macro_regime_none_values_treated_as_no_warning(self):
        from quantlab.providers.fred import MacroSnapshot, classify_macro_regime
        snap = MacroSnapshot(as_of=date(2026, 1, 1))  # all None
        assert classify_macro_regime(snap) == "risk_on"

    def test_fetch_series_mocked(self):
        mock_json = {
            "observations": [
                {"date": "2026-01-02", "value": "0.45"},
                {"date": "2026-01-03", "value": "-0.12"},
                {"date": "2026-01-04", "value": "."},  # missing — should be skipped
            ]
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_json
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_resp):
            from quantlab.providers.fred import fetch_series
            result = fetch_series("T10Y2Y", date(2026, 1, 1), date(2026, 1, 31), "test_key")

        assert date(2026, 1, 2) in result
        assert result[date(2026, 1, 2)] == pytest.approx(0.45)
        assert date(2026, 1, 3) in result
        assert date(2026, 1, 4) not in result  # '.' skipped

    def test_fetch_macro_snapshot_mocked(self):
        def _make_obs(val: str) -> dict:
            return {"observations": [{"date": "2026-01-10", "value": val}]}

        call_count = 0
        responses = [
            _make_obs("0.40"),   # T10Y2Y
            _make_obs("-0.30"),  # T10Y3M
            _make_obs("3.50"),   # BAMLH0A0HYM2
            _make_obs("4.35"),   # DGS10
            _make_obs("5.25"),   # FEDFUNDS
            _make_obs("78.40"),  # DCOILWTICO
        ]

        def side_effect(*args, **kwargs):
            nonlocal call_count
            mock_resp = MagicMock()
            mock_resp.json.return_value = responses[call_count % len(responses)]
            mock_resp.raise_for_status = MagicMock()
            call_count += 1
            return mock_resp

        with patch("requests.get", side_effect=side_effect):
            with patch("quantlab.providers.fred._store_snapshot"):
                from quantlab.providers.fred import fetch_macro_snapshot
                snap = fetch_macro_snapshot("test_key", date(2026, 1, 10))

        assert snap.yield_spread_10y2y == pytest.approx(0.40)
        assert snap.treasury_10y == pytest.approx(4.35)
        assert snap.wti_crude == pytest.approx(78.40)
        assert snap.macro_regime == "risk_on"


# ══════════════════════════════════════════════════════════════════════════════
# SEC EDGAR
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgar:

    def test_fundamental_snapshot_defaults(self):
        from quantlab.providers.edgar import FundamentalSnapshot
        snap = FundamentalSnapshot(ticker="AAPL", cik="0000320193", as_of=date(2026, 1, 1))
        assert snap.revenue is None
        assert snap.eps_history == []
        assert snap.net_income_history == []

    def test_compute_earnings_acceleration_none_when_no_data(self):
        """No earnings history at all → None (unavailable), not a neutral score."""
        from quantlab.providers.edgar import FundamentalSnapshot, compute_earnings_acceleration
        snap = FundamentalSnapshot(ticker="X", cik="0", as_of=date(2026, 1, 1))
        assert compute_earnings_acceleration(snap) is None

    def test_compute_earnings_acceleration_none_two_points(self):
        """Two data points are below the 3-quarter minimum → None."""
        from quantlab.providers.edgar import FundamentalSnapshot, compute_earnings_acceleration
        snap = FundamentalSnapshot(ticker="X", cik="0", as_of=date(2026, 1, 1))
        snap.eps_history = [1.0, 1.1]
        assert compute_earnings_acceleration(snap) is None

    def test_compute_earnings_acceleration_accelerating(self):
        from quantlab.providers.edgar import FundamentalSnapshot, compute_earnings_acceleration
        # Growth: 10% then 20% — accelerating
        snap = FundamentalSnapshot(ticker="X", cik="0", as_of=date(2026, 1, 1))
        snap.eps_history = [1.0, 1.10, 1.32]  # +10%, then +20%
        score = compute_earnings_acceleration(snap)
        assert score > 0.5, f"Expected accelerating score > 0.5, got {score}"

    def test_compute_earnings_acceleration_decelerating(self):
        from quantlab.providers.edgar import FundamentalSnapshot, compute_earnings_acceleration
        # Growth: 20% then 5% — decelerating
        snap = FundamentalSnapshot(ticker="X", cik="0", as_of=date(2026, 1, 1))
        snap.eps_history = [1.0, 1.20, 1.26]  # +20%, then +5%
        score = compute_earnings_acceleration(snap)
        assert score < 0.5, f"Expected decelerating score < 0.5, got {score}"

    def test_compute_earnings_acceleration_uses_ni_fallback(self):
        from quantlab.providers.edgar import FundamentalSnapshot, compute_earnings_acceleration
        snap = FundamentalSnapshot(ticker="X", cik="0", as_of=date(2026, 1, 1))
        snap.net_income_history = [100.0, 110.0, 132.0]  # accelerating NI
        score = compute_earnings_acceleration(snap)
        assert score > 0.5

    def test_compute_earnings_acceleration_bounded(self):
        from quantlab.providers.edgar import FundamentalSnapshot, compute_earnings_acceleration
        snap = FundamentalSnapshot(ticker="X", cik="0", as_of=date(2026, 1, 1))
        snap.eps_history = [0.01, 10.0, 1.0]  # extreme values
        score = compute_earnings_acceleration(snap)
        assert 0.0 <= score <= 1.0

    def test_lookup_cik_mocked(self):
        mock_data = {
            "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
            "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_data
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_resp):
            from quantlab.providers.edgar import lookup_cik
            cik = lookup_cik("AAPL")

        assert cik == "0000320193"

    def test_lookup_cik_case_insensitive(self):
        mock_data = {
            "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_data
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_resp):
            from quantlab.providers.edgar import lookup_cik
            cik = lookup_cik("aapl")

        assert cik == "0000320193"

    def test_lookup_cik_not_found(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_resp):
            from quantlab.providers.edgar import lookup_cik
            with pytest.raises(ValueError, match="not found"):
                lookup_cik("ZZZZ")

    def test_fetch_fundamentals_mocked(self):
        ticker_data = {
            "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        }
        # Duration-explicit extraction requires period start dates (real
        # companyfacts always carry them for income-statement facts)
        eps_obs = [
            {"start": "2025-01-01", "end": "2025-03-31", "val": 1.52, "form": "10-Q", "filed": "2025-05-01"},
            {"start": "2025-04-01", "end": "2025-06-30", "val": 1.65, "form": "10-Q", "filed": "2025-08-01"},
            {"start": "2025-07-01", "end": "2025-09-30", "val": 1.82, "form": "10-Q", "filed": "2025-11-01"},
        ]
        ni_obs = [
            {"start": "2025-01-01", "end": "2025-03-31", "val": 24780000000, "form": "10-Q", "filed": "2025-05-01"},
            {"start": "2025-04-01", "end": "2025-06-30", "val": 26870000000, "form": "10-Q", "filed": "2025-08-01"},
            {"start": "2025-07-01", "end": "2025-09-30", "val": 29600000000, "form": "10-Q", "filed": "2025-11-01"},
        ]
        facts_data = {
            "facts": {
                "us-gaap": {
                    "EarningsPerShareDiluted": {"units": {"USD": eps_obs}},
                    "NetIncomeLoss": {"units": {"USD": ni_obs}},
                }
            }
        }
        mock_facts_resp = MagicMock()
        mock_facts_resp.json.return_value = facts_data
        mock_facts_resp.raise_for_status = MagicMock()

        # Patch _get_company_tickers directly to bypass the LRU cache so the
        # single requests.get call goes to the companyfacts URL only.
        with patch("quantlab.providers.edgar._get_company_tickers", return_value=ticker_data):
            with patch("requests.get", return_value=mock_facts_resp):
                from quantlab.providers.edgar import fetch_fundamentals
                snap = fetch_fundamentals("AAPL", metrics=["eps_diluted", "net_income"])

        assert snap.ticker == "AAPL"
        assert snap.cik == "0000320193"
        assert snap.eps_diluted == pytest.approx(1.82)
        assert len(snap.eps_history) == 3
        assert snap.net_income == pytest.approx(29600000000)
        # EPS QoQ growth: (1.82 - 1.65) / 1.65 ≈ 10.3%
        assert snap.eps_qoq_growth is not None
        assert snap.eps_qoq_growth > 0


# ══════════════════════════════════════════════════════════════════════════════
# Execution integration — macro_regime / vix_regime on ScanResult
# ══════════════════════════════════════════════════════════════════════════════

class TestExecutionMacroIntegration:

    def _base_result(self, signal: bool = True, **kwargs) -> "ScanResult":
        from quantlab.execution import ScanResult
        defaults = dict(
            symbol="AAPL",
            scan_date="2026-06-06",
            signal_type="breakout",
            signal=signal,
            entry_close=200.0,
            indicator_value=199.0,
            lookback=20,
            regime_bullish=True,
        )
        defaults.update(kwargs)
        return ScanResult(**defaults)

    def test_scan_result_has_macro_regime_field(self):
        r = self._base_result()
        assert hasattr(r, "macro_regime")
        assert r.macro_regime == "risk_on"

    def test_scan_result_has_vix_regime_field(self):
        r = self._base_result()
        assert hasattr(r, "vix_regime")
        assert r.vix_regime == "low"

    def test_score_conviction_risk_on_no_penalty(self):
        from quantlab.execution import score_conviction
        r = self._base_result(macro_regime="risk_on")
        score_risk_on = score_conviction(r)

        r2 = self._base_result()  # default is risk_on
        assert score_conviction(r2) == pytest.approx(score_risk_on)

    def test_score_conviction_risk_off_reduces_score(self):
        from quantlab.execution import score_conviction
        r_on = self._base_result(macro_regime="risk_on")
        r_off = self._base_result(macro_regime="risk_off")
        assert score_conviction(r_off) < score_conviction(r_on)
        assert score_conviction(r_on) - score_conviction(r_off) == pytest.approx(0.05)

    def test_score_conviction_stress_reduces_score_more(self):
        from quantlab.execution import score_conviction
        r_on = self._base_result(macro_regime="risk_on")
        r_stress = self._base_result(macro_regime="stress")
        assert score_conviction(r_stress) < score_conviction(r_on)
        assert score_conviction(r_on) - score_conviction(r_stress) == pytest.approx(0.10)

    def test_score_conviction_stress_greater_penalty_than_risk_off(self):
        from quantlab.execution import score_conviction
        r_off = self._base_result(macro_regime="risk_off")
        r_stress = self._base_result(macro_regime="stress")
        assert score_conviction(r_stress) < score_conviction(r_off)

    def test_score_conviction_clamped_to_zero(self):
        from quantlab.execution import score_conviction
        # Even with stress, score cannot go below 0
        r = self._base_result(macro_regime="stress", regime_bullish=False)
        assert score_conviction(r) >= 0.0

    def test_score_conviction_stress_no_effect_without_signal(self):
        from quantlab.execution import score_conviction
        r = self._base_result(signal=False, macro_regime="stress")
        assert score_conviction(r) == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# EDGAR cache layer
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgarCache:

    def test_count_consecutive_beats_all_up(self):
        from quantlab.providers.edgar import _count_consecutive_beats
        history = [1.0, 1.1, 1.2, 1.3]
        assert _count_consecutive_beats(history) == 3

    def test_count_consecutive_beats_none(self):
        from quantlab.providers.edgar import _count_consecutive_beats
        history = [1.3, 1.2, 1.1]  # declining
        assert _count_consecutive_beats(history) == 0

    def test_count_consecutive_beats_partial(self):
        from quantlab.providers.edgar import _count_consecutive_beats
        # Last two up, one down before that
        history = [1.0, 0.9, 1.0, 1.1]
        assert _count_consecutive_beats(history) == 2

    def test_count_consecutive_beats_empty(self):
        from quantlab.providers.edgar import _count_consecutive_beats
        assert _count_consecutive_beats([]) == 0

    def test_get_edgar_acceleration_uses_cache(self):
        """Cache hit should return without making any HTTP calls."""
        from quantlab.providers.edgar import get_edgar_acceleration

        with patch("quantlab.providers.edgar._load_edgar_cache", return_value=(True, 0.65)):
            with patch("quantlab.providers.edgar.fetch_fundamentals") as mock_fetch:
                result = get_edgar_acceleration("AAPL")

        assert result == pytest.approx(0.65)
        mock_fetch.assert_not_called()  # no network call needed

    def test_get_edgar_acceleration_fetches_on_cache_miss(self):
        """Cache miss should fetch from EDGAR and save result."""
        from quantlab.providers.edgar import (
            FundamentalSnapshot, get_edgar_acceleration
        )
        snap = FundamentalSnapshot(ticker="AAPL", cik="0000320193", as_of=date(2026, 1, 1))
        snap.eps_history = [1.0, 1.10, 1.32]  # accelerating

        with patch("quantlab.providers.edgar._load_edgar_cache", return_value=(False, None)):
            with patch("quantlab.providers.edgar.fetch_fundamentals", return_value=snap):
                with patch("quantlab.providers.edgar._save_edgar_cache") as mock_save:
                    result = get_edgar_acceleration("AAPL")

        assert result is not None
        assert result > 0.5  # accelerating
        mock_save.assert_called_once()

    def test_get_edgar_acceleration_returns_none_on_failure(self):
        """Network/lookup failure should return None, not raise."""
        from quantlab.providers.edgar import get_edgar_acceleration

        with patch("quantlab.providers.edgar._load_edgar_cache", return_value=(False, None)):
            with patch(
                "quantlab.providers.edgar.fetch_fundamentals",
                side_effect=ValueError("Ticker not found in SEC filing index: ZZZZ"),
            ):
                result = get_edgar_acceleration("ZZZZ")

        assert result is None

    def test_cik_lru_cache_hits(self):
        """Second lookup_cik call should not make a new HTTP request."""
        from quantlab.providers.edgar import _get_company_tickers, lookup_cik

        mock_data = {
            "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        }
        # Clear the lru_cache before this test
        _get_company_tickers.cache_clear()

        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = mock_data
            mock_resp.raise_for_status = MagicMock()
            mock_get.return_value = mock_resp

            lookup_cik("AAPL")
            lookup_cik("AAPL")  # second call — should use lru_cache

        # company_tickers.json fetched exactly once despite two lookups
        assert mock_get.call_count == 1
        _get_company_tickers.cache_clear()  # restore state


# ══════════════════════════════════════════════════════════════════════════════
# EDGAR / ScanResult integration
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgarScanIntegration:

    def _base_result(self, signal: bool = True, **kwargs):
        from quantlab.execution import ScanResult
        defaults = dict(
            symbol="AAPL",
            scan_date="2026-06-06",
            signal_type="breakout",
            signal=signal,
            entry_close=200.0,
            indicator_value=199.0,
            lookback=20,
            regime_bullish=True,
        )
        defaults.update(kwargs)
        return ScanResult(**defaults)

    def test_scan_result_has_edgar_acceleration_field(self):
        r = self._base_result()
        assert hasattr(r, "edgar_acceleration")
        assert r.edgar_acceleration is None  # unset by default

    def test_score_conviction_prefers_edgar_over_ohlcv(self):
        """When edgar_acceleration is set, it should override ohlcv inference."""
        from quantlab.execution import score_conviction

        # OHLCV says below threshold (0.3), EDGAR says above (0.7)
        r_edgar = self._base_result(
            earnings_acceleration=0.3,
            edgar_acceleration=0.7,
        )
        # Only OHLCV, below threshold
        r_ohlcv = self._base_result(
            earnings_acceleration=0.3,
            edgar_acceleration=None,
        )

        assert score_conviction(r_edgar) > score_conviction(r_ohlcv)
        assert score_conviction(r_edgar) - score_conviction(r_ohlcv) == pytest.approx(0.10)

    def test_score_conviction_falls_back_to_ohlcv_when_no_edgar(self):
        """With edgar_acceleration=None, should use earnings_acceleration."""
        from quantlab.execution import score_conviction

        r_high = self._base_result(earnings_acceleration=0.8, edgar_acceleration=None)
        r_low = self._base_result(earnings_acceleration=0.1, edgar_acceleration=None)

        assert score_conviction(r_high) > score_conviction(r_low)
        assert score_conviction(r_high) - score_conviction(r_low) == pytest.approx(0.10)

    def test_score_conviction_edgar_below_threshold_no_bonus(self):
        """EDGAR score below 0.5 means no acceleration bonus."""
        from quantlab.execution import score_conviction

        r = self._base_result(
            earnings_acceleration=0.9,  # OHLCV says high
            edgar_acceleration=0.3,      # EDGAR says low — EDGAR wins
        )
        r_no_accel = self._base_result(
            earnings_acceleration=0.1,
            edgar_acceleration=0.3,
        )

        assert score_conviction(r) == score_conviction(r_no_accel)

    def test_score_conviction_edgar_exactly_at_threshold(self):
        """EDGAR score of exactly 0.5 should trigger the bonus."""
        from quantlab.execution import score_conviction

        r_on = self._base_result(edgar_acceleration=0.5)
        r_off = self._base_result(edgar_acceleration=0.49)

        assert score_conviction(r_on) > score_conviction(r_off)
        assert score_conviction(r_on) - score_conviction(r_off) == pytest.approx(0.10)
