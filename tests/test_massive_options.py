"""
tests/test_massive_options.py — Tests for MassiveOptionsProvider.

All tests use mocked HTTP / DuckDB responses — no network access required.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quantlab.providers.massive_options import MassiveOptionsProvider, OptionContract


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_chain(
    symbol: str = "AAPL",
    spot: float = 200.0,
    n_strikes: int = 5,
    base_iv_call: float = 0.25,
    base_iv_put: float = 0.30,
    oi_per_contract: float = 1000.0,
    vol_per_contract: float = 200.0,
) -> list[OptionContract]:
    """Build a synthetic option chain centred around spot."""
    contracts: list[OptionContract] = []
    for i in range(-n_strikes, n_strikes + 1):
        strike = round(spot * (1 + i * 0.03), 2)
        for opt_type, iv_base in (("C", base_iv_call), ("P", base_iv_put)):
            # OTM contracts have higher IV for puts (normal smirk)
            iv_adj = iv_base + abs(i) * 0.01
            ticker = f"O:{symbol}260620{'C' if opt_type == 'C' else 'P'}{int(strike*1000):08d}"
            contracts.append(OptionContract(
                ticker=ticker,
                expiry=date(2026, 6, 20),
                strike=strike,
                option_type=opt_type,
                bid=max(0.01, (strike - spot) * 0.5 if opt_type == "C" else 1.0),
                ask=max(0.02, (strike - spot) * 0.5 + 0.1 if opt_type == "C" else 1.1),
                volume=vol_per_contract,
                open_interest=oi_per_contract,
                iv=iv_adj,
                delta=0.5 - i * 0.05 if opt_type == "C" else -0.5 + i * 0.05,
                gamma=0.02,
                theta=-0.05,
                vega=0.10,
            ))
    return contracts


def _provider() -> MassiveOptionsProvider:
    return MassiveOptionsProvider(api_key="test_key")


# ══════════════════════════════════════════════════════════════════════════════
# parse_option_ticker
# ══════════════════════════════════════════════════════════════════════════════

class TestParseOptionTicker:

    def test_call_basic(self):
        sym, expiry, opt_type, strike = MassiveOptionsProvider.parse_option_ticker(
            "O:AAPL260620C00310000"
        )
        assert sym == "AAPL"
        assert expiry == date(2026, 6, 20)
        assert opt_type == "C"
        assert strike == pytest.approx(310.0)

    def test_put_basic(self):
        sym, expiry, opt_type, strike = MassiveOptionsProvider.parse_option_ticker(
            "O:AAPL260620P00290000"
        )
        assert sym == "AAPL"
        assert opt_type == "P"
        assert strike == pytest.approx(290.0)

    def test_fractional_strike(self):
        # Strike 155.5 → 00155500
        sym, expiry, opt_type, strike = MassiveOptionsProvider.parse_option_ticker(
            "O:TSLA260620C00155500"
        )
        assert sym == "TSLA"
        assert strike == pytest.approx(155.5)

    def test_multi_char_symbol(self):
        sym, expiry, opt_type, strike = MassiveOptionsProvider.parse_option_ticker(
            "O:GOOGL260620C02000000"
        )
        assert sym == "GOOGL"
        assert strike == pytest.approx(2000.0)

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="too short"):
            MassiveOptionsProvider.parse_option_ticker("O:X")

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="option type"):
            MassiveOptionsProvider.parse_option_ticker("O:AAPL260620X00310000")

    def test_roundtrip_strike_precision(self):
        # Strike 0.50 → 00000500
        _, _, _, strike = MassiveOptionsProvider.parse_option_ticker(
            "O:SPY260620C00000500"
        )
        assert strike == pytest.approx(0.5)


# ══════════════════════════════════════════════════════════════════════════════
# OptionContract dataclass
# ══════════════════════════════════════════════════════════════════════════════

class TestOptionContract:

    def test_activity_prefers_oi(self):
        c = OptionContract(
            ticker="O:AAPL260620C00310000",
            expiry=date(2026, 6, 20),
            strike=310.0,
            option_type="C",
            volume=100.0,
            open_interest=5000.0,
        )
        assert c.activity == 5000.0

    def test_activity_falls_back_to_volume(self):
        c = OptionContract(
            ticker="O:AAPL260620C00310000",
            expiry=date(2026, 6, 20),
            strike=310.0,
            option_type="C",
            volume=300.0,
            open_interest=None,
        )
        assert c.activity == 300.0

    def test_activity_zero_when_both_none(self):
        c = OptionContract(
            ticker="O:AAPL260620C00310000",
            expiry=date(2026, 6, 20),
            strike=310.0,
            option_type="C",
        )
        assert c.activity == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# get_put_call_ratio
# ══════════════════════════════════════════════════════════════════════════════

class TestPutCallRatio:

    def test_neutral_equal_oi(self):
        p = _provider()
        chain = _make_chain(spot=200.0, oi_per_contract=1000.0)
        p._chain_cache["AAPL"] = chain
        pcr = p.get_put_call_ratio("AAPL", spot_price=200.0)
        # Equal OI → PCR near 1.0 for ATM strikes
        assert 0.5 < pcr < 2.0

    def test_bullish_when_more_calls(self):
        p = _provider()
        chain = []
        for strike in [195.0, 200.0, 205.0]:
            chain.append(OptionContract(
                ticker=f"O:AAPL260620C{int(strike*1000):08d}",
                expiry=date(2026, 6, 20), strike=strike, option_type="C",
                open_interest=5000.0, volume=500.0,
            ))
            chain.append(OptionContract(
                ticker=f"O:AAPL260620P{int(strike*1000):08d}",
                expiry=date(2026, 6, 20), strike=strike, option_type="P",
                open_interest=1000.0, volume=100.0,
            ))
        p._chain_cache["AAPL"] = chain
        pcr = p.get_put_call_ratio("AAPL", spot_price=200.0)
        assert pcr < 0.5  # heavy call bias

    def test_neutral_on_empty_chain(self):
        p = _provider()
        p._chain_cache["AAPL"] = []
        assert p.get_put_call_ratio("AAPL", spot_price=200.0) == 1.0

    def test_no_spot_uses_full_chain(self):
        p = _provider()
        chain = _make_chain(spot=200.0)
        p._chain_cache["AAPL"] = chain
        pcr_full = p.get_put_call_ratio("AAPL")  # no spot → full chain
        pcr_atm  = p.get_put_call_ratio("AAPL", spot_price=200.0)
        # Both should be numeric and non-negative
        assert pcr_full >= 0
        assert pcr_atm >= 0


# ══════════════════════════════════════════════════════════════════════════════
# get_iv_skew
# ══════════════════════════════════════════════════════════════════════════════

class TestIvSkew:

    def test_neutral_when_no_iv_data(self):
        p = _provider()
        chain = [
            OptionContract(
                ticker="O:AAPL260620C00220000",
                expiry=date(2026, 6, 20), strike=220.0, option_type="C",
            ),
        ]
        p._chain_cache["AAPL"] = chain
        assert p.get_iv_skew("AAPL", spot_price=200.0) == pytest.approx(0.5)

    def test_bullish_skew_when_call_iv_high(self):
        """Calls more expensive than puts → score > 0.5."""
        p = _provider()
        chain = [
            OptionContract(
                ticker="O:AAPL260620C00220000",
                expiry=date(2026, 6, 20), strike=220.0, option_type="C",
                iv=0.40,  # high call IV (OTM)
            ),
            OptionContract(
                ticker="O:AAPL260620P00180000",
                expiry=date(2026, 6, 20), strike=180.0, option_type="P",
                iv=0.20,  # normal put IV
            ),
        ]
        p._chain_cache["AAPL"] = chain
        score = p.get_iv_skew("AAPL", spot_price=200.0)
        assert score > 0.5

    def test_bearish_skew_when_put_iv_high(self):
        """Puts more expensive than calls → score < 0.5 (normal smirk)."""
        p = _provider()
        chain = [
            OptionContract(
                ticker="O:AAPL260620C00220000",
                expiry=date(2026, 6, 20), strike=220.0, option_type="C",
                iv=0.20,
            ),
            OptionContract(
                ticker="O:AAPL260620P00180000",
                expiry=date(2026, 6, 20), strike=180.0, option_type="P",
                iv=0.40,  # expensive puts
            ),
        ]
        p._chain_cache["AAPL"] = chain
        score = p.get_iv_skew("AAPL", spot_price=200.0)
        assert score < 0.5

    def test_skew_bounded_0_1(self):
        p = _provider()
        chain = _make_chain(spot=200.0, base_iv_call=0.90, base_iv_put=0.05)
        p._chain_cache["AAPL"] = chain
        score = p.get_iv_skew("AAPL", spot_price=200.0)
        assert 0.0 <= score <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# get_unusual_call_activity
# ══════════════════════════════════════════════════════════════════════════════

class TestUnusualCallActivity:

    def test_empty_when_volume_uniform(self):
        p = _provider()
        chain = _make_chain(spot=200.0, vol_per_contract=200.0)
        p._chain_cache["AAPL"] = chain
        assert p.get_unusual_call_activity("AAPL") == []

    def test_detects_spike(self):
        p = _provider()
        chain = [
            OptionContract(
                ticker="O:AAPL260620C00195000",
                expiry=date(2026, 6, 20), strike=195.0, option_type="C",
                volume=100.0,
            ),
            OptionContract(
                ticker="O:AAPL260620C00200000",
                expiry=date(2026, 6, 20), strike=200.0, option_type="C",
                volume=100.0,
            ),
            OptionContract(
                ticker="O:AAPL260620C00205000",
                expiry=date(2026, 6, 20), strike=205.0, option_type="C",
                volume=1000.0,  # 5× average — unusual
            ),
        ]
        p._chain_cache["AAPL"] = chain
        unusual = p.get_unusual_call_activity("AAPL")
        assert len(unusual) == 1
        assert unusual[0].strike == pytest.approx(205.0)

    def test_empty_when_fewer_than_two_calls(self):
        p = _provider()
        chain = [
            OptionContract(
                ticker="O:AAPL260620C00200000",
                expiry=date(2026, 6, 20), strike=200.0, option_type="C",
                volume=500.0,
            ),
        ]
        p._chain_cache["AAPL"] = chain
        assert p.get_unusual_call_activity("AAPL") == []


# ══════════════════════════════════════════════════════════════════════════════
# compute_options_score
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeOptionsScore:

    def test_high_score_bullish_setup(self):
        """Strongly bullish chain → score near max."""
        p = _provider()
        # PCR < 0.50: large call OI, small put OI
        chain = []
        for strike in [195.0, 200.0, 205.0]:
            chain.append(OptionContract(
                ticker=f"O:AAPL260620C{int(strike*1000):08d}",
                expiry=date(2026, 6, 20), strike=strike, option_type="C",
                open_interest=10000.0, volume=2000.0, iv=0.35,
            ))
            chain.append(OptionContract(
                ticker=f"O:AAPL260620P{int(strike*1000):08d}",
                expiry=date(2026, 6, 20), strike=strike, option_type="P",
                open_interest=1000.0, volume=100.0, iv=0.25,
            ))
        p._chain_cache["AAPL"] = chain

        with patch.object(p, "_load_cache", return_value=None):
            with patch.object(p, "_save_cache"):
                score = p.compute_options_score("AAPL", spot_price=200.0)

        assert score >= 0.60

    def test_zero_score_when_chain_empty(self):
        p = _provider()
        with patch.object(p, "_load_cache", return_value=None):
            with patch.object(p, "get_options_chain", side_effect=Exception("No data")):
                score = p.compute_options_score("AAPL", spot_price=200.0)
        assert score == 0.0

    def test_uses_cache_hit(self):
        p = _provider()
        with patch.object(p, "_load_cache", return_value=0.75):
            with patch.object(p, "get_options_chain") as mock_fetch:
                score = p.compute_options_score("AAPL", spot_price=200.0)
        assert score == pytest.approx(0.75)
        mock_fetch.assert_not_called()

    def test_score_bounded_0_1(self):
        p = _provider()
        chain = _make_chain(spot=200.0, oi_per_contract=50000.0, vol_per_contract=10000.0)
        p._chain_cache["AAPL"] = chain
        with patch.object(p, "_load_cache", return_value=None):
            with patch.object(p, "_save_cache"):
                score = p.compute_options_score("AAPL", spot_price=200.0)
        assert 0.0 <= score <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# get_options_chain (REST API)
# ══════════════════════════════════════════════════════════════════════════════

class TestGetOptionsChain:

    def _make_polygon_result(self, strike: float, opt_type: str) -> dict:
        ticker = f"O:AAPL260620{'C' if opt_type == 'C' else 'P'}{int(strike*1000):08d}"
        return {
            "details": {
                "ticker": ticker,
                "contract_type": "call" if opt_type == "C" else "put",
                "expiration_date": "2026-06-20",
                "strike_price": strike,
            },
            "greeks": {"delta": 0.5, "gamma": 0.02, "theta": -0.05, "vega": 0.10},
            "implied_volatility": 0.28,
            "last_quote": {"bid": 8.3, "ask": 8.7},
            "day": {"volume": 500},
            "open_interest": 3000,
        }

    def test_parses_single_page(self):
        p = _provider()
        mock_data = {
            "results": [
                self._make_polygon_result(200.0, "C"),
                self._make_polygon_result(195.0, "P"),
            ],
            # no next_cursor → single page
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_data
        mock_resp.raise_for_status = MagicMock()

        with patch.object(p._session, "get", return_value=mock_resp):
            chain = p.get_options_chain("AAPL")

        assert len(chain) == 2
        calls = [c for c in chain if c.option_type == "C"]
        assert len(calls) == 1
        assert calls[0].strike == pytest.approx(200.0)
        assert calls[0].iv == pytest.approx(0.28)
        assert calls[0].delta == pytest.approx(0.5)

    def test_paginates_until_no_cursor(self):
        p = _provider()
        page1 = {
            "results": [self._make_polygon_result(200.0, "C")],
            "next_cursor": "abc123",
        }
        page2 = {
            "results": [self._make_polygon_result(205.0, "C")],
            # no next_cursor → done
        }
        responses = [page1, page2]
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            r = MagicMock()
            r.json.return_value = responses[call_count]
            r.raise_for_status = MagicMock()
            call_count += 1
            return r

        with patch.object(p._session, "get", side_effect=side_effect):
            chain = p.get_options_chain("AAPL")

        assert len(chain) == 2
        assert call_count == 2  # two pages fetched

    def test_skips_malformed_ticker(self):
        p = _provider()
        mock_data = {
            "results": [
                {"details": {"ticker": "BADTICKER"}, "greeks": {}, "last_quote": {}, "day": {}},
                self._make_polygon_result(200.0, "C"),
            ]
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_data
        mock_resp.raise_for_status = MagicMock()

        with patch.object(p._session, "get", return_value=mock_resp):
            chain = p.get_options_chain("AAPL")

        assert len(chain) == 1  # bad ticker skipped


# ══════════════════════════════════════════════════════════════════════════════
# get_historical_ohlcv — REST fallback
# ══════════════════════════════════════════════════════════════════════════════

class TestHistoricalOhlcv:

    def test_rest_fallback_returns_bars(self):
        p = _provider()
        # 1780099200000 ms = 2026-05-30 00:00:00 UTC
        mock_data = {
            "results": [
                {"t": 1780099200000, "o": 8.1, "h": 8.5, "l": 7.9, "c": 8.3, "v": 400, "n": 25},
            ]
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_data
        mock_resp.raise_for_status = MagicMock()

        with patch.object(p, "_s3_historical", side_effect=ImportError("no boto3")):
            with patch.object(p._session, "get", return_value=mock_resp):
                bars = p.get_historical_ohlcv(
                    "O:AAPL260620C00310000",
                    date(2026, 5, 30),
                    date(2026, 5, 31),
                )

        assert len(bars) == 1
        assert bars[0]["close"] == pytest.approx(8.3)
        assert bars[0]["volume"] == pytest.approx(400.0)

    def test_returns_empty_on_no_results(self):
        p = _provider()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": []}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(p, "_s3_historical", side_effect=ImportError("no boto3")):
            with patch.object(p._session, "get", return_value=mock_resp):
                bars = p.get_historical_ohlcv(
                    "O:AAPL260620C00310000",
                    date(2026, 5, 30),
                    date(2026, 5, 30),
                )
        assert bars == []


# ══════════════════════════════════════════════════════════════════════════════
# DuckDB cache
# ══════════════════════════════════════════════════════════════════════════════

class TestOptionsCache:

    def test_load_cache_returns_none_when_no_db(self):
        p = _provider()
        with patch("quantlab.providers.massive_options.logger"):
            with patch("duckdb.connect", side_effect=Exception("no db")):
                result = p._load_cache("AAPL")
        assert result is None

    def test_save_cache_is_nonfatal(self):
        p = _provider()
        with patch("duckdb.connect", side_effect=Exception("no db")):
            p._save_cache("AAPL", 200.0, 0.6, 0.55, True, 0.70, 10, 8)  # should not raise

    def test_compute_options_score_uses_and_populates_cache(self):
        p = _provider()
        chain = _make_chain(spot=200.0)
        p._chain_cache["AAPL"] = chain

        saved_calls = []

        with patch.object(p, "_load_cache", return_value=None):
            with patch.object(p, "_save_cache", side_effect=lambda **kw: saved_calls.append(kw)):
                score = p.compute_options_score("AAPL", spot_price=200.0)

        assert 0.0 <= score <= 1.0
        assert len(saved_calls) == 1
        assert saved_calls[0]["symbol"] == "AAPL"
        assert saved_calls[0]["options_score"] == pytest.approx(score)


# ══════════════════════════════════════════════════════════════════════════════
# ScanResult.options_score + score_conviction integration
# ══════════════════════════════════════════════════════════════════════════════

class TestScanResultOptionsScore:

    def _base_result(self, **kwargs):
        from quantlab.execution import ScanResult
        defaults = dict(
            symbol="AAPL", scan_date="2026-06-06",
            signal_type="breakout", signal=True,
            entry_close=200.0, indicator_value=199.0, lookback=20,
            regime_bullish=True,
        )
        defaults.update(kwargs)
        return ScanResult(**defaults)

    def test_options_score_field_defaults_none(self):
        """Never-enriched = None; a measured 0.0 is real data (MISSING ≠ ZERO)."""
        r = self._base_result()
        assert hasattr(r, "options_score")
        assert r.options_score is None
        assert r.options_conviction is None

    def test_score_conviction_prefers_options_score_over_conviction(self):
        from quantlab.execution import score_conviction
        # Polygon score 0.7 (should trigger +0.10)
        r_poly = self._base_result(options_score=0.70, options_conviction=None)
        # No options data at all
        r_none = self._base_result(options_score=None, options_conviction=None)
        assert score_conviction(r_poly) > score_conviction(r_none)
        assert score_conviction(r_poly) - score_conviction(r_none) == pytest.approx(0.10)

    def test_score_conviction_strong_options_score(self):
        from quantlab.execution import score_conviction
        r_strong = self._base_result(options_score=0.85)
        r_moderate = self._base_result(options_score=0.65)
        assert score_conviction(r_strong) > score_conviction(r_moderate)
        assert score_conviction(r_strong) - score_conviction(r_moderate) == pytest.approx(0.05)

    def test_score_conviction_falls_back_to_ibkr_conviction(self):
        from quantlab.execution import score_conviction
        # Polygon never enriched (None) → IBKR's 0.7 carries the bonus
        r_ibkr = self._base_result(options_score=None, options_conviction=0.70)
        r_none = self._base_result(options_score=None, options_conviction=None)
        assert score_conviction(r_ibkr) > score_conviction(r_none)
        assert score_conviction(r_ibkr) - score_conviction(r_none) == pytest.approx(0.10)

    def test_polygon_score_overrides_ibkr_when_both_set(self):
        from quantlab.execution import score_conviction
        # Polygon=0.85 (strong), IBKR=0.65 (moderate) → use Polygon (+0.15)
        r = self._base_result(options_score=0.85, options_conviction=0.65)
        r_ibkr_only = self._base_result(options_score=None, options_conviction=0.65)
        # r gets +0.15, r_ibkr_only gets +0.10
        assert score_conviction(r) > score_conviction(r_ibkr_only)
        assert score_conviction(r) - score_conviction(r_ibkr_only) == pytest.approx(0.05)

    def test_measured_zero_polygon_score_blocks_ibkr_fallback(self):
        from quantlab.execution import score_conviction
        # Polygon measured 0.0 ("scored, no unusual flow") is information —
        # it must NOT fall through to IBKR's stale 0.7
        r_zero = self._base_result(options_score=0.0, options_conviction=0.70)
        r_none = self._base_result(options_score=None, options_conviction=None)
        assert score_conviction(r_zero) == score_conviction(r_none)   # no bonus
