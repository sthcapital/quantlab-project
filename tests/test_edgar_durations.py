"""
Tests for duration-explicit XBRL quarterly extraction and period-matched YoY.

2026-06-12 incident: the pipeline silently dropped Q4 quarters (10-Ks carry
only full-year totals; no discrete Q4 fact) and aligned YoY BY LIST POSITION
(history[i] vs history[i-4]) — with a quarter missing, "4 back" is not the
same fiscal quarter.  AEIS rendered Rev +167.5% (actual +26%), UNFI +135.9%
on revenue that actually SHRANK 4%, and the EPS acceleration score saturated
into a wall of +100.0% across dozens of candidates.

Fixtures are trimmed REAL SEC companyfacts:
    AEIS — calendar FY; Q4 exists only inside the FY total (YTD derivation).
    UNFI — early-August FYE (non-calendar), negative quarters (Jul-23/Jul-25
           era), sign flips, a near-zero base, and a dead "Revenues" tag
           (stale since 2019) that exercises freshest-field selection.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from quantlab.providers.edgar import (
    FundamentalSnapshot,
    _dated_yoy_series,
    _extract_quarterly_dated,
    _EPS_MIN_BASE,
    _REV_MIN_BASE,
    YOY_QUARANTINE,
    compute_earnings_acceleration,
    winsorize_yoy,
)

_FIX = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> dict:
    return json.loads((_FIX / f"{name}_companyfacts.json").read_text())


def _mk_facts(field: str, unit: str, obs: list[dict]) -> dict:
    return {"us-gaap": {field: {"units": {unit: obs}}}}


def _q(start: str, end: str, val: float, form: str = "10-Q", filed: str = "2026-01-01") -> dict:
    return {"start": start, "end": end, "val": val, "form": form, "filed": filed}


# ══════════════════════════════════════════════════════════════════════════════
# 1. Duration selection and YTD derivation
# ══════════════════════════════════════════════════════════════════════════════

class TestDurationSelection:

    def test_discrete_3mo_picked_over_6mo_ytd_same_end(self):
        """A Q2 10-Q reports both the discrete quarter and the 6-month YTD
        with the same end date — the ~3-month fact must win."""
        facts = _mk_facts("Revenues", "USD", [
            _q("2025-04-01", "2025-06-30", 441.5e6),   # 90d  — discrete Q2
            _q("2025-01-01", "2025-06-30", 846.1e6),   # 180d — 6mo YTD
            _q("2025-01-01", "2025-03-31", 404.6e6),   # 89d  — Q1
        ])
        dated = _extract_quarterly_dated(facts, "revenue", 8)
        assert dict(dated)[date(2025, 6, 30)] == pytest.approx(441.5e6)

    def test_q4_derived_from_fy_minus_9mo(self):
        """Most 10-Ks report no discrete Q4 — derive it from FY − 9mo YTD
        (same period start ⇒ same fiscal year)."""
        facts = _mk_facts("Revenues", "USD", [
            _q("2025-01-01", "2025-09-30", 1309.4e6),            # 9mo YTD
            _q("2025-01-01", "2025-12-31", 1798.8e6, "10-K"),    # FY total
        ])
        dated = dict(_extract_quarterly_dated(facts, "revenue", 8))
        assert dated[date(2025, 12, 31)] == pytest.approx(489.4e6)

    def test_ytd_ladder_derives_q2_and_q3(self):
        """6mo−Q1 → Q2 and 9mo−6mo → Q3 when only YTD cumulatives exist."""
        facts = _mk_facts("Revenues", "USD", [
            _q("2025-01-01", "2025-03-31", 100e6),    # Q1 (discrete = YTD)
            _q("2025-01-01", "2025-06-30", 230e6),    # 6mo
            _q("2025-01-01", "2025-09-30", 390e6),    # 9mo
        ])
        dated = dict(_extract_quarterly_dated(facts, "revenue", 8))
        assert dated[date(2025, 6, 30)] == pytest.approx(130e6)
        assert dated[date(2025, 9, 30)] == pytest.approx(160e6)

    def test_non_calendar_fiscal_year(self):
        """April-FYE (CRDO-style): start-date grouping needs no calendar
        alignment — the FY−9mo subtraction works on any fiscal calendar."""
        facts = _mk_facts("Revenues", "USD", [
            _q("2024-04-28", "2025-01-25", 300e6),            # 9mo of FY25
            _q("2024-04-28", "2025-04-26", 420e6, "10-K"),    # FY25 total
            _q("2025-04-27", "2025-07-26", 150e6),            # Q1 FY26
        ])
        dated = dict(_extract_quarterly_dated(facts, "revenue", 8))
        assert dated[date(2025, 4, 26)] == pytest.approx(120e6)   # derived Q4
        assert dated[date(2025, 7, 26)] == pytest.approx(150e6)

    def test_freshest_field_wins_over_stale_tag(self):
        """UNFI's 'Revenues' tag died in 2019 — first-tag-with-data returned
        a years-stale history.  The freshest series must win."""
        facts = {
            "us-gaap": {
                "Revenues": {"units": {"USD": [
                    _q("2019-01-27", "2019-04-27", 5962.6e6),
                ]}},
                "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
                    _q("2026-02-01", "2026-05-02", 7723e6),
                ]}},
            }
        }
        dated = _extract_quarterly_dated(facts, "revenue", 8)
        assert dated[-1][0] == date(2026, 5, 2)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Period-matched YoY
# ══════════════════════════════════════════════════════════════════════════════

def _dated(*pairs):
    return [(date.fromisoformat(d), v) for d, v in pairs]


class TestPeriodMatchedYoY:

    def test_gap_does_not_misalign(self):
        """The incident mechanism: with a quarter missing, position-based
        alignment compares the wrong quarters.  Period matching must find the
        ~365-day-earlier entry regardless of gaps."""
        series = _dated(
            ("2025-03-31", 100.0),
            ("2025-06-30", 110.0),
            ("2025-09-30", 120.0),
            # Q4-2025 missing entirely
            ("2026-03-31", 126.0),
        )
        _, latest, _ = _dated_yoy_series(series, min_base=1.0)
        assert latest == pytest.approx(0.26)     # vs 2025-03-31, same quarter

    def test_no_base_within_window_yields_none(self):
        series = _dated(("2025-09-30", 120.0), ("2026-03-31", 126.0))
        _, latest, tp = _dated_yoy_series(series, min_base=1.0)
        assert latest is None and tp is False

    def test_turned_positive_flagged_not_computed(self):
        series = _dated(("2025-05-03", -0.12), ("2026-05-02", 0.52))
        rates, latest, tp = _dated_yoy_series(series, min_base=_EPS_MIN_BASE)
        assert latest is None
        assert tp is True
        assert rates == []

    def test_negative_to_negative_is_null_no_flag(self):
        series = _dated(("2024-08-03", -0.63), ("2025-08-02", -1.44))
        _, latest, tp = _dated_yoy_series(series, min_base=_EPS_MIN_BASE)
        assert latest is None and tp is False

    def test_near_zero_base_quarantined(self):
        """UNFI Jul-24-style 0.01 base: a rate off a sub-materiality
        denominator is quarantined (NULL), never computed."""
        series = _dated(("2024-08-03", 0.01), ("2025-08-02", 0.85))
        _, latest, tp = _dated_yoy_series(series, min_base=_EPS_MIN_BASE)
        assert latest is None and tp is False

    def test_extreme_rate_quarantined_not_clamped(self):
        series = _dated(("2025-03-28", 0.19), ("2026-04-03", 23.03))
        # (23.03-0.19)/0.19 ≈ +12,021% > 1000% quarantine bound
        assert (23.03 - 0.19) / 0.19 > YOY_QUARANTINE
        _, latest, _ = _dated_yoy_series(series, min_base=_EPS_MIN_BASE)
        assert latest is None

    def test_raw_rate_stored_uncapped(self):
        series = _dated(("2025-03-31", 0.19), ("2026-03-31", 1.90))
        _, latest, _ = _dated_yoy_series(series, min_base=_EPS_MIN_BASE)
        assert latest == pytest.approx(9.0)      # +900% raw — not winsorized here


# ══════════════════════════════════════════════════════════════════════════════
# 3. Real-data fixtures — the externally verified cases
# ══════════════════════════════════════════════════════════════════════════════

class TestAEISFixture:
    """AEIS Mar-26: external check Rev +26% (404.6→511); GAAP EPS 0.65→1.58."""

    def test_q4_2025_derived(self):
        dated = dict(_extract_quarterly_dated(_load_fixture("aeis"), "revenue", 12))
        assert dated[date(2025, 12, 31)] == pytest.approx(489.4e6, rel=1e-3)

    def test_revenue_yoy_correct(self):
        dated = _extract_quarterly_dated(_load_fixture("aeis"), "revenue", 12)
        _, latest, _ = _dated_yoy_series(dated, min_base=_REV_MIN_BASE)
        assert latest == pytest.approx(0.263, abs=0.005)   # +26.3%, was +167.5%

    def test_eps_yoy_correct(self):
        dated = _extract_quarterly_dated(_load_fixture("aeis"), "eps_diluted", 12)
        assert dict(dated)[date(2026, 3, 31)] == pytest.approx(1.58)
        _, latest, tp = _dated_yoy_series(dated, min_base=_EPS_MIN_BASE)
        # GAAP 0.65 → 1.58 (the press-release +70% is the adjusted figure)
        assert latest == pytest.approx(1.431, abs=0.005)
        assert tp is False


class TestUNFIFixture:
    """UNFI Apr-26: revenue SHRANK 4% (8,059→7,723) — was reported +135.9%;
    GAAP EPS −0.12 → +0.52 = turned_positive.  Early-August FYE."""

    def test_revenue_yoy_negative(self):
        dated = _extract_quarterly_dated(_load_fixture("unfi"), "revenue", 12)
        assert dated[-1][0] == date(2026, 5, 2)   # current, not the dead tag
        _, latest, _ = _dated_yoy_series(dated, min_base=_REV_MIN_BASE)
        assert latest == pytest.approx(-0.042, abs=0.005)

    def test_eps_turned_positive(self):
        dated = _extract_quarterly_dated(_load_fixture("unfi"), "eps_diluted", 12)
        _, latest, tp = _dated_yoy_series(dated, min_base=_EPS_MIN_BASE)
        assert latest is None
        assert tp is True

    def test_non_calendar_fye_aligns(self):
        """Aug-FYE quarters end on drifting dates (2026-05-02 vs 2025-05-03);
        the 330–400-day window still pairs the same fiscal quarter."""
        dated = _extract_quarterly_dated(_load_fixture("unfi"), "revenue", 12)
        ends = [d for d, _ in dated]
        assert date(2026, 5, 2) in ends and date(2025, 5, 3) in ends


# ══════════════════════════════════════════════════════════════════════════════
# 4. Scoring semantics — winsorize, turned-positive max strength
# ══════════════════════════════════════════════════════════════════════════════

class TestScoringSemantics:

    def test_winsorize_default_cap(self):
        assert winsorize_yoy(5.0, cap=3.0) == 3.0
        assert winsorize_yoy(-5.0, cap=3.0) == -3.0
        assert winsorize_yoy(0.5, cap=3.0) == 0.5
        assert winsorize_yoy(None) is None

    def test_turned_positive_scores_max_strength(self):
        snap = FundamentalSnapshot(ticker="UNFI", cik="0", as_of=date.today())
        snap.eps_turned_positive = True
        snap.eps_yoy_pct = None
        score = compute_earnings_acceleration(snap)
        assert score == 1.0   # top band — qualifies wherever "EPS YoY ≥ X%" is asked

    def test_turned_positive_with_revenue_penalty(self):
        snap = FundamentalSnapshot(ticker="X", cik="0", as_of=date.today())
        snap.eps_turned_positive = True
        snap.revenue_yoy_pct = -0.30   # shrinking revenue still penalizes
        score = compute_earnings_acceleration(snap)
        assert score == pytest.approx(0.7)   # 1.0 − 0.30

    def test_extreme_yoy_winsorized_at_scoring(self):
        snap = FundamentalSnapshot(ticker="X", cik="0", as_of=date.today())
        snap.eps_yoy_history = [25.0]   # +2500% raw — stored raw, scored winsorized
        score = compute_earnings_acceleration(snap)
        assert score == 1.0   # lands in the >100% band, not an error


# ══════════════════════════════════════════════════════════════════════════════
# 5. Display rendering
# ══════════════════════════════════════════════════════════════════════════════

class TestDisplayRendering:

    @staticmethod
    def _gr():
        import importlib.util
        root = Path(__file__).parent.parent
        spec = importlib.util.spec_from_file_location(
            "generate_report", root / "scripts" / "generate_report.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_eps_cell_variants(self):
        gr = self._gr()
        assert gr._eps_cell({"eps_yoy": 1.431, "turned_positive": False}) == "+143.1%"
        assert gr._eps_cell({"eps_yoy": 25.0, "turned_positive": False}) == ">999%"
        assert gr._eps_cell({"eps_yoy": None, "turned_positive": True}) == "−→+"
        assert gr._eps_cell({"eps_yoy": None, "turned_positive": False}) == "—"
        assert gr._eps_cell(None) == "—"

    def test_rev_pct_renders_real_hypergrowth(self):
        """SNDK +251% is real (FactSet-confirmed) — display must not suppress
        it as N/A; quarantine happens upstream, display only handles overflow."""
        gr = self._gr()
        assert gr._rev_pct(2.51) == "+251.0%"
        assert gr._rev_pct(25.0) == ">999%"
        assert gr._rev_pct(None) == "—"
        assert gr._rev_pct(-0.042) == "-4.2%"
