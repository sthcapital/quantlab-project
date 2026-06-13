"""
Tests for the growth-stock pre-filter (quantlab.growth_filter).

The filter inverts the candidate funnel: a two-tier growth pre-filter runs
BEFORE stage analysis so every downstream signal is computed only on the
$1B–$10B / high-ADR / liquid / growth population the system was designed for.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pytest

from quantlab.growth_filter import (
    FunnelCounts,
    GrowthFacts,
    GrowthFilterConfig,
    QUALIFIED,
    UNQUALIFIED_DATA,
    FAILED_GROWTH,
    FAILED_CAP,
    FAILED_ADR,
    FAILED_LIQUIDITY,
    build_growth_universe,
    compute_adr_dollar_vol,
    compute_funnel,
    evaluate_symbol,
    passes_cap_band,
    qualify_growth,
)

CFG = GrowthFilterConfig()
B = 1_000_000_000.0


def _facts(**kw) -> GrowthFacts:
    """A symbol that PASSES every Tier-1 gate by default; override per test."""
    base = dict(
        symbol="X",
        market_cap=5 * B,        # core band
        adr_pct=0.05,            # ≥ 3.5%
        dollar_vol=50_000_000,   # ≥ $20M
        price=50.0,              # ≥ $10
    )
    base.update(kw)
    return GrowthFacts(**base)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Tier-1 gate boundaries
# ══════════════════════════════════════════════════════════════════════════════

class TestCapBand:

    @pytest.mark.parametrize("cap,ok", [
        (0.9 * 1e9, False),   # below $1B
        (1.0 * 1e9, True),    # exactly $1B
        (5.0 * 1e9, True),    # core
        (10.0 * 1e9, True),   # exactly $10B
        (11.0 * 1e9, False),  # above core, no ADR rescue tested here
    ])
    def test_core_band_endpoints(self, cap, ok):
        # ADR below the soft-band floor so only the core band can pass it
        assert passes_cap_band(_facts(market_cap=cap, adr_pct=0.04), CFG) is ok

    def test_soft_band_30b_with_adr_5pct_passes(self):
        assert passes_cap_band(_facts(market_cap=30 * B, adr_pct=0.05), CFG) is True

    def test_soft_band_30b_with_adr_4pct_fails(self):
        assert passes_cap_band(_facts(market_cap=30 * B, adr_pct=0.04), CFG) is False

    def test_above_soft_band_55b_fails_even_with_high_adr(self):
        assert passes_cap_band(_facts(market_cap=55 * B, adr_pct=0.12), CFG) is False

    def test_missing_cap_fails_gate(self):
        assert passes_cap_band(_facts(market_cap=None), CFG) is False


class TestADRAndLiquidity:

    def test_adr_boundary(self):
        assert evaluate_symbol(_facts(adr_pct=0.034, market_cap=5 * B), CFG).pass_adr is False
        assert evaluate_symbol(_facts(adr_pct=0.035, market_cap=5 * B), CFG).pass_adr is True

    def test_dollar_vol_at_threshold(self):
        assert evaluate_symbol(_facts(dollar_vol=19_999_999), CFG).pass_liquidity is False
        assert evaluate_symbol(_facts(dollar_vol=20_000_000), CFG).pass_liquidity is True

    def test_price_floor(self):
        assert evaluate_symbol(_facts(price=9.99), CFG).pass_liquidity is False
        assert evaluate_symbol(_facts(price=10.0), CFG).pass_liquidity is True


class TestBucketOrdering:
    """A symbol is bucketed at the FIRST gate it fails, in funnel order."""

    def test_fails_liquidity_first(self):
        r = evaluate_symbol(_facts(dollar_vol=1, market_cap=None, adr_pct=0.0), CFG)
        assert r.bucket == FAILED_LIQUIDITY

    def test_fails_cap_when_liquid(self):
        r = evaluate_symbol(_facts(market_cap=200 * B, adr_pct=0.0), CFG)
        assert r.bucket == FAILED_CAP

    def test_fails_adr_when_cap_ok(self):
        r = evaluate_symbol(_facts(adr_pct=0.01), CFG)
        assert r.bucket == FAILED_ADR


# ══════════════════════════════════════════════════════════════════════════════
# 2. Tier-2 growth qualification + acceleration
# ══════════════════════════════════════════════════════════════════════════════

class TestGrowthQualification:

    def test_eps_yoy_qualifies(self):
        b, _ = qualify_growth(_facts(eps_yoy=0.30, rev_yoy=0.05), CFG)
        assert b == QUALIFIED

    def test_revenue_yoy_qualifies(self):
        b, reasons = qualify_growth(_facts(eps_yoy=0.0, rev_yoy=0.25), CFG)
        assert b == QUALIFIED and "rev" in reasons

    def test_turned_positive_qualifies(self):
        b, reasons = qualify_growth(
            _facts(eps_yoy=None, rev_yoy=0.0, turned_positive=True), CFG)
        assert b == QUALIFIED and "turned_positive" in reasons

    def test_below_thresholds_fails_growth(self):
        b, _ = qualify_growth(_facts(eps_yoy=0.10, rev_yoy=0.10), CFG)
        assert b == FAILED_GROWTH

    def test_rising_yoy_qualifies_at_15pct(self):
        """Acceleration: latest YoY rate > prior quarter's, latest ≥ +15%."""
        b, reasons = qualify_growth(
            _facts(rev_yoy=0.15, eps_yoy=0.05, rev_accel=True, eps_accel=False), CFG)
        assert b == QUALIFIED and "rev_accel" in reasons

    def test_flat_10pct_does_not_qualify(self):
        b, _ = qualify_growth(
            _facts(rev_yoy=0.10, eps_yoy=0.10, rev_accel=False, eps_accel=False), CFG)
        assert b == FAILED_GROWTH

    def test_accel_below_floor_does_not_qualify(self):
        """Accelerating but latest YoY only +12% (< +15% floor) → no qualifier."""
        b, _ = qualify_growth(
            _facts(rev_yoy=0.12, eps_yoy=0.05, rev_accel=True, eps_accel=False), CFG)
        assert b == FAILED_GROWTH


class TestIPOPath:

    def test_recent_ipo_qualifies_on_revenue_alone(self):
        """≥2 quarters, EPS negative — qualify on revenue, no EPS positivity."""
        b, reasons = qualify_growth(
            _facts(n_quarters=3, rev_yoy=0.40, eps_yoy=-0.80), CFG)
        assert b == QUALIFIED and "ipo_rev" in reasons

    def test_ipo_weak_revenue_fails_growth(self):
        b, _ = qualify_growth(_facts(n_quarters=3, rev_yoy=0.05), CFG)
        assert b == FAILED_GROWTH

    def test_ipo_no_revenue_is_unqualified_data(self):
        b, _ = qualify_growth(_facts(n_quarters=3, rev_yoy=None), CFG)
        assert b == UNQUALIFIED_DATA

    def test_too_new_is_unqualified_data(self):
        b, _ = qualify_growth(_facts(n_quarters=1, rev_yoy=0.90), CFG)
        assert b == UNQUALIFIED_DATA


# ══════════════════════════════════════════════════════════════════════════════
# 3. Missing-data routing (MISSING ≠ ZERO)
# ══════════════════════════════════════════════════════════════════════════════

class TestMissingDataRouting:

    def test_foreign_filer_null_everything_is_unqualified_data(self):
        """20-F/40-F filer: no GAAP YoY at all → unqualified-data, NOT failed."""
        b, _ = qualify_growth(
            _facts(eps_yoy=None, rev_yoy=None, turned_positive=False,
                   rev_accel=None, eps_accel=None, n_quarters=None), CFG)
        assert b == UNQUALIFIED_DATA

    def test_quarantined_yoy_is_unqualified_data(self):
        """Quarantined YoY stored NULL (EDGAR fix) → unavailable, not zero."""
        r = evaluate_symbol(
            _facts(eps_yoy=None, rev_yoy=None, turned_positive=False), CFG)
        assert r.bucket == UNQUALIFIED_DATA

    def test_partial_data_revenue_present_is_judged(self):
        """EPS quarantined (NULL) but revenue present & weak → real failed_growth."""
        b, _ = qualify_growth(_facts(eps_yoy=None, rev_yoy=0.05), CFG)
        assert b == FAILED_GROWTH


# ══════════════════════════════════════════════════════════════════════════════
# 4. Funnel counts + render
# ══════════════════════════════════════════════════════════════════════════════

class TestFunnel:

    def _mixed(self) -> list:
        return [
            evaluate_symbol(_facts(symbol="QUAL", eps_yoy=0.50), CFG),       # qualified
            evaluate_symbol(_facts(symbol="UNQ", eps_yoy=None, rev_yoy=None,
                                   turned_positive=False), CFG),             # unq-data
            evaluate_symbol(_facts(symbol="FG", eps_yoy=0.05, rev_yoy=0.05), CFG),  # failed growth
            evaluate_symbol(_facts(symbol="ADR", adr_pct=0.01), CFG),        # failed adr
            evaluate_symbol(_facts(symbol="CAP", market_cap=300 * B,
                                   adr_pct=0.0), CFG),                       # failed cap
            evaluate_symbol(_facts(symbol="LIQ", dollar_vol=1), CFG),        # failed liquidity
        ]

    def test_counts_sum_correctly(self):
        f = compute_funnel(self._mixed())
        assert f.total == 6
        assert f.after_liquidity == 5      # all but LIQ
        assert f.after_cap == 4            # minus CAP
        assert f.after_adr == 3            # minus ADR — Tier-1 survivors
        # Tier-1 survivors partition cleanly into the three growth buckets
        assert f.after_adr == f.growth_qualified + f.unqualified_data + f.failed_growth
        assert (f.growth_qualified, f.unqualified_data, f.failed_growth) == (1, 1, 1)

    def test_render_shows_binding_constraint(self):
        f = FunnelCounts(total=1477, after_liquidity=900, after_cap=300,
                         after_adr=180, growth_qualified=42, unqualified_data=15)
        s = f.render()
        assert "Universe 1,477" in s
        assert "growth-qualified 42" in s
        assert "15 unqualified-data" in s


# ══════════════════════════════════════════════════════════════════════════════
# 5. ADR / dollar-vol computation from bars
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _Bar:
    as_of: date
    open: float
    high: float
    low: float
    close: float
    volume: float


class TestADRComputation:

    def test_adr_and_dollar_vol(self):
        bars = [
            _Bar(date(2026, 6, 1), 100, 110, 100, 105, 1_000_000),  # range 10%
            _Bar(date(2026, 6, 2), 105, 110, 100, 108, 2_000_000),  # range 10%
        ]
        adr, dvol, price = compute_adr_dollar_vol(bars, CFG)
        assert adr == pytest.approx(0.10)
        assert dvol == pytest.approx((105e6 + 216e6) / 2)
        assert price == 108           # most recent close

    def test_empty_history(self):
        assert compute_adr_dollar_vol([], CFG) == (None, None, None)

    def test_unordered_bars_use_latest_close(self):
        bars = [
            _Bar(date(2026, 6, 2), 105, 110, 100, 108, 2_000_000),
            _Bar(date(2026, 6, 1), 100, 110, 100, 105, 1_000_000),
        ]
        _, _, price = compute_adr_dollar_vol(bars, CFG)
        assert price == 108


# ══════════════════════════════════════════════════════════════════════════════
# 6. Bypass flag
# ══════════════════════════════════════════════════════════════════════════════

class TestBypass:

    def _facts_map(self):
        return {
            "GOOD": _facts(symbol="GOOD", eps_yoy=0.50),
            "BAD":  _facts(symbol="BAD", market_cap=300 * B, adr_pct=0.0),
        }

    def test_bypass_on_returns_full_universe(self):
        cfg = GrowthFilterConfig(bypass=True)
        syms, results, funnel = build_growth_universe(
            ["GOOD", "BAD"], cfg=cfg, persist=False,
            facts_by_symbol=self._facts_map(),
        )
        assert set(syms) == {"GOOD", "BAD"}        # full universe returned
        assert funnel.growth_qualified == 1        # but funnel still computed

    def test_bypass_off_returns_qualified_only(self):
        cfg = GrowthFilterConfig(bypass=False)
        syms, results, funnel = build_growth_universe(
            ["GOOD", "BAD"], cfg=cfg, persist=False,
            facts_by_symbol=self._facts_map(),
        )
        assert syms == ["GOOD"]                     # only the qualified subset


# ══════════════════════════════════════════════════════════════════════════════
# 7. Config overrides
# ══════════════════════════════════════════════════════════════════════════════

class TestConfig:

    def test_defaults(self):
        c = GrowthFilterConfig()
        assert c.cap_min == 1e9 and c.cap_core_max == 1e10
        assert c.adr_min == 0.035 and c.min_dollar_vol == 2e7
        assert c.bypass is True

    def test_revenue_weighted_at_least_as_heavily_as_eps(self):
        c = GrowthFilterConfig()
        assert c.rank_rev_weight >= c.rank_eps_weight

    def test_from_config_default_bypass_on(self):
        assert GrowthFilterConfig.from_config().bypass is True


# ══════════════════════════════════════════════════════════════════════════════
# 8. Persistence round-trip (DuckDB)
# ══════════════════════════════════════════════════════════════════════════════

class TestPersistence:

    def _db(self, tmp_path):
        return str(tmp_path / "gf.duckdb")

    def test_save_load_funnel_and_excluded(self, tmp_path):
        from quantlab.growth_filter import (
            save_growth_universe, load_growth_funnel, load_excluded_defensive,
        )
        as_of = date(2026, 6, 12)
        results = [
            evaluate_symbol(_facts(symbol="QUAL", eps_yoy=0.50), CFG),
            evaluate_symbol(_facts(symbol="UNQ", eps_yoy=None, rev_yoy=None,
                                   turned_positive=False), CFG),
            evaluate_symbol(_facts(symbol="KO", market_cap=300 * B, adr_pct=0.017,
                                   dollar_vol=1_000_000_000), CFG),   # failed_cap
            evaluate_symbol(_facts(symbol="UTIL", adr_pct=0.01,
                                   dollar_vol=100_000_000), CFG),     # failed_adr
        ]
        db = self._db(tmp_path)
        save_growth_universe(as_of, results, db_path=db)

        f = load_growth_funnel(as_of, db_path=db)
        assert f is not None
        assert f.total == 4
        assert f.growth_qualified == 1
        assert f.unqualified_data == 1

        # KO (huge dollar_vol) leads the excluded-defensive context list
        excl = load_excluded_defensive(as_of, limit=10, db_path=db)
        syms = [r["symbol"] for r in excl]
        assert "KO" in syms and "UTIL" in syms
        assert syms[0] == "KO"          # ordered by dollar_vol desc

    def test_save_is_idempotent_per_date(self, tmp_path):
        from quantlab.growth_filter import save_growth_universe, load_growth_funnel
        as_of = date(2026, 6, 12)
        db = self._db(tmp_path)
        r = [evaluate_symbol(_facts(symbol="QUAL", eps_yoy=0.50), CFG)]
        save_growth_universe(as_of, r, db_path=db)
        save_growth_universe(as_of, r, db_path=db)   # re-run same date
        f = load_growth_funnel(as_of, db_path=db)
        assert f.total == 1                          # not duplicated
