"""
Tests for the recalibrated relative options scorer (2026-06).

Covers the saturation fix for the absolute-threshold detector that flagged
347/357 monitored symbols (97%) on 2026-06-11 and scored 81% of the
watchlist ≥ 0.6:

  1. Per-symbol z-score math on synthetic series (MISSING ≠ ZERO: short
     baseline → None, never 0.0).
  2. Cross-sectional percentile gate caps the daily flag rate by
     construction, and is a cap — not a quota — when z-scores are supplied.
  3. options_signal_gating_enabled defaults to False (display-only).
  4. The report's signal-rate header line renders from options_snapshots,
     including the schema migration and the legacy fallback.
"""

from __future__ import annotations

import importlib.util
import statistics
from datetime import date
from pathlib import Path

import duckdb
import pytest

from quantlab.signals.options_relative import (
    MIN_BASELINE_SESSIONS,
    MIN_FLAG_ZSCORE,
    MIN_TODAY_CALL_VOLUME,
    ZSCORE_CAP,
    cross_sectional_flags,
    percentile,
    relative_options_score,
    volume_zscore,
)

_ROOT = Path(__file__).parent.parent


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ══════════════════════════════════════════════════════════════════════════════
# 1. Per-symbol z-score math
# ══════════════════════════════════════════════════════════════════════════════

class TestVolumeZscore:

    BASELINE = [800.0, 1200.0, 900.0, 1100.0, 1000.0,
                950.0, 1050.0, 850.0, 1150.0, 1000.0,
                900.0, 1100.0, 1000.0, 950.0, 1050.0,
                800.0, 1200.0, 900.0, 1100.0, 1000.0]   # mean 1000

    def test_exact_zscore_on_synthetic_series(self):
        mean = statistics.mean(self.BASELINE)
        std = statistics.stdev(self.BASELINE)
        today = mean + 3.0 * std
        assert volume_zscore(today, self.BASELINE) == pytest.approx(3.0)

    def test_three_times_own_normal_volume_is_a_strong_signal(self):
        z = volume_zscore(3000.0, self.BASELINE)
        assert z is not None and z > 4.0

    def test_at_baseline_mean_is_zero(self):
        assert volume_zscore(1000.0, self.BASELINE) == pytest.approx(0.0)

    def test_below_baseline_is_negative(self):
        z = volume_zscore(500.0, self.BASELINE)
        assert z is not None and z < 0.0

    def test_short_baseline_returns_none_not_zero(self):
        # MISSING ≠ ZERO: no history → no claim, never 0.0
        short = self.BASELINE[: MIN_BASELINE_SESSIONS - 1]
        assert volume_zscore(3000.0, short) is None

    def test_empty_baseline_returns_none(self):
        assert volume_zscore(3000.0, []) is None

    def test_none_volume_returns_none(self):
        assert volume_zscore(None, self.BASELINE) is None

    def test_materiality_floor_pins_micro_volume_to_zero(self):
        # 12 contracts over a 2-contract baseline is noise, not accumulation
        # (TRNO flagged on 8 contracts in the first 06-11 rescore pass)
        thin = [2.0] * 19 + [3.0]
        assert volume_zscore(12.0, thin) == 0.0

    def test_volume_at_floor_is_scored(self):
        z = volume_zscore(MIN_TODAY_CALL_VOLUME, [10.0] * 19 + [12.0])
        assert z is not None and z > 0.0

    def test_zero_variance_baseline(self):
        flat = [500.0] * 20
        assert volume_zscore(500.0, flat) == 0.0
        assert volume_zscore(800.0, flat) == ZSCORE_CAP
        assert volume_zscore(200.0, flat) == -ZSCORE_CAP

    def test_zscore_capped(self):
        z = volume_zscore(1_000_000.0, self.BASELINE)
        assert z == ZSCORE_CAP


# ══════════════════════════════════════════════════════════════════════════════
# 2. Composite relative score
# ══════════════════════════════════════════════════════════════════════════════

class TestRelativeScore:

    def test_none_zscore_returns_none(self):
        # MISSING ≠ ZERO: without the volume anomaly there is no signal
        assert relative_options_score(None, pcr=0.1, iv_skew=0.9) is None

    def test_maximum_score(self):
        # 4σ volume + zero puts + max skew = 1.0 on every component
        assert relative_options_score(4.0, pcr=0.0, iv_skew=1.0) == 1.0

    def test_quiet_bearish_chain_scores_low(self):
        score = relative_options_score(0.0, pcr=5.0, iv_skew=0.0)
        assert score is not None and score < 0.05

    def test_monotone_in_zscore(self):
        scores = [relative_options_score(z, pcr=1.0, iv_skew=0.5)
                  for z in (0.0, 1.0, 2.0, 3.0, 4.0)]
        assert scores == sorted(scores)

    def test_weights_renormalised_when_tilts_missing(self):
        # vol z alone: 2σ → component 0.5 → score exactly 0.5
        assert relative_options_score(2.0) == pytest.approx(0.5)

    def test_bounds(self):
        for z in (-10.0, 0.0, 4.0, 10.0):
            for pcr in (0.0, 1.0, 50.0):
                for skew in (0.0, 0.5, 1.0):
                    s = relative_options_score(z, pcr=pcr, iv_skew=skew)
                    assert 0.0 <= s <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# 3. Cross-sectional percentile gate
# ══════════════════════════════════════════════════════════════════════════════

def _universe(n: int, z: float = 5.0) -> tuple[dict, dict]:
    """n symbols with distinct scores 1/n … 1.0, all z-eligible."""
    scores = {f"S{i:03d}": (i + 1) / n for i in range(n)}
    zscores = {sym: z for sym in scores}
    return scores, zscores


class TestCrossSectionalFlags:

    def test_p90_flags_top_decile_exactly(self):
        scores, zscores = _universe(100)
        flagged = cross_sectional_flags(scores, percentile_cut=90.0, zscores=zscores)
        assert len(flagged) == 10
        assert flagged == {f"S{i:03d}" for i in range(90, 100)}

    @pytest.mark.parametrize("n", [20, 100, 357, 1000])
    @pytest.mark.parametrize("pctl", [80.0, 90.0, 95.0])
    def test_rate_capped_by_construction(self, n, pctl):
        scores, zscores = _universe(n)
        flagged = cross_sectional_flags(scores, percentile_cut=pctl, zscores=zscores)
        assert len(flagged) <= n * (100.0 - pctl) / 100.0 + 1

    def test_percentile_configurable(self):
        scores, zscores = _universe(100)
        flagged = cross_sectional_flags(scores, percentile_cut=80.0, zscores=zscores)
        assert len(flagged) == 20

    def test_none_scores_excluded(self):
        scores, zscores = _universe(100)
        scores["NOBASE"] = None
        flagged = cross_sectional_flags(scores, zscores=zscores)
        assert "NOBASE" not in flagged

    def test_small_universe_flags_nothing(self):
        scores, zscores = _universe(9)
        assert cross_sectional_flags(scores, zscores=zscores) == set()

    def test_degenerate_all_equal_flags_nothing(self):
        # Most scores tying at the threshold must flag nothing, not everything —
        # this is the saturation failure mode the recalibration exists to prevent
        scores = {f"S{i}": 0.85 for i in range(100)}
        assert cross_sectional_flags(scores) == set()

    def test_cap_not_quota_quiet_day_flags_nothing(self):
        # All symbols below MIN_FLAG_ZSCORE: the decile must not be filled
        # with non-anomalies (FITB on 06-11: z = −0.72, carried by IV skew)
        scores, _ = _universe(100)
        zscores = {sym: 1.0 for sym in scores}
        assert cross_sectional_flags(scores, zscores=zscores) == set()

    def test_z_eligibility_filters_within_top_decile(self):
        scores, zscores = _universe(100)
        zscores["S099"] = MIN_FLAG_ZSCORE - 0.1   # top score, ineligible z
        flagged = cross_sectional_flags(scores, zscores=zscores)
        assert "S099" not in flagged
        assert "S098" in flagged

    def test_without_zscores_pure_percentile(self):
        scores, _ = _universe(100)
        assert len(cross_sectional_flags(scores)) == 10

    # ── Liquidity floor (gate eligibility only) ───────────────────────────────

    def test_tiny_baseline_blocked_from_flag(self):
        """The EG case (2026-06-11): z=10 on a 24-contract baseline — one
        hedger rolling a position, not accumulation.  Scored, but no flag."""
        scores, zscores = _universe(100)
        base_means = {sym: 500.0 for sym in scores}
        base_means["S099"] = 24.0   # top score, illiquid baseline
        flagged = cross_sectional_flags(
            scores, zscores=zscores, baseline_means=base_means,
        )
        assert "S099" not in flagged
        assert "S098" in flagged   # liquid neighbours unaffected

    def test_baseline_at_floor_is_eligible(self):
        scores, zscores = _universe(100)
        base_means = {sym: 75.0 for sym in scores}   # exactly at the default floor
        flagged = cross_sectional_flags(
            scores, zscores=zscores, baseline_means=base_means,
        )
        assert len(flagged) == 10

    def test_floor_configurable(self):
        scores, zscores = _universe(100)
        base_means = {sym: 50.0 for sym in scores}
        assert cross_sectional_flags(
            scores, zscores=zscores, baseline_means=base_means, min_baseline=40.0,
        )   # all eligible at a 40-contract floor
        assert cross_sectional_flags(
            scores, zscores=zscores, baseline_means=base_means, min_baseline=60.0,
        ) == set()

    def test_unknown_baseline_mean_blocked(self):
        """No baseline mean recorded → cannot make the liquidity claim → no flag."""
        scores, zscores = _universe(100)
        base_means = {sym: 500.0 for sym in scores}
        base_means["S099"] = None
        flagged = cross_sectional_flags(
            scores, zscores=zscores, baseline_means=base_means,
        )
        assert "S099" not in flagged

    def test_no_baseline_means_keeps_prior_behavior(self):
        scores, zscores = _universe(100)
        assert len(cross_sectional_flags(scores, zscores=zscores)) == 10

    # ── Direction ceiling (PCR — gate eligibility only) ───────────────────────

    def test_pcr_above_ceiling_blocks_flag(self):
        """The HST case (2026-06-11): z=10 with PCR 6.25 — put-dominated flow
        is not LONG-accumulation evidence.  Scored, but no flag."""
        scores, zscores = _universe(100)
        pcrs = {sym: 0.5 for sym in scores}
        pcrs["S099"] = 6.25   # top score, put-dominated session
        flagged = cross_sectional_flags(scores, zscores=zscores, pcrs=pcrs)
        assert "S099" not in flagged
        assert "S098" in flagged   # call-dominated neighbours unaffected

    def test_pcr_below_ceiling_passes(self):
        scores, zscores = _universe(100)
        pcrs = {sym: 1.5 for sym in scores}   # exactly at the default ceiling
        flagged = cross_sectional_flags(scores, zscores=zscores, pcrs=pcrs)
        assert len(flagged) == 10

    def test_pcr_ceiling_configurable(self):
        scores, zscores = _universe(100)
        pcrs = {sym: 2.0 for sym in scores}
        assert cross_sectional_flags(
            scores, zscores=zscores, pcrs=pcrs, max_pcr=2.5,
        )
        assert cross_sectional_flags(
            scores, zscores=zscores, pcrs=pcrs, max_pcr=1.5,
        ) == set()

    def test_unknown_pcr_passes(self):
        """No PCR measured → no evidence of put domination → eligible."""
        scores, zscores = _universe(100)
        pcrs = {sym: 0.5 for sym in scores}
        pcrs["S099"] = None
        flagged = cross_sectional_flags(scores, zscores=zscores, pcrs=pcrs)
        assert "S099" in flagged

    def test_percentile_helper(self):
        vals = list(range(1, 101))
        assert percentile(vals, 50.0) == pytest.approx(50.5)
        assert percentile(vals, 90.0) == pytest.approx(90.1)
        assert percentile([7.0], 90.0) == 7.0
        with pytest.raises(ValueError):
            percentile([], 90.0)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Gating config defaults — display-only until reviewed
# ══════════════════════════════════════════════════════════════════════════════

class TestGatingConfig:

    def test_gating_disabled_by_default(self):
        from quantlab.utils import get_config
        assert get_config("scanner").get("options_signal_gating_enabled") is False

    def test_percentile_default_is_p90(self):
        from quantlab.utils import get_config
        assert get_config("scanner").get("options_unusual_percentile") == 90.0


# ══════════════════════════════════════════════════════════════════════════════
# 5. Report signal-rate header line
# ══════════════════════════════════════════════════════════════════════════════

class TestReportSignalRate:

    def _seed(self, db_path, snap_date: date, n_total: int, n_flagged: int,
              relative: bool = True) -> None:
        from quantlab.providers.massive_options import MassiveOptionsProvider
        con = duckdb.connect(str(db_path))
        # Legacy table first, then the migration — exercises ADD COLUMN
        con.execute("""
            CREATE TABLE IF NOT EXISTS options_snapshots (
                symbol VARCHAR, snap_date DATE, spot_price DOUBLE,
                pcr DOUBLE, iv_skew DOUBLE, unusual_calls BOOLEAN,
                options_score DOUBLE, call_count INTEGER, put_count INTEGER,
                PRIMARY KEY (symbol, snap_date)
            )
        """)
        MassiveOptionsProvider._ensure_table(con)
        for i in range(n_total):
            flagged = i < n_flagged
            # Explicit column list — unnamed columns get their NULL defaults,
            # so schema additions don't break this seed
            con.execute(
                "INSERT OR REPLACE INTO options_snapshots "
                "(symbol, snap_date, spot_price, pcr, iv_skew, unusual_calls, "
                " options_score, call_count, put_count, call_volume, "
                " put_volume, vol_zscore, rel_score, unusual_flag) "
                "VALUES (?, ?, 100.0, 0.5, 0.2, ?, 0.85, 10, 5, ?, ?, ?, ?, ?)",
                [f"SYM{i:03d}", snap_date, flagged,
                 5000.0 if relative else None,
                 3000.0 if relative else None,
                 (3.0 if flagged else 0.5) if relative else None,
                 (0.8 if flagged else 0.3) if relative else None,
                 flagged if relative else None],
            )
        con.close()

    def test_rate_line_renders(self, tmp_path):
        gr = _load_script("generate_report")
        db = tmp_path / "test.duckdb"
        self._seed(db, date(2026, 6, 12), n_total=357, n_flagged=31)
        # No options_session_status row → session is unfinalized, so the header
        # carries the intraday basis suffix (item 3d).
        line = gr._options_signal_rate(date(2026, 6, 12), str(db))
        assert line == "Options: 31/357 unusual, 8.7% (intraday — finalizes overnight)"

    def test_rate_line_marks_final_basis(self, tmp_path):
        import duckdb
        from quantlab.options_finalize import BASIS_FINAL, set_session_status
        gr = _load_script("generate_report")
        db = tmp_path / "test.duckdb"
        self._seed(db, date(2026, 6, 12), n_total=357, n_flagged=31)
        con = duckdb.connect(str(db))
        set_session_status(con, date(2026, 6, 12), finalized=True, basis=BASIS_FINAL)
        con.close()
        line = gr._options_signal_rate(date(2026, 6, 12), str(db))
        assert line == "Options: 31/357 unusual, 8.7% (final)"

    def test_legacy_fallback_for_prerecalibration_sessions(self, tmp_path):
        gr = _load_script("generate_report")
        db = tmp_path / "test.duckdb"
        self._seed(db, date(2026, 6, 10), n_total=315, n_flagged=0, relative=False)
        line = gr._options_signal_rate(date(2026, 6, 10), str(db))
        assert line is not None and "(legacy scorer)" in line

    def test_none_when_no_snapshots(self, tmp_path):
        gr = _load_script("generate_report")
        db = tmp_path / "test.duckdb"
        self._seed(db, date(2026, 6, 11), n_total=5, n_flagged=1)
        assert gr._options_signal_rate(date(2026, 6, 12), str(db)) is None

    def test_none_when_table_missing(self, tmp_path):
        gr = _load_script("generate_report")
        db = tmp_path / "empty.duckdb"
        duckdb.connect(str(db)).close()
        assert gr._options_signal_rate(date(2026, 6, 12), str(db)) is None


class TestPutDominatedTag:
    """The put_dominated snapshot tag — short-side signal data persistence."""

    def _seed_snapshots(self, db_path, snap_date, symbols):
        from quantlab.providers.massive_options import MassiveOptionsProvider
        con = duckdb.connect(str(db_path))
        MassiveOptionsProvider._ensure_table(con)
        for sym in symbols:
            con.execute(
                "INSERT INTO options_snapshots "
                "(symbol, snap_date, spot_price, pcr, iv_skew, rel_score) "
                "VALUES (?, ?, 100.0, 0.5, 0.1, 0.8)",
                [sym, snap_date],
            )
        # One unscored row — must keep NULL tags (MISSING ≠ ZERO)
        con.execute(
            "INSERT INTO options_snapshots (symbol, snap_date, spot_price) "
            "VALUES ('NOSCORE', ?, 100.0)",
            [snap_date],
        )
        con.close()

    def test_put_dominated_tag_persists(self, tmp_path, monkeypatch):
        import quantlab.storage as storage
        from quantlab.providers.massive_options import MassiveOptionsProvider
        db = tmp_path / "test.duckdb"
        monkeypatch.setattr(storage, "DB_PATH", db)
        day = date(2026, 6, 11)
        self._seed_snapshots(db, day, ["HST", "ASH", "CB"])

        mp = MassiveOptionsProvider(api_key="test")
        mp.mark_unusual_flags({"ASH", "CB"}, snap_date=day, put_dominated={"HST"})

        con = duckdb.connect(str(db))
        rows = dict(con.execute(
            "SELECT symbol, put_dominated FROM options_snapshots WHERE snap_date = ?",
            [day],
        ).fetchall())
        flags = dict(con.execute(
            "SELECT symbol, unusual_flag FROM options_snapshots WHERE snap_date = ?",
            [day],
        ).fetchall())
        con.close()

        assert rows["HST"] is True          # tagged — short-side record exists
        assert flags["HST"] is False        # but no LONG flag / gate credit
        assert rows["ASH"] is False and flags["ASH"] is True
        assert rows["NOSCORE"] is None      # unscored stays NULL
