"""
Tests for universe build stability — completeness check, sanity gate, and
streak neutrality.

Background (2026-06-12 diagnosis): universe builds ran pre-market against
Polygon's same-day grouped aggregates, producing 457 → 2,325 → 1,694 →
1,079 → 1,477 symbol swings on fixed thresholds.  Two failure variants:
partial volumes (06-10 build saw a median 17% of final dollar-volume) and a
truncated 457-ticker response read mid-publication (06-04).
"""

from __future__ import annotations

import importlib.util
from datetime import date, datetime, timedelta
from pathlib import Path

import duckdb
import pytest

from quantlab.providers.base import Bar
from quantlab.universe import (
    UniverseManager,
    gate_baseline_median,
    most_recent_completed_session,
    universe_gate_check,
)
from quantlab.watchlist import _trading_days_elapsed

_ROOT = Path(__file__).parent.parent


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _seed_history(db_path, rows):
    """Insert (date, final_count, gate_accepted) rows into universe_history."""
    from quantlab.storage import _ensure_schema
    con = duckdb.connect(str(db_path))
    _ensure_schema(con)
    for d, fc, acc in rows:
        con.execute(
            "INSERT OR REPLACE INTO universe_history (date, final_count, gate_accepted) "
            "VALUES (?, ?, ?)",
            [d, fc, acc],
        )
    con.close()


# ══════════════════════════════════════════════════════════════════════════════
# universe_gate_check — pure sanity gate
# ══════════════════════════════════════════════════════════════════════════════

class TestUniverseGateCheck:

    def test_within_deviation_accepted(self):
        ok, reason = universe_gate_check(2100, 2000.0, max_deviation=0.15)
        assert ok and reason == ""

    def test_degenerate_small_build_refused(self):
        # The 06-04 incident: 457 names against a ~2,000-name baseline
        ok, reason = universe_gate_check(457, 2000.0, max_deviation=0.15)
        assert not ok
        assert "-77%" in reason

    def test_oversized_build_refused(self):
        ok, _ = universe_gate_check(2800, 2000.0, max_deviation=0.15)
        assert not ok

    def test_boundary_is_inclusive(self):
        ok, _ = universe_gate_check(2300, 2000.0, max_deviation=0.15)  # exactly +15%
        assert ok

    def test_deviation_configurable(self):
        ok, _ = universe_gate_check(2400, 2000.0, max_deviation=0.25)  # +20% < 25%
        assert ok


class TestBootstrapSeeding:
    """With no accepted history, only a full-universe build may seed the
    baseline — a degenerate first build must NOT be accepted (the 2026-06-14
    freeze: final_count=1 bootstrap-accepted → median 1 → everything refused)."""

    def test_sane_full_build_seeds_baseline(self):
        # Today's 2,093 build: no baseline yet, within the seeding range
        ok, reason = universe_gate_check(2093, None, max_deviation=0.15)
        assert ok and reason == ""

    def test_degenerate_build_refused_not_seeded(self):
        # The poison: a single-symbol build must never become the baseline
        ok, reason = universe_gate_check(1, None, max_deviation=0.15)
        assert not ok
        assert "bootstrap" in reason and "[2000, 2400]" in reason

    def test_truncated_build_not_seeded(self):
        # The 06-04 truncation (457 names) is below the seeding floor
        ok, reason = universe_gate_check(457, None, max_deviation=0.15)
        assert not ok and "bootstrap" in reason

    def test_oversized_build_not_seeded(self):
        ok, _ = universe_gate_check(5000, None, max_deviation=0.15)
        assert not ok

    def test_seeding_bounds_inclusive(self):
        assert universe_gate_check(2000, None)[0]    # lower bound
        assert universe_gate_check(2400, None)[0]    # upper bound
        assert not universe_gate_check(1999, None)[0]
        assert not universe_gate_check(2401, None)[0]

    def test_zero_baseline_treated_as_bootstrap(self):
        # A baseline that computed to 0 must not divide-by-zero; bootstrap path
        ok, _ = universe_gate_check(2093, 0.0, max_deviation=0.15)
        assert ok


# ══════════════════════════════════════════════════════════════════════════════
# most_recent_completed_session — never build from an in-flight session
# ══════════════════════════════════════════════════════════════════════════════

class TestCompletedSession:
    # 2026-06-11 = Thursday, 06-10 = Wednesday, 06-12 = Friday

    def test_premarket_resolves_to_previous_day(self):
        now = datetime(2026, 6, 11, 8, 30)   # 8:30 AM ET — the incident timing
        assert most_recent_completed_session(date(2026, 6, 11), now=now) == date(2026, 6, 10)

    def test_intraday_resolves_to_previous_day(self):
        now = datetime(2026, 6, 11, 13, 0)
        assert most_recent_completed_session(date(2026, 6, 11), now=now) == date(2026, 6, 10)

    def test_just_after_close_still_previous_day(self):
        # 17:00 ET is inside the publication buffer (final at 18:00)
        now = datetime(2026, 6, 11, 17, 0)
        assert most_recent_completed_session(date(2026, 6, 11), now=now) == date(2026, 6, 10)

    def test_late_evening_uses_same_day(self):
        now = datetime(2026, 6, 11, 19, 0)
        assert most_recent_completed_session(date(2026, 6, 11), now=now) == date(2026, 6, 11)

    def test_weekend_resolves_to_friday(self):
        now = datetime(2026, 6, 13, 12, 0)   # Saturday
        assert most_recent_completed_session(date(2026, 6, 13), now=now) == date(2026, 6, 12)

    def test_past_date_unchanged(self):
        now = datetime(2026, 6, 11, 8, 30)
        assert most_recent_completed_session(date(2026, 6, 9), now=now) == date(2026, 6, 9)


# ══════════════════════════════════════════════════════════════════════════════
# gate_baseline_median — only gate-accepted builds seed the baseline
# ══════════════════════════════════════════════════════════════════════════════

class TestGateBaseline:

    def test_median_over_accepted_only(self, tmp_path):
        db = tmp_path / "test.duckdb"
        _seed_history(db, [
            (date(2026, 6, 4), 457, None),     # pre-gate degenerate — excluded
            (date(2026, 6, 8), 2325, None),    # pre-gate — excluded
            (date(2026, 6, 15), 2000, True),
            (date(2026, 6, 16), 2100, True),
            (date(2026, 6, 17), 2200, True),
            (date(2026, 6, 18), 500, False),   # refused — excluded
        ])
        assert gate_baseline_median(db_path=db) == pytest.approx(2100.0)

    def test_no_accepted_rows_returns_none(self, tmp_path):
        db = tmp_path / "test.duckdb"
        _seed_history(db, [(date(2026, 6, 4), 457, None)])
        assert gate_baseline_median(db_path=db) is None

    def test_empty_db_returns_none(self, tmp_path):
        db = tmp_path / "test.duckdb"
        _seed_history(db, [])
        assert gate_baseline_median(db_path=db) is None

    def test_degenerate_accepted_rows_self_heal(self, tmp_path):
        """The 2026-06-14 poison: final_count=1 builds marked accepted.  The
        schema migration must correct them so the median is no longer 1 and the
        gate can recover (median over the remaining real accepted build)."""
        from quantlab.storage import _ensure_schema
        db = tmp_path / "test.duckdb"
        _seed_history(db, [
            (date(2026, 6, 4), 1, True),       # poison
            (date(2026, 6, 5), 1, True),       # poison
            (date(2026, 6, 16), 2100, True),   # genuine accepted build
        ])
        # Re-running the schema migration corrects the degenerate accepted rows
        con = duckdb.connect(str(db))
        _ensure_schema(con)
        con.close()
        # Median now reflects only the real build, not the poison
        assert gate_baseline_median(db_path=db) == pytest.approx(2100.0)

    def test_self_heal_to_none_rebootstraps(self, tmp_path):
        """When the poison rows were the ONLY accepted rows, the baseline
        becomes None and the gate falls back to bootstrap seeding."""
        from quantlab.storage import _ensure_schema
        db = tmp_path / "test.duckdb"
        _seed_history(db, [(date(2026, 6, 4), 1, True), (date(2026, 6, 5), 1, True)])
        con = duckdb.connect(str(db))
        _ensure_schema(con)
        con.close()
        assert gate_baseline_median(db_path=db) is None
        # …and 2,093 now seeds cleanly via the bootstrap path
        assert universe_gate_check(2093, gate_baseline_median(db_path=db))[0]


# ══════════════════════════════════════════════════════════════════════════════
# Build integration — completeness floor and gate refusal keep prior universe
# ══════════════════════════════════════════════════════════════════════════════

def _sym(i: int) -> str:
    """Clean 3-letter ticker (passes the symbol-quality filter)."""
    return chr(65 + i // 676) + chr(65 + (i // 26) % 26) + chr(65 + i % 26)


def _grouped(n_total: int, n_passing: int, d: date) -> dict:
    """Fake grouped-daily dict: n_passing symbols pass all gates, rest fail price."""
    out = {}
    for i in range(n_passing):
        out[_sym(i)] = Bar(as_of=d, open=20, high=21, low=19, close=20.0,
                           volume=1_000_000.0)          # dv = $20M — passes
    for i in range(n_passing, n_total):
        out[f"X{i:05d}"] = Bar(as_of=d, open=1, high=1, low=1, close=1.0,
                               volume=1_000.0)           # fails $10 price gate
    return out


class FakeProvider:
    def __init__(self, by_date: dict):
        self.by_date = by_date
        self.calls: list[date] = []

    def get_grouped_daily(self, d: date) -> dict:
        self.calls.append(d)
        return self.by_date.get(d, {})


@pytest.fixture
def patched_env(tmp_path, monkeypatch):
    """Isolate cache dir + DuckDB; no Polygon key so the CS filter is skipped."""
    import quantlab.storage as storage
    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    monkeypatch.setattr(storage, "DATA_PROCESSED", data_dir)
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "test.duckdb")
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    return tmp_path


class TestBuildSafeguards:
    # Build for Wednesday 2026-06-10 (in the past → no session-resolution shift)
    TRADE = date(2026, 6, 10)
    PREV  = date(2026, 6, 9)

    # Some generated 3-letter symbols collide with the hard-exclusion list
    # (real ETF tickers), so ~2,000 passing inputs yield ~1,700 survivors.
    # The baseline is seeded to match the post-filter count.
    BASELINE = 1700

    def _seed_prior(self, tmp_path, n=1700):
        """Accepted baseline + a confirmed-final prior-day cache."""
        from quantlab.universe import save_universe_cache
        _seed_history(tmp_path / "test.duckdb", [
            (date(2026, 6, 1) + timedelta(days=i), self.BASELINE, True) for i in range(5)
        ])
        syms = [_sym(i) for i in range(n)]
        save_universe_cache(self.PREV, syms, [2e7] * n)
        return syms

    def test_gate_refusal_keeps_prior_universe(self, patched_env):
        prior = self._seed_prior(patched_env)
        # Full-size raw response (passes the floor) but degenerate final count
        provider = FakeProvider({self.TRADE: _grouped(9000, 50, self.TRADE)})
        mgr = UniverseManager()
        syms, stats = mgr.build_tradeable_universe(
            self.TRADE, provider, ib=None, optionable_only=False,
        )
        # Prior day's universe returned, degenerate cache NOT written
        assert syms == prior
        from quantlab.universe import _universe_cache_path
        assert not _universe_cache_path(self.TRADE).exists()
        # Refusal recorded under the build date for report/health check
        con = duckdb.connect(str(patched_env / "test.duckdb"))
        row = con.execute(
            "SELECT gate_accepted, final_count FROM universe_history WHERE date = ?",
            [self.TRADE],
        ).fetchone()
        con.close()
        assert row is not None
        assert row[0] is False
        assert row[1] < 100   # degenerate count (filters trim the 50 slightly)

    def test_sane_build_accepted_and_cached(self, patched_env):
        self._seed_prior(patched_env)
        provider = FakeProvider({self.TRADE: _grouped(9000, 2000, self.TRADE)})
        mgr = UniverseManager()
        syms, stats = mgr.build_tradeable_universe(
            self.TRADE, provider, ib=None, optionable_only=False,
        )
        # Within ±15% of the 1,700 baseline despite exclusion-list attrition
        assert 1450 <= len(syms) <= 2000
        assert stats.gate_accepted is True
        from quantlab.universe import _universe_cache_path
        assert _universe_cache_path(self.TRADE).exists()

    def test_truncated_response_walks_back_to_prior_cache(self, patched_env):
        """The 06-04 variant: 457 tickers mid-publication — below the
        completeness floor, the build must not use it."""
        prior = self._seed_prior(patched_env)
        provider = FakeProvider({self.TRADE: _grouped(457, 457, self.TRADE)})
        mgr = UniverseManager()
        syms, _ = mgr.build_tradeable_universe(
            self.TRADE, provider, ib=None, optionable_only=False,
        )
        assert syms == prior
        from quantlab.universe import _universe_cache_path
        assert not _universe_cache_path(self.TRADE).exists()


# ══════════════════════════════════════════════════════════════════════════════
# Streak neutrality — degenerate-build days neither break nor extend streaks
# ══════════════════════════════════════════════════════════════════════════════

def _weekdays_back(n: int, end: date) -> list[date]:
    """The last n weekdays strictly before ``end``, oldest first."""
    days, cur = [], end - timedelta(days=1)
    while len(days) < n:
        if cur.weekday() < 5:
            days.append(cur)
        cur -= timedelta(days=1)
    return list(reversed(days))


class TestStreakNeutrality:

    def test_trading_days_elapsed_skips_neutral_dates(self):
        start, end = date(2026, 6, 1), date(2026, 6, 12)   # Mon → Fri, 9 weekdays
        assert _trading_days_elapsed(start, end) == 9
        skip = {date(2026, 6, 4), date(2026, 6, 10)}
        assert _trading_days_elapsed(start, end, skip_dates=skip) == 7

    def test_skip_weekend_dates_are_noop(self):
        start, end = date(2026, 6, 1), date(2026, 6, 12)
        skip = {date(2026, 6, 6), date(2026, 6, 7)}        # Sat/Sun
        assert _trading_days_elapsed(start, end, skip_dates=skip) == 9

    def test_remove_stale_neutralizes_degenerate_days(self, tmp_path):
        """A symbol inactive for 7 weekdays, 3 of which were gate-refused
        builds, has only 4 effective inactive days → must NOT be pruned."""
        from quantlab.watchlist import InstitutionalWatchlist
        db = tmp_path / "test.duckdb"
        iwl = InstitutionalWatchlist(db_path=db)

        weekdays = _weekdays_back(7, date.today())
        last_seen = weekdays[0] - timedelta(days=1)
        con = duckdb.connect(str(db))
        con.execute(
            "INSERT INTO institutional_watchlist "
            "(symbol, first_seen, last_seen, consecutive_days, stage, conviction_score) "
            "VALUES ('KEEP', ?, ?, 3, 2, 0.8)",
            [last_seen, last_seen],
        )
        for d in weekdays[:3]:   # 3 degenerate-build days inside the gap
            con.execute(
                "INSERT OR REPLACE INTO universe_history (date, final_count, gate_accepted) "
                "VALUES (?, 457, FALSE)",
                [d],
            )
        con.close()

        assert iwl.remove_stale(max_days_inactive=5) == 0
        assert any(e["symbol"] == "KEEP" for e in iwl.get_candidates())

    def test_remove_stale_prunes_without_degenerate_days(self, tmp_path):
        """Same 7-weekday gap with healthy builds → pruned as before."""
        from quantlab.watchlist import InstitutionalWatchlist
        db = tmp_path / "test.duckdb"
        iwl = InstitutionalWatchlist(db_path=db)

        weekdays = _weekdays_back(7, date.today())
        last_seen = weekdays[0] - timedelta(days=1)
        con = duckdb.connect(str(db))
        con.execute(
            "INSERT INTO institutional_watchlist "
            "(symbol, first_seen, last_seen, consecutive_days, stage, conviction_score) "
            "VALUES ('GONE', ?, ?, 3, 2, 0.8)",
            [last_seen, last_seen],
        )
        con.close()

        assert iwl.remove_stale(max_days_inactive=5) == 1
        assert iwl.get_candidates() == []


# ══════════════════════════════════════════════════════════════════════════════
# Surfacing — report header warning and health-check line
# ══════════════════════════════════════════════════════════════════════════════

class TestGateSurfacing:

    def test_report_warning_on_refused_build(self, tmp_path):
        gr = _load_script("generate_report")
        db = tmp_path / "test.duckdb"
        _seed_history(db, [(date(2026, 6, 10), 457, False)])
        w = gr._universe_gate_warning(date(2026, 6, 10), str(db))
        assert w is not None and "UNIVERSE GATE" in w and "457" in w

    def test_no_report_warning_on_accepted_build(self, tmp_path):
        gr = _load_script("generate_report")
        db = tmp_path / "test.duckdb"
        _seed_history(db, [(date(2026, 6, 10), 2000, True)])
        assert gr._universe_gate_warning(date(2026, 6, 10), str(db)) is None

    def test_health_check_flags_refused_build(self, tmp_path):
        cdr = _load_script("check_daily_runs")
        db = tmp_path / "test.duckdb"
        _seed_history(db, [(date(2026, 6, 10), 457, False)])
        assert cdr.check_universe_gate(date(2026, 6, 10), db) is True
        assert cdr.check_universe_gate(date(2026, 6, 11), db) is False
