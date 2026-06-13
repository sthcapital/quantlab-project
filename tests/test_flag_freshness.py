"""
Tests for options flag-freshness tracking (first_flagged_date / flag_streak).

The unusual flag alone is memoryless: it cannot distinguish the FIRST day
flow appears (positioning starting while price still bases) from the Nth
consecutive flagged day (campaign confirmation early, crowding risk late).
A multi-day campaign also inflates the symbol's own 20-session baseline —
the frozen-baseline diagnostic makes that decay measurable without changing
scoring.
"""

from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pytest

from quantlab.signals.options_relative import (
    FLAG_EPISODE_LAPSE_SESSIONS,
    flag_freshness,
    frozen_vs_live_zscores,
)

_ROOT = Path(__file__).parent.parent

D = lambda day: date(2026, 6, day)   # June 2026 shorthand


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ══════════════════════════════════════════════════════════════════════════════
# 1. Streak math
# ══════════════════════════════════════════════════════════════════════════════

class TestStreakMath:

    def test_first_ever_flag_starts_episode(self):
        first, streak = flag_freshness(True, D(11), [])
        assert (first, streak) == (D(11), 1)

    def test_consecutive_flags_increment_and_preserve_first(self):
        history = [
            (D(9),  True,  D(9)),
            (D(10), True,  D(9)),
        ]
        first, streak = flag_freshness(True, D(11), history)
        assert (first, streak) == (D(9), 3)

    def test_not_flagged_returns_none_zero(self):
        history = [(D(10), True, D(10))]
        assert flag_freshness(False, D(11), history) == (None, 0)

    def test_gated_unflagged_session_resets_streak(self):
        history = [
            (D(8),  True,  D(8)),
            (D(9),  True,  D(8)),
            (D(10), False, None),   # gated, not flagged → streak broken
        ]
        first, streak = flag_freshness(True, D(11), history)
        assert streak == 1                    # streak resets on any unflagged session
        assert first == D(8)                  # 1-session gap < lapse → same episode

    def test_legacy_rows_without_first_date_fall_back_to_session_date(self):
        history = [(D(10), True, None)]       # pre-freshness row
        first, streak = flag_freshness(True, D(11), history)
        assert (first, streak) == (D(10), 2)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Episode lapse / restart
# ══════════════════════════════════════════════════════════════════════════════

class TestEpisodeLapse:

    def test_gap_below_lapse_continues_episode(self):
        history = [
            (D(5), True,  D(5)),
            (D(8), False, None),
            (D(9), False, None),   # 2 unflagged sessions < default lapse 3
        ]
        first, streak = flag_freshness(True, D(11), history)
        assert (first, streak) == (D(5), 1)

    def test_gap_at_lapse_starts_new_episode(self):
        history = [
            (D(4), True,  D(4)),
            (D(8), False, None),
            (D(9), False, None),
            (D(10), False, None),  # 3 unflagged sessions = default lapse
        ]
        first, streak = flag_freshness(True, D(11), history)
        assert (first, streak) == (D(11), 1)
        assert FLAG_EPISODE_LAPSE_SESSIONS == 3

    def test_lapse_configurable(self):
        history = [
            (D(8), True,  D(8)),
            (D(9), False, None),
            (D(10), False, None),
        ]
        first, _ = flag_freshness(True, D(11), history, lapse_sessions=2)
        assert first == D(11)      # 2-session gap ends the episode at lapse=2
        first, _ = flag_freshness(True, D(11), history, lapse_sessions=3)
        assert first == D(8)       # but continues it at lapse=3


# ══════════════════════════════════════════════════════════════════════════════
# 3. Neutral sessions — skip_dates and ungated days
# ══════════════════════════════════════════════════════════════════════════════

class TestNeutralSessions:

    def test_skip_dates_do_not_break_streak(self):
        """Gate-refused / degenerate-universe sessions are neutral — same
        convention as remove_stale's skip_dates."""
        history = [
            (D(9),  True,  D(9)),
            (D(10), False, None),     # degenerate-build day
        ]
        first, streak = flag_freshness(
            True, D(11), history, skip_dates={D(10)},
        )
        assert (first, streak) == (D(9), 2)   # streak continues across it

    def test_skip_dates_do_not_count_toward_lapse(self):
        history = [
            (D(5), True,  D(5)),
            (D(8), False, None),
            (D(9), False, None),
            (D(10), False, None),     # would be the 3rd lapse session...
        ]
        first, _ = flag_freshness(
            True, D(11), history, skip_dates={D(10)},   # ...but it's neutral
        )
        assert first == D(5)          # episode survives: effective gap = 2

    def test_ungated_sessions_are_neutral(self):
        """unusual_flag None (symbol not gated that day) carries no
        information — neither breaks the streak nor counts toward lapse."""
        history = [
            (D(9),  True, D(9)),
            (D(10), None, None),      # not scored that session
        ]
        first, streak = flag_freshness(True, D(11), history)
        assert (first, streak) == (D(9), 2)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Persistence — mark_unusual_flags round trip
# ══════════════════════════════════════════════════════════════════════════════

def _seed_day(db, day, scored, unscored=()):
    from quantlab.providers.massive_options import MassiveOptionsProvider
    con = duckdb.connect(str(db))
    MassiveOptionsProvider._ensure_table(con)
    for sym in scored:
        con.execute(
            "INSERT INTO options_snapshots "
            "(symbol, snap_date, spot_price, rel_score) VALUES (?, ?, 100.0, 0.5)",
            [sym, day],
        )
    for sym in unscored:
        con.execute(
            "INSERT INTO options_snapshots (symbol, snap_date, spot_price) "
            "VALUES (?, ?, 100.0)",
            [sym, day],
        )
    con.close()


def _freshness_rows(db, day):
    con = duckdb.connect(str(db))
    rows = {
        sym: (first, streak)
        for sym, first, streak in con.execute(
            "SELECT symbol, first_flagged_date, flag_streak "
            "FROM options_snapshots WHERE snap_date = ?",
            [day],
        ).fetchall()
    }
    con.close()
    return rows


class TestFreshnessPersistence:

    def test_multi_day_streak_and_reset(self, tmp_path, monkeypatch):
        import quantlab.storage as storage
        from quantlab.providers.massive_options import MassiveOptionsProvider
        db = tmp_path / "test.duckdb"
        monkeypatch.setattr(storage, "DB_PATH", db)
        mp = MassiveOptionsProvider(api_key="test")

        # Day 1: ASH flagged (fresh), CB gated-unflagged, NOSCORE ungated
        _seed_day(db, D(9), ["ASH", "CB"], unscored=["NOSCORE"])
        mp.mark_unusual_flags({"ASH"}, snap_date=D(9))
        rows = _freshness_rows(db, D(9))
        assert rows["ASH"] == (D(9), 1)
        assert rows["CB"] == (None, 0)        # gated, not flagged — measured 0
        assert rows["NOSCORE"] == (None, None)  # ungated stays NULL

        # Day 2: ASH flagged again → streak 2, episode start preserved
        _seed_day(db, D(10), ["ASH", "CB"])
        mp.mark_unusual_flags({"ASH"}, snap_date=D(10))
        assert _freshness_rows(db, D(10))["ASH"] == (D(9), 2)

        # Day 3: ASH not flagged → reset
        _seed_day(db, D(11), ["ASH"])
        mp.mark_unusual_flags(set(), snap_date=D(11))
        assert _freshness_rows(db, D(11))["ASH"] == (None, 0)

        # Day 4: re-flag within lapse → streak restarts at 1, episode continues
        _seed_day(db, D(12), ["ASH"])
        mp.mark_unusual_flags({"ASH"}, snap_date=D(12))
        assert _freshness_rows(db, D(12))["ASH"] == (D(9), 1)

    def test_episode_restart_after_lapse(self, tmp_path, monkeypatch):
        import quantlab.storage as storage
        from quantlab.providers.massive_options import MassiveOptionsProvider
        db = tmp_path / "test.duckdb"
        monkeypatch.setattr(storage, "DB_PATH", db)
        mp = MassiveOptionsProvider(api_key="test")

        _seed_day(db, D(1), ["ASH"])
        mp.mark_unusual_flags({"ASH"}, snap_date=D(1))
        for day in (D(2), D(3), D(4)):        # 3 gated-unflagged sessions = lapse
            _seed_day(db, day, ["ASH"])
            mp.mark_unusual_flags(set(), snap_date=day)
        _seed_day(db, D(5), ["ASH"])
        mp.mark_unusual_flags({"ASH"}, snap_date=D(5))
        assert _freshness_rows(db, D(5))["ASH"] == (D(5), 1)   # NEW episode

    def test_degenerate_universe_day_is_neutral(self, tmp_path, monkeypatch):
        import quantlab.storage as storage
        from quantlab.providers.massive_options import MassiveOptionsProvider
        from quantlab.storage import _ensure_schema
        db = tmp_path / "test.duckdb"
        monkeypatch.setattr(storage, "DB_PATH", db)
        mp = MassiveOptionsProvider(api_key="test")

        _seed_day(db, D(9), ["ASH"])
        mp.mark_unusual_flags({"ASH"}, snap_date=D(9))

        # Day 2 was a gate-refused build: snapshot row exists but unflagged,
        # and universe_history marks the date degenerate → neutral
        _seed_day(db, D(10), ["ASH"])
        mp.mark_unusual_flags(set(), snap_date=D(10))
        con = duckdb.connect(str(db))
        _ensure_schema(con)
        con.execute(
            "INSERT OR REPLACE INTO universe_history (date, final_count, gate_accepted) "
            "VALUES (?, 457, FALSE)",
            [D(10)],
        )
        con.close()

        _seed_day(db, D(11), ["ASH"])
        mp.mark_unusual_flags({"ASH"}, snap_date=D(11))
        # Streak survives the degenerate day: 2 consecutive flagged sessions
        assert _freshness_rows(db, D(11))["ASH"] == (D(9), 2)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Rendering — fresh vs streak, no glyph for blocked names
# ══════════════════════════════════════════════════════════════════════════════

class TestFreshnessRendering:

    # Marker is ASCII "F"/"FdN" — the report's Helvetica Type1 font has no
    # U+2691 (⚑) glyph, which rendered as "n" in the generated PDF.

    def test_fresh_flag_renders_marker(self):
        gr = _load_script("generate_report")
        assert gr._opts_cell(True, {"unusual_flag": True, "flag_streak": 1}) == "F"

    def test_streak_renders_day_count(self):
        gr = _load_script("generate_report")
        assert gr._opts_cell(True, {"unusual_flag": True, "flag_streak": 3}) == "Fd3"

    def test_marker_is_winansi_safe(self):
        """Every char of the marker must exist in the Type1 Helvetica
        encoding ReportLab uses — non-encodable chars render as wrong
        glyphs in the PDF (the original ⚑→'n' bug)."""
        gr = _load_script("generate_report")
        for streak in (1, 2, 12):
            cell = gr._opts_cell(True, {"unusual_flag": True,
                                        "flag_streak": streak})
            assert all(ord(ch) < 128 for ch in cell), cell

    def test_blocked_names_get_no_marker(self):
        """Floor-blocked / put-dominated rows are gated but unflagged — they
        keep the plain dash, never a flag marker."""
        gr = _load_script("generate_report")
        cell = gr._opts_cell(False, {"unusual_flag": False, "flag_streak": 0,
                                     "put_dominated": True})
        assert cell == "–"

    def test_no_snapshot_data_falls_back_to_legacy_tick(self):
        gr = _load_script("generate_report")
        assert gr._opts_cell(True, None) == "✓"
        assert gr._opts_cell(False, None) == "–"


# ══════════════════════════════════════════════════════════════════════════════
# 6. Baseline-inflation diagnostic (never changes scoring)
# ══════════════════════════════════════════════════════════════════════════════

class TestFrozenBaselineDiagnostic:

    def test_inflated_live_baseline_depresses_z(self):
        """A 5-day campaign at elevated volume inflates the trailing baseline;
        the frozen-at-episode-start baseline shows what z would have been."""
        quiet = [1000.0] * 20                       # baseline at episode start
        # Live baseline: campaign days (5 × 5000) displaced the oldest five
        live = [1000.0] * 15 + [5000.0] * 5
        z_live, z_frozen = frozen_vs_live_zscores(5000.0, live, quiet)
        assert z_live is not None and z_frozen is not None
        assert z_frozen > z_live                    # decay is measurable
        assert z_frozen == pytest.approx(10.0)      # capped — hugely unusual vs quiet base

    def test_streak_count_in_rate_header(self, tmp_path):
        gr = _load_script("generate_report")
        from quantlab.providers.massive_options import MassiveOptionsProvider
        db = tmp_path / "test.duckdb"
        con = duckdb.connect(str(db))
        MassiveOptionsProvider._ensure_table(con)
        for i, streak in enumerate([6, 5, 1, 0]):
            con.execute(
                "INSERT INTO options_snapshots "
                "(symbol, snap_date, spot_price, rel_score, unusual_flag, flag_streak) "
                "VALUES (?, ?, 100.0, 0.8, ?, ?)",
                [f"S{i}", D(11), streak > 0, streak],
            )
        con.close()
        line = gr._options_signal_rate(D(11), str(db))
        assert "2 streak≥5 (baseline-inflation watch)" in line

    def test_no_streak_note_when_all_fresh(self, tmp_path):
        gr = _load_script("generate_report")
        from quantlab.providers.massive_options import MassiveOptionsProvider
        db = tmp_path / "test.duckdb"
        con = duckdb.connect(str(db))
        MassiveOptionsProvider._ensure_table(con)
        for i in range(12):
            con.execute(
                "INSERT INTO options_snapshots "
                "(symbol, snap_date, spot_price, rel_score, unusual_flag, flag_streak) "
                "VALUES (?, ?, 100.0, 0.8, ?, ?)",
                [f"S{i}", D(11), i == 0, 1 if i == 0 else 0],
            )
        con.close()
        line = gr._options_signal_rate(D(11), str(db))
        assert "streak≥5" not in line
