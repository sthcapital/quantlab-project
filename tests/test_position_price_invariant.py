"""
Tests for the position-price invariant: an open position is NEVER monitored
with a null current price (SNEX 2026-06-12 / KO incidents — a same-session
entry's cold IBKR delayed-data subscription returned NaN, the updater
silently skipped it, and the stop could never trigger).
"""

from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import duckdb
import pytest

_ROOT = Path(__file__).parent.parent


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _patch_db(tmp_path, monkeypatch):
    import quantlab.storage as storage
    import quantlab.watchlist as watchlist
    db = tmp_path / "test.duckdb"
    monkeypatch.setattr(storage, "DB_PATH", db)
    monkeypatch.setattr(watchlist, "DB_PATH", db)
    return db


def _scan_result(symbol="SNEX", entry=129.74, stop=119.40):
    from quantlab.execution import ScanResult
    r = ScanResult(
        symbol=symbol, scan_date=date.today().isoformat(),
        signal_type="breakout", signal=True,
        entry_close=entry, indicator_value=None, lookback=5,
        conviction_score=0.95, stage=2,
    )
    r.atr_stop = stop
    return r


class _BlindIB:
    """IBKR mock whose market data never warms up (the SNEX failure mode)."""
    def reqMarketDataType(self, _t): pass
    def qualifyContracts(self, _c): return []   # nothing resolvable


# ══════════════════════════════════════════════════════════════════════════════
# 1. Entries are born priced
# ══════════════════════════════════════════════════════════════════════════════

class TestEntrySeedsPrice:

    def test_new_entry_has_current_price_immediately(self, tmp_path, monkeypatch):
        db = _patch_db(tmp_path, monkeypatch)
        from quantlab.watchlist import add_to_watchlist
        assert add_to_watchlist(_scan_result()) is True
        con = duckdb.connect(str(db))
        row = con.execute(
            "SELECT current_price, days_on_watch, status FROM watchlist "
            "WHERE symbol = 'SNEX'"
        ).fetchone()
        con.close()
        assert row[0] == pytest.approx(129.74)   # seeded with entry close
        assert row[1] == 0
        assert row[2] == "watching"


# ══════════════════════════════════════════════════════════════════════════════
# 2. Updater never leaves a position blind — snapshot fallback
# ══════════════════════════════════════════════════════════════════════════════

class TestSnapshotFallback:

    def test_flatfile_fallback_when_ib_blind(self, tmp_path, monkeypatch):
        db = _patch_db(tmp_path, monkeypatch)
        import quantlab.watchlist as wl
        from quantlab.watchlist import add_to_watchlist, update_watchlist_prices
        add_to_watchlist(_scan_result())
        # IBKR yields nothing; flat-file close steps in
        monkeypatch.setattr(wl, "_latest_flatfile_close", lambda s: 128.50)
        updates = update_watchlist_prices(_BlindIB())
        assert len(updates) == 1
        assert updates[0]["current_price"] == pytest.approx(128.50)
        con = duckdb.connect(str(db))
        px = con.execute(
            "SELECT current_price FROM watchlist WHERE symbol='SNEX'"
        ).fetchone()[0]
        con.close()
        assert px == pytest.approx(128.50)       # row updated, not skipped

    def test_snex_stop_triggers_on_fallback_price(self, tmp_path, monkeypatch):
        """The 2026-06-12 verification: with a non-null price below the
        $119.40 stop, the position actually stops out — under the old code
        this cycle was silently skipped and the stop could never fire."""
        db = _patch_db(tmp_path, monkeypatch)
        import quantlab.watchlist as wl
        from quantlab.watchlist import add_to_watchlist, update_watchlist_prices
        add_to_watchlist(_scan_result("SNEX", entry=129.74, stop=119.40))
        monkeypatch.setattr(wl, "_latest_flatfile_close", lambda s: 119.00)
        updates = update_watchlist_prices(_BlindIB())
        assert updates[0]["status"] == "stopped_out"
        con = duckdb.connect(str(db))
        status = con.execute(
            "SELECT status FROM watchlist WHERE symbol='SNEX'"
        ).fetchone()[0]
        con.close()
        assert status == "stopped_out"

    def test_no_source_logs_blind_and_keeps_row(self, tmp_path, monkeypatch, caplog):
        _patch_db(tmp_path, monkeypatch)
        import logging
        import quantlab.watchlist as wl
        from quantlab.watchlist import add_to_watchlist, update_watchlist_prices
        add_to_watchlist(_scan_result())
        monkeypatch.setattr(wl, "_latest_flatfile_close", lambda s: None)
        with caplog.at_level(logging.ERROR):
            updates = update_watchlist_prices(_BlindIB())
        assert updates == []
        assert any("monitored blind" in r.message for r in caplog.records)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Health-check assertion
# ══════════════════════════════════════════════════════════════════════════════

class TestPositionPriceHealthCheck:

    def _seed_watchlist(self, db, rows):
        from quantlab.storage import _ensure_schema
        con = duckdb.connect(str(db))
        _ensure_schema(con)
        for sym, price, status in rows:
            con.execute(
                "INSERT INTO watchlist (watch_id, symbol, date_added, "
                "entry_price, status, current_price) VALUES (?, ?, ?, 100.0, ?, ?)",
                [f"{sym}_2026-06-12", sym, date(2026, 6, 12), status, price],
            )
        con.close()

    def test_null_priced_open_position_flagged(self, tmp_path):
        cdr = _load_script("check_daily_runs")
        db = tmp_path / "test.duckdb"
        self._seed_watchlist(db, [
            ("SNEX", None, "watching"),
            ("CVS", 101.0, "watching"),
            ("OLD", None, "stopped_out"),   # closed — exempt
        ])
        assert cdr.check_position_prices(db) == ["SNEX"]

    def test_all_priced_passes(self, tmp_path):
        cdr = _load_script("check_daily_runs")
        db = tmp_path / "test.duckdb"
        self._seed_watchlist(db, [("SNEX", 129.74, "watching")])
        assert cdr.check_position_prices(db) == []

    def test_health_check_exit_1_on_null_price(self, tmp_path):
        cdr = _load_script("check_daily_runs")
        check_date = date(2026, 6, 11)   # Thursday, NYSE open
        db = tmp_path / "test.duckdb"
        self._seed_watchlist(db, [("SNEX", None, "watching")])
        # options heartbeat satisfied so only the price invariant fails
        con = duckdb.connect(str(db))
        con.execute(
            "CREATE TABLE IF NOT EXISTS options_snapshots "
            "(symbol VARCHAR, snap_date DATE, spot_price DOUBLE)"
        )
        con.execute("INSERT INTO options_snapshots VALUES ('X', ?, 1.0)", [check_date])
        con.close()
        log = tmp_path / "scan.log"
        log.write_text(
            f"[{check_date} 08:45:01] QuantLab Morning Check\n"
            f"[{check_date} 13:00:08] monitor_options: checking 357 watchlist symbols ...\n"
            f"[{check_date} 16:30:00] Forward Return Tracker\n"
            f"[{check_date} 17:00:05] Starting universe scan\n"
            f"[{check_date} 17:10:00] tape=BULL\n"
        )
        assert cdr.check_and_report(log, check_date, quiet=True, db_path=db) == 1
