"""
Regime exposure policy — maps the 5-state tape classification to explicit
position-initiation rules.

Scanning and watchlist accumulation run in ALL regimes (the institutional
watchlist keeps building consecutive-days counters through corrections —
O'Neil/Minervini: corrections are for building watchlists).  This policy
governs only what happens at ENTRY time:

    BULL       — normal entries, full size.
    RECOVERY   — entries allowed at half size; each entry additionally needs a
                 confirming signal (options OR breakout volume ≥ 2× average OR
                 volume dry-up).
    NEUTRAL    — entries allowed at half size, top-3 instead of top-5.
    CORRECTION — NO new entries.  Scanning continues, open positions are
                 managed normally (stops unchanged).
    BEAR       — no new long entries; stops on open positions tightened by
                 stop_tighten_factor.
    UNKNOWN    — FAIL CLOSED: no new entries.  An unreadable/missing tape
                 state is an infrastructure failure, not a market opinion —
                 2026-06-12: a swallowed DuckDB lock error during concurrent
                 scans read tape=None, the old NEUTRAL fallback entered SNEX
                 at half size in a CORRECTION tape.

Every gate decision is persisted to the regime_gate_log DuckDB table so the
daily report can render it ("3 candidates qualified, 0 entered — regime
CORRECTION") — suppressed entries are visible, never silent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

logger = logging.getLogger("quantlab.risk.regime_policy")


@dataclass(frozen=True)
class RegimeRule:
    """Entry/exposure rules for one tape state."""
    allow_entries: bool        # False → no new positions in this regime
    size_factor: float         # 1.0 = full size, 0.5 = half size, 0.0 = none
    max_new_positions: int     # cap on entries per scan day
    require_confirming: bool   # entry needs options / 2× breakout vol / VDU
    stop_tighten_factor: float # 1.0 = stops unchanged; 0.5 = halve the
                               # entry→stop distance on OPEN positions


DEFAULT_REGIME_POLICY: dict[str, RegimeRule] = {
    "BULL":       RegimeRule(allow_entries=True,  size_factor=1.0,
                             max_new_positions=5, require_confirming=False,
                             stop_tighten_factor=1.0),
    "RECOVERY":   RegimeRule(allow_entries=True,  size_factor=0.5,
                             max_new_positions=5, require_confirming=True,
                             stop_tighten_factor=1.0),
    "NEUTRAL":    RegimeRule(allow_entries=True,  size_factor=0.5,
                             max_new_positions=3, require_confirming=False,
                             stop_tighten_factor=1.0),
    "CORRECTION": RegimeRule(allow_entries=False, size_factor=0.0,
                             max_new_positions=0, require_confirming=False,
                             stop_tighten_factor=1.0),
    "BEAR":       RegimeRule(allow_entries=False, size_factor=0.0,
                             max_new_positions=0, require_confirming=False,
                             stop_tighten_factor=0.5),
    # FAIL CLOSED: unknown regime = no new entries.  Scanning and watchlist
    # accumulation continue; only position initiation is blocked.
    "UNKNOWN":    RegimeRule(allow_entries=False, size_factor=0.0,
                             max_new_positions=0, require_confirming=False,
                             stop_tighten_factor=1.0),
}


def get_regime_rule(
    tape: str | None,
    policy: dict[str, RegimeRule] | None = None,
) -> RegimeRule:
    """Return the RegimeRule for a tape state.

    Unknown or missing tape FAILS CLOSED (no new entries) and logs loudly —
    a missing tape state means the classifier's record was unreadable, which
    is an infrastructure fault, not information about the market.  The old
    NEUTRAL fallback entered positions on exactly such a fault (2026-06-12:
    SNEX entered at half size while the persisted tape said CORRECTION).
    """
    p = policy or DEFAULT_REGIME_POLICY
    key = (tape or "").upper()
    if key not in p:
        logger.warning(
            "REGIME GATE FAIL-CLOSED: tape state %r is unknown — "
            "no new entries this session", tape,
        )
        return p["UNKNOWN"]
    return p[key]


def load_session_tape(
    session_date: date | None = None,
    db_path: str | None = None,
    max_back_sessions: int = 1,
    retries: int = 3,
    retry_wait: float = 2.0,
) -> str | None:
    """
    Tape state for the session from the PERSISTED breadth_history record —
    the same record the daily report renders, so gate and report can never
    disagree about the regime.

    Hard dependency, not timing luck: transient DuckDB errors (concurrent
    scans — the 2026-06-12 failure mode) are retried, not swallowed into a
    silent None.  Walks back at most ``max_back_sessions`` trading days
    (covers the Sunday refresh reading Friday's row).  Returns None only
    when no tape row exists for the session — callers must fail closed.
    """
    import time

    from quantlab.market_calendar import prev_trading_day

    d = session_date or date.today()
    candidates = [d]
    for _ in range(max_back_sessions):
        candidates.append(prev_trading_day(candidates[-1]))

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            import duckdb
            from quantlab.storage import DB_PATH
            con = duckdb.connect(str(db_path or DB_PATH), read_only=True)
            placeholders = ", ".join("?" for _ in candidates)
            row = con.execute(
                f"SELECT tape FROM breadth_history "
                f"WHERE date IN ({placeholders}) ORDER BY date DESC LIMIT 1",
                candidates,
            ).fetchone()
            con.close()
            return row[0] if row and row[0] else None
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "load_session_tape attempt %d/%d failed (%s) — retrying",
                attempt, retries, exc,
            )
            time.sleep(retry_wait)
    logger.error(
        "load_session_tape: all %d attempts failed (%s) — gate must fail closed",
        retries, last_exc,
    )
    return None


def has_confirming_signal(
    scan_result,
    iwl_entry: dict | None = None,
    options_gating_enabled: bool | None = None,
) -> bool:
    """RECOVERY-entry confirmation: options activity OR breakout volume ≥ 2×
    average (breakout_volume_score ≥ 0.7 — Weinstein's 2× rule) OR volume
    dry-up then expansion (IWL volume_dry_up flag).

    Options activity counts only when options_signal_gating_enabled is on
    (None → scanner config) — the same display-only contract as
    select_top_candidates.  The entry path must never grant credit the
    qualification path withholds.
    """
    iwl_entry = iwl_entry or {}
    if options_gating_enabled is None:
        try:
            from quantlab.utils import get_config
            options_gating_enabled = bool(
                get_config("scanner").get("options_signal_gating_enabled", False)
            )
        except Exception:
            options_gating_enabled = False
    _opt_score = getattr(scan_result, "options_score", None)
    options = options_gating_enabled and (
        getattr(scan_result, "unusual_options_score", 0.0) >= 0.5
        or (_opt_score is not None and _opt_score >= 0.6)
        or bool(iwl_entry.get("options_signal", False))
    )
    breakout_vol = getattr(scan_result, "breakout_volume_score", 0.0) >= 0.7
    vdu = bool(iwl_entry.get("volume_dry_up", False))
    return options or breakout_vol or vdu


def apply_regime_gate(
    ranked_items: list,
    rule: RegimeRule,
    iwl_state: dict | None = None,
) -> tuple[list, list[str]]:
    """Apply the regime rule to ranked candidate tuples (ScanResult, earn, cdays).

    Returns (entry_items, suppressed_symbols).  Suppressed symbols stay on the
    institutional watchlist accumulating consecutive-days — they are withheld
    from position initiation only.
    """
    if not rule.allow_entries:
        return [], [it[0].symbol for it in ranked_items]
    entries = list(ranked_items)
    if rule.require_confirming:
        entries = [
            it for it in entries
            if has_confirming_signal(it[0], (iwl_state or {}).get(it[0].symbol))
        ]
    entries = entries[: rule.max_new_positions]
    chosen = {it[0].symbol for it in entries}
    suppressed = [it[0].symbol for it in ranked_items if it[0].symbol not in chosen]
    return entries, suppressed


def effective_stop_price(
    entry_price: float | None,
    atr_stop: float | None,
    factor: float,
) -> float | None:
    """Stop level after regime tightening.

    factor scales the entry→stop distance: 1.0 leaves the stop unchanged,
    0.5 (BEAR) moves it halfway up toward entry.  Returns the original stop
    when inputs are unusable or no tightening applies.
    """
    if not entry_price or not atr_stop or factor >= 1.0 or entry_price <= atr_stop:
        return atr_stop
    return entry_price - factor * (entry_price - atr_stop)


@dataclass
class RegimeGateDecision:
    """Outcome of applying the regime rule to one scan day's candidates."""
    gate_date: str
    tape: str
    qualified: int                       # candidates passing the strict filter
    entered: int                         # actually added as positions
    size_factor: float
    suppressed: list[str] = field(default_factory=list)  # symbols held back

    def summary(self) -> str:
        s = (f"{self.qualified} candidate(s) qualified, "
             f"{self.entered} entered — regime {self.tape}")
        if self.size_factor not in (0.0, 1.0) and self.entered > 0:
            s += f" (size ×{self.size_factor:g})"
        if self.suppressed:
            s += f" | held back: {', '.join(self.suppressed)}"
        return s


def log_regime_gate(decision: RegimeGateDecision, db_path: str | None = None) -> None:
    """Persist the gate decision so generate_report.py can render it. Non-fatal."""
    try:
        import duckdb
        from quantlab.storage import DB_PATH, _ensure_schema
        con = duckdb.connect(db_path or str(DB_PATH))
        _ensure_schema(con)
        con.execute(
            """
            INSERT OR REPLACE INTO regime_gate_log
                (date, tape, qualified, entered, size_factor, suppressed_symbols)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                decision.gate_date, decision.tape, decision.qualified,
                decision.entered, decision.size_factor,
                ",".join(decision.suppressed),
            ],
        )
        con.close()
    except Exception as exc:
        logger.warning("regime_gate_log write failed: %s", exc)


def load_regime_gate(
    gate_date: date | str | None = None,
    db_path: str | None = None,
) -> RegimeGateDecision | None:
    """Load the gate decision for a date (default today). None when absent."""
    d = gate_date or date.today()
    d_str = d.isoformat() if hasattr(d, "isoformat") else str(d)
    try:
        import duckdb
        from quantlab.storage import DB_PATH, _ensure_schema
        con = duckdb.connect(db_path or str(DB_PATH))
        _ensure_schema(con)
        row = con.execute(
            "SELECT date, tape, qualified, entered, size_factor, suppressed_symbols "
            "FROM regime_gate_log WHERE CAST(date AS VARCHAR) = ?",
            [d_str],
        ).fetchone()
        con.close()
        if row is None:
            return None
        return RegimeGateDecision(
            gate_date=str(row[0]), tape=row[1], qualified=row[2],
            entered=row[3], size_factor=row[4],
            suppressed=[s for s in (row[5] or "").split(",") if s],
        )
    except Exception as exc:
        logger.debug("regime_gate_log read failed: %s", exc)
        return None
