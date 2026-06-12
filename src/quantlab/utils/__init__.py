"""
Shared utilities — logging setup, config loading, date helpers.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with a clean, consistent format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ── Date helpers ──────────────────────────────────────────────────────────────

def today() -> date:
    return date.today()


def n_days_ago(n: int) -> date:
    return date.today() - timedelta(days=n)


def parse_date(value: str) -> date:
    """Parse YYYY-MM-DD string to date, with helpful error message."""
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise ValueError(f"Invalid date format '{value}'. Expected YYYY-MM-DD.")


def trading_days_between(start: date, end: date) -> int:
    """Rough estimate: 5/7 of calendar days."""
    calendar_days = (end - start).days
    return int(calendar_days * 5 / 7)


# ── Run ID ────────────────────────────────────────────────────────────────────

def make_run_id(symbol: str, signal_type: str, today_str: str | None = None) -> str:
    """Generate a unique run identifier for DuckDB records."""
    from datetime import datetime
    ts = today_str or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{symbol}_{signal_type}_{ts}"


# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "ibkr": {
        "host": "172.23.208.1",  # Windows host IP as seen from WSL2
        "port": 7497,
        "client_id": 1,          # historical data
        "spot_client_id": 51,              # get_spot_price — dedicated slot avoids collision
        "news_client_id": 41,              # news fetch in scanner — dedicated slot
        "options_chain_client_id": 21,     # reqSecDefOptParams / option chain scan
        "options_quotes_client_id": 22,    # reqTickers for option Greeks/IV
        "timeout": 10,
    },
    "backtest": {
        "lookback": 20,
        "initial_capital": 10_000.0,
        "cost_bps": 10.0,
        "min_trades": 30,
    },
    "scanner": {
        "universe": "small",
        "signal_type": "breakout",
        "min_conviction": 0.4,
        "min_rel_volume": 1.5,
        "news_lookback_days": 7,
        "SHORT_SIGNAL_ENABLED": False,   # activate after long side validated in paper trading (Phase 8+)
        # Recalibrated unusual-options signal (per-symbol baseline z-score +
        # cross-sectional top-decile gate; see signals/options_relative.py).
        # Display-only until the rescored 2026-06-11 output is reviewed:
        # options_signal renders in the report and persists to DuckDB, but it
        # satisfies the confirming-signal gate in select_top_candidates() and
        # adds the monitor's conviction bonus only when this is True.
        "options_signal_gating_enabled": False,
        # Cross-sectional gate percentile: "unusual" = the day's scores
        # strictly above this percentile (90 → ~top decile, ≤ ~10%/day).
        "options_unusual_percentile": 90.0,
        # Liquidity floor for gate eligibility: baseline average below this
        # many contracts cannot flag (still scored/displayed/persisted).
        # EG 2026-06-11: z=10 on a 24-contract baseline = one hedger rolling.
        "options_min_baseline_contracts": 75,
        # Universe build stability (2026-06-12 incident: builds against
        # partial-day Polygon grouped data produced 457–2,325 symbol swings).
        # Floor on raw grouped-ticker count — below this the response is
        # treated as mid-publication/truncated and the build walks back a day.
        "universe_min_grouped_rows": 8000,
        # Post-build sanity gate: refuse to replace the cache when final_count
        # deviates more than this fraction from the trailing 10-accepted-build
        # median.  A stale-but-sane universe beats a fresh-but-degenerate one.
        "universe_gate_max_deviation": 0.15,
    },
    "news": {
        "provider_codes": "BRFG+BRFUPDN+DJNL",
        "days": 120,
        "limit": 100,
    },
}


def _providers_config() -> dict:
    """
    Build the external data-provider config from environment variables.

    Evaluated at call time (not at import time) so tests can set env vars
    after importing without needing to reload the module.

    Environment variables:
        POLYGON_API_KEY    — Polygon.io REST API key
        FACTSET_USERNAME   — FactSet serial / username  (e.g. S123456@company)
        FACTSET_API_KEY    — FactSet API key
        FACTSET_HOST       — FactSet API base URL
                             (default: https://api.factset.com/content)
    """
    return {
        "polygon": {
            "api_key": os.environ.get("POLYGON_API_KEY", ""),
        },
        "factset": {
            "username": os.environ.get("FACTSET_USERNAME", ""),
            "api_key":  os.environ.get("FACTSET_API_KEY",  ""),
            "host":     os.environ.get(
                "FACTSET_HOST", "https://api.factset.com/content"
            ),
        },
    }


def get_config(section: str | None = None) -> dict:
    """
    Return config values merged from DEFAULT_CONFIG and live env vars.

    Provider credentials (Polygon, FactSet) are read from environment
    variables at call time so they are always current.

    Args:
        section: If given, return only that top-level section dict.
                 Supported sections: ibkr, backtest, scanner, news, providers.

    Returns:
        Full config dict, or the requested section dict (empty dict if absent).
    """
    cfg = dict(DEFAULT_CONFIG)
    cfg["providers"] = _providers_config()
    if section:
        return cfg.get(section, {})
    return cfg
