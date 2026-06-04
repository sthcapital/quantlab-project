"""
Shared utilities — logging setup, config loading, date helpers.
"""

from __future__ import annotations

import logging
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
    },
    "news": {
        "provider_codes": "BRFG+BRFUPDN+DJNL",
        "days": 120,
        "limit": 100,
    },
}


def get_config(section: str | None = None) -> dict:
    """
    Return config values. Reads from DEFAULT_CONFIG for now.
    Will support config.toml overrides in Phase 3.
    """
    if section:
        return DEFAULT_CONFIG.get(section, {})
    return DEFAULT_CONFIG
