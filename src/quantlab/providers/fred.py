"""FRED macroeconomic data — requires FRED_API_KEY env var."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
_HEADERS = {"User-Agent": "QuantLab Research quantlab@sthcapital.com"}

FRED_SERIES: dict[str, str] = {
    "T10Y2Y":       "yield_spread_10y2y",   # 10Y minus 2Y Treasury spread
    "T10Y3M":       "yield_spread_10y3m",   # 10Y minus 3M Treasury spread
    "BAMLH0A0HYM2": "hy_credit_spread",     # High yield OAS spread
    "DGS10":        "treasury_10y",          # 10Y Treasury constant maturity
    "FEDFUNDS":     "fed_funds_rate",        # Effective federal funds rate
    "DCOILWTICO":   "wti_crude",             # WTI crude oil price
}


@dataclass
class MacroSnapshot:
    as_of: date
    yield_spread_10y2y: Optional[float] = None
    yield_spread_10y3m: Optional[float] = None
    hy_credit_spread: Optional[float] = None
    treasury_10y: Optional[float] = None
    fed_funds_rate: Optional[float] = None
    wti_crude: Optional[float] = None
    vix_close: Optional[float] = None   # populated from CBOE data by caller
    macro_regime: str = "risk_on"


def fetch_series(series_id: str, start: date, end: date, api_key: str) -> dict[date, float]:
    """
    Fetch a single FRED series and return {date: value} for dates in [start, end].

    Non-numeric values (e.g. '.' for missing observations) are silently dropped.
    """
    params = {
        "series_id": series_id,
        "observation_start": start.isoformat(),
        "observation_end": end.isoformat(),
        "api_key": api_key,
        "file_type": "json",
    }
    resp = requests.get(_FRED_BASE, params=params, headers=_HEADERS, timeout=30)
    resp.raise_for_status()

    result: dict[date, float] = {}
    for obs in resp.json().get("observations", []):
        try:
            dt = datetime.strptime(obs["date"], "%Y-%m-%d").date()
            result[dt] = float(obs["value"])
        except (KeyError, ValueError, TypeError):
            continue
    return result


def _latest_value(series: dict[date, float], as_of: date) -> Optional[float]:
    """Return the most recent value on or before as_of."""
    candidates = {d: v for d, v in series.items() if d <= as_of}
    if not candidates:
        return None
    return candidates[max(candidates)]


def fetch_macro_snapshot(api_key: str, as_of_date: Optional[date] = None) -> MacroSnapshot:
    """
    Fetch all FRED_SERIES and return a MacroSnapshot for the given date.

    Uses the most recent available observation on or before as_of_date.
    Falls back gracefully per series — missing fields remain None.
    Persists the snapshot to DuckDB macro_snapshots table.

    Args:
        api_key:      FRED API key.
        as_of_date:   Reference date (defaults to today).
    """
    as_of = as_of_date or date.today()
    lookback = as_of - timedelta(days=30)

    series_data: dict[str, dict[date, float]] = {}
    for series_id in FRED_SERIES:
        try:
            series_data[series_id] = fetch_series(series_id, lookback, as_of, api_key)
        except Exception as exc:
            logger.warning("FRED series %s failed: %s", series_id, exc)
            series_data[series_id] = {}

    snap = MacroSnapshot(
        as_of=as_of,
        yield_spread_10y2y=_latest_value(series_data.get("T10Y2Y", {}), as_of),
        yield_spread_10y3m=_latest_value(series_data.get("T10Y3M", {}), as_of),
        hy_credit_spread=_latest_value(series_data.get("BAMLH0A0HYM2", {}), as_of),
        treasury_10y=_latest_value(series_data.get("DGS10", {}), as_of),
        fed_funds_rate=_latest_value(series_data.get("FEDFUNDS", {}), as_of),
        wti_crude=_latest_value(series_data.get("DCOILWTICO", {}), as_of),
    )
    snap.macro_regime = classify_macro_regime(snap)

    _store_snapshot(snap)
    return snap


def classify_macro_regime(snapshot: MacroSnapshot) -> str:
    """
    Classify macro regime. Returns "stress" when ≥2 warnings are triggered:
        - Yield curve inverted: yield_spread_10y2y < 0
        - Credit stress: hy_credit_spread > 5.0
        - Volatility spike: vix_close > 25.0

    Returns: "stress" | "risk_off" | "risk_on"
    """
    warnings = 0
    if snapshot.yield_spread_10y2y is not None and snapshot.yield_spread_10y2y < 0:
        warnings += 1
    if snapshot.hy_credit_spread is not None and snapshot.hy_credit_spread > 5.0:
        warnings += 1
    if snapshot.vix_close is not None and snapshot.vix_close > 25.0:
        warnings += 1

    if warnings >= 2:
        return "stress"
    if warnings >= 1:
        return "risk_off"
    return "risk_on"


def _store_snapshot(snap: MacroSnapshot) -> None:
    """Persist snapshot to DuckDB macro_snapshots table. Non-fatal."""
    try:
        import duckdb
        from quantlab.storage import DB_PATH

        con = duckdb.connect(str(DB_PATH))
        con.execute("""
            CREATE TABLE IF NOT EXISTS macro_snapshots (
                as_of DATE PRIMARY KEY,
                yield_spread_10y2y DOUBLE,
                yield_spread_10y3m DOUBLE,
                hy_credit_spread DOUBLE,
                treasury_10y DOUBLE,
                fed_funds_rate DOUBLE,
                wti_crude DOUBLE,
                vix_close DOUBLE,
                macro_regime VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        con.execute(
            """
            INSERT OR REPLACE INTO macro_snapshots
                (as_of, yield_spread_10y2y, yield_spread_10y3m, hy_credit_spread,
                 treasury_10y, fed_funds_rate, wti_crude, vix_close, macro_regime)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                snap.as_of.isoformat(),
                snap.yield_spread_10y2y,
                snap.yield_spread_10y3m,
                snap.hy_credit_spread,
                snap.treasury_10y,
                snap.fed_funds_rate,
                snap.wti_crude,
                snap.vix_close,
                snap.macro_regime,
            ],
        )
        con.close()
    except Exception as exc:
        logger.warning("macro_snapshots storage failed: %s", exc)
