"""SEC EDGAR fundamentals — no API key required."""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
_HEADERS = {"User-Agent": "QuantLab Research quantlab@sthcapital.com"}

DEFAULT_METRICS = [
    "revenue", "net_income", "eps_diluted", "total_assets",
    "total_debt", "operating_cashflow", "capex", "shares_out",
]

# US-GAAP concept candidates tried in order (first match wins)
_GAAP_FIELDS: dict[str, list[str]] = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
    ],
    "net_income": ["NetIncomeLoss"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "total_assets": ["Assets"],
    "total_debt": [
        "LongTermDebt",
        "DebtAndCapitalLeaseObligations",
        "LongTermDebtAndCapitalLeaseObligations",
    ],
    "operating_cashflow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "shares_out": ["CommonStockSharesOutstanding"],
}


@dataclass
class FundamentalSnapshot:
    ticker: str
    cik: str
    as_of: date
    # Latest values (most recent available quarter)
    revenue: Optional[float] = None
    net_income: Optional[float] = None
    eps_diluted: Optional[float] = None
    total_assets: Optional[float] = None
    total_debt: Optional[float] = None
    operating_cashflow: Optional[float] = None
    capex: Optional[float] = None
    shares_out: Optional[float] = None
    # QoQ growth rates (latest quarter vs prior quarter)
    revenue_qoq_growth: Optional[float] = None
    net_income_qoq_growth: Optional[float] = None
    eps_qoq_growth: Optional[float] = None
    # Sequential history for earnings acceleration (oldest first)
    eps_history: list[float] = field(default_factory=list)
    net_income_history: list[float] = field(default_factory=list)


# ── CIK lookup — cached for process lifetime ──────────────────────────────────

@functools.lru_cache(maxsize=1)
def _get_company_tickers() -> dict:
    """Download the SEC company tickers JSON once per process."""
    resp = requests.get(_TICKERS_URL, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def lookup_cik(ticker: str) -> str:
    """
    Resolve ticker to CIK string (zero-padded to 10 digits).

    The company_tickers.json is downloaded once and cached in memory for the
    process lifetime — subsequent calls for different symbols pay no network cost.

    Raises:
        ValueError: If ticker not found in the SEC filing index.
        requests.HTTPError: On network failure.
    """
    ticker_upper = ticker.upper()
    for entry in _get_company_tickers().values():
        if entry.get("ticker", "").upper() == ticker_upper:
            return str(entry["cik_str"]).zfill(10)
    raise ValueError(f"Ticker not found in SEC filing index: {ticker}")


# ── DuckDB cache helpers ───────────────────────────────────────────────────────

def _ensure_edgar_table(con) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS edgar_fundamentals (
            symbol VARCHAR,
            fetch_date DATE,
            acceleration_score DOUBLE,
            revenue_growth DOUBLE,
            eps_growth DOUBLE,
            consecutive_beats INTEGER,
            PRIMARY KEY (symbol, fetch_date)
        )
    """)


def _load_edgar_cache(symbol: str, max_age_days: int) -> Optional[float]:
    """Return cached acceleration_score if within max_age_days, else None."""
    try:
        import duckdb
        from quantlab.storage import DB_PATH

        cutoff = (date.today() - timedelta(days=max_age_days)).isoformat()
        con = duckdb.connect(str(DB_PATH))
        _ensure_edgar_table(con)
        row = con.execute(
            """
            SELECT acceleration_score FROM edgar_fundamentals
            WHERE symbol = ? AND fetch_date >= ?
            ORDER BY fetch_date DESC LIMIT 1
            """,
            [symbol, cutoff],
        ).fetchone()
        con.close()
        return float(row[0]) if row is not None else None
    except Exception as exc:
        logger.debug("EDGAR cache lookup failed for %s: %s", symbol, exc)
        return None


def _save_edgar_cache(
    symbol: str,
    snap: FundamentalSnapshot,
    acceleration_score: float,
    consecutive_beats: int,
) -> None:
    """Write EDGAR-derived metrics to the edgar_fundamentals cache table. Non-fatal."""
    try:
        import duckdb
        from quantlab.storage import DB_PATH

        con = duckdb.connect(str(DB_PATH))
        _ensure_edgar_table(con)
        con.execute(
            """
            INSERT OR REPLACE INTO edgar_fundamentals
                (symbol, fetch_date, acceleration_score, revenue_growth,
                 eps_growth, consecutive_beats)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                symbol,
                date.today().isoformat(),
                acceleration_score,
                snap.revenue_qoq_growth,
                snap.eps_qoq_growth,
                consecutive_beats,
            ],
        )
        con.close()
    except Exception as exc:
        logger.warning("edgar_fundamentals cache write failed for %s: %s", symbol, exc)


# ── Fundamental data fetching ─────────────────────────────────────────────────

def _extract_periods(facts: dict, metric: str, periods: int) -> list[float]:
    """Extract up to `periods` quarterly values for a metric. Returns oldest-first."""
    gaap_fields = _GAAP_FIELDS.get(metric, [])
    us_gaap = facts.get("us-gaap", {})

    for field_name in gaap_fields:
        if field_name not in us_gaap:
            continue
        units = us_gaap[field_name].get("units", {})
        unit_data = units.get("USD") or units.get("shares") or units.get("pure")
        if not unit_data:
            continue

        quarterly = [
            obs for obs in unit_data
            if obs.get("form") in ("10-Q", "10-K") and obs.get("end")
        ]
        if not quarterly:
            continue

        # Deduplicate by period end (keep latest filing per end date)
        seen: dict[str, float] = {}
        for obs in sorted(quarterly, key=lambda x: (x["end"], x.get("filed", ""))):
            seen[obs["end"]] = obs["val"]

        values = [seen[k] for k in sorted(seen.keys())]
        return values[-periods:]

    return []


def _qoq_growth(history: list[float]) -> Optional[float]:
    if len(history) < 2:
        return None
    prev, curr = history[-2], history[-1]
    if abs(prev) < 1e-9:
        return None
    return round((curr - prev) / abs(prev), 6)


def _count_consecutive_beats(history: list[float]) -> int:
    """Count consecutive quarters of positive QoQ growth from the most recent period."""
    count = 0
    for i in range(len(history) - 1, 0, -1):
        if abs(history[i - 1]) < 1e-9:
            break
        if history[i] > history[i - 1]:
            count += 1
        else:
            break
    return count


def fetch_fundamentals(
    ticker: str,
    metrics: Optional[list[str]] = None,
    periods: int = 12,
) -> FundamentalSnapshot:
    """
    Fetch quarterly fundamentals from the SEC EDGAR companyfacts API.

    Args:
        ticker:  Ticker symbol (e.g. "AAPL").
        metrics: Subset of DEFAULT_METRICS to fetch (None = all).
        periods: Number of quarters to retrieve (default 12).

    Returns:
        FundamentalSnapshot with latest values and QoQ growth rates.
    """
    if metrics is None:
        metrics = DEFAULT_METRICS

    cik = lookup_cik(ticker)
    url = _FACTS_URL.format(cik=cik)
    resp = requests.get(url, headers=_HEADERS, timeout=60)
    resp.raise_for_status()

    facts = resp.json().get("facts", {})
    snap = FundamentalSnapshot(ticker=ticker, cik=cik, as_of=date.today())

    if "revenue" in metrics:
        h = _extract_periods(facts, "revenue", periods)
        if h:
            snap.revenue = h[-1]
            snap.revenue_qoq_growth = _qoq_growth(h)

    if "net_income" in metrics:
        h = _extract_periods(facts, "net_income", periods)
        if h:
            snap.net_income = h[-1]
            snap.net_income_history = h
            snap.net_income_qoq_growth = _qoq_growth(h)

    if "eps_diluted" in metrics:
        h = _extract_periods(facts, "eps_diluted", periods)
        if h:
            snap.eps_diluted = h[-1]
            snap.eps_history = h
            snap.eps_qoq_growth = _qoq_growth(h)

    if "total_assets" in metrics:
        h = _extract_periods(facts, "total_assets", periods)
        if h:
            snap.total_assets = h[-1]

    if "total_debt" in metrics:
        h = _extract_periods(facts, "total_debt", periods)
        if h:
            snap.total_debt = h[-1]

    if "operating_cashflow" in metrics:
        h = _extract_periods(facts, "operating_cashflow", periods)
        if h:
            snap.operating_cashflow = h[-1]

    if "capex" in metrics:
        h = _extract_periods(facts, "capex", periods)
        if h:
            snap.capex = h[-1]

    if "shares_out" in metrics:
        h = _extract_periods(facts, "shares_out", periods)
        if h:
            snap.shares_out = h[-1]

    return snap


# ── High-level scanner integration ────────────────────────────────────────────

def get_edgar_acceleration(symbol: str, max_age_days: int = 7) -> Optional[float]:
    """
    Return EDGAR-based earnings acceleration score for `symbol`.

    Checks the DuckDB edgar_fundamentals cache first. Re-fetches from EDGAR
    only when the cached entry is older than `max_age_days` (default 7 days,
    matching quarterly earnings cadence).

    On any failure (network error, ticker not in SEC index, DuckDB unavailable),
    returns None so the caller can fall back to OHLCV inference.

    Returns:
        float 0–1 when data is available, None on any failure.
    """
    cached = _load_edgar_cache(symbol, max_age_days)
    if cached is not None:
        logger.debug("%s: EDGAR acceleration from cache: %.4f", symbol, cached)
        return cached

    try:
        snap = fetch_fundamentals(symbol, metrics=["eps_diluted", "net_income", "revenue"])
        score = compute_earnings_acceleration(snap)
        consecutive = _count_consecutive_beats(snap.eps_history or snap.net_income_history)
        _save_edgar_cache(symbol, snap, score, consecutive)
        logger.debug(
            "%s: EDGAR acceleration fetched: %.4f  consecutive_beats=%d",
            symbol, score, consecutive,
        )
        return score
    except Exception as exc:
        logger.debug("EDGAR acceleration unavailable for %s: %s", symbol, exc)
        return None


# ── Scoring ───────────────────────────────────────────────────────────────────

def compute_earnings_acceleration(snap: FundamentalSnapshot) -> float:
    """
    Score 0-1 reflecting EPS (or net income) growth acceleration.

    Uses the most recent 3 periods of EPS history (falls back to NI history).
    Returns 0.5 (neutral) when fewer than 3 data points are available.

    Interpretation:
        > 0.5: growth rate is accelerating
        = 0.5: neutral / stable
        < 0.5: growth rate is decelerating

    Replaces the bar-based earnings_acceleration placeholder in conviction scoring
    when EDGAR fundamental data is available.
    """
    history = snap.eps_history if len(snap.eps_history) >= 3 else snap.net_income_history
    if len(history) < 3:
        return 0.5

    a, b, c = history[-3], history[-2], history[-1]
    if abs(a) < 1e-9 or abs(b) < 1e-9:
        return 0.5

    prior_growth = (b - a) / abs(a)
    recent_growth = (c - b) / abs(b)
    acceleration = recent_growth - prior_growth

    # Clip to [-2, 2] and normalize to [0, 1]
    clipped = max(-2.0, min(2.0, acceleration))
    return round((clipped + 2.0) / 4.0, 4)
