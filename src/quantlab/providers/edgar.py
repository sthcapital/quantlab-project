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
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
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
    "eps_diluted": [
        "EarningsPerShareDiluted",
        "EarningsPerShareBasicAndDiluted",          # used by many mid/small-caps
        "IncomeLossFromContinuingOperationsPerDilutedShare",
        "EarningsPerShareDilutedIncludingDiscontinuedOperations",
    ],
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
    # YoY same-quarter growth rates (most recent quarter vs same quarter prior year)
    # These eliminate seasonal bias present in QoQ comparisons.
    revenue_yoy_pct: Optional[float] = None           # e.g. 0.47 = +47%
    eps_yoy_pct: Optional[float] = None               # e.g. 0.88 = +88%
    revenue_yoy_history: list[float] = field(default_factory=list)  # last 4 quarters, oldest-first
    eps_yoy_history: list[float] = field(default_factory=list)      # last 4 quarters, oldest-first
    # True when BOTH revenue AND eps YoY growth rates are increasing for 2+ consecutive quarters
    is_accelerating: bool = False


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
            eps_diluted DOUBLE,
            PRIMARY KEY (symbol, fetch_date)
        )
    """)
    # Migration: add eps_diluted when table already existed without it
    try:
        cols = {r[1] for r in con.execute("PRAGMA table_info(edgar_fundamentals)").fetchall()}
        if "eps_diluted" not in cols:
            con.execute("ALTER TABLE edgar_fundamentals ADD COLUMN eps_diluted DOUBLE")
    except Exception:
        pass


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
        # Prefer YoY growth rates; fall back to QoQ; store 0.0 (not NULL) when
        # only one quarter of data is available so the scorer never fails silently.
        _rev_growth = (
            snap.revenue_yoy_pct if snap.revenue_yoy_pct is not None
            else (snap.revenue_qoq_growth if snap.revenue_qoq_growth is not None else 0.0)
        )
        _eps_growth = (
            snap.eps_yoy_pct if snap.eps_yoy_pct is not None
            else (snap.eps_qoq_growth if snap.eps_qoq_growth is not None else 0.0)
        )
        con.execute(
            """
            INSERT OR REPLACE INTO edgar_fundamentals
                (symbol, fetch_date, acceleration_score, revenue_growth,
                 eps_growth, consecutive_beats, eps_diluted)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                symbol,
                date.today().isoformat(),
                acceleration_score,
                _rev_growth,
                _eps_growth,
                consecutive_beats,
                snap.eps_diluted,
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
        # EPS fields use "USD/shares" (dollars per share); revenue/income use "USD".
        unit_data = (
            units.get("USD")
            or units.get("USD/shares")
            or units.get("shares")
            or units.get("pure")
        )
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


def _yoy_growth_series(history: list[float], max_quarters: int = 4) -> list[float]:
    """
    Compute year-over-year (same-quarter) growth rates from quarterly history.

    Requires at least 5 data points (current quarter + same quarter prior year).
    Skips quarters where the year-ago value is near zero (unreliable denominator).

    Args:
        history:      Quarterly values oldest-first (from _extract_periods).
        max_quarters: Maximum number of YoY rates to return.

    Returns:
        List of YoY growth rates, oldest-first. Empty if insufficient data.
    """
    if len(history) < 5:
        return []
    rates: list[float] = []
    for i in range(4, len(history)):
        prior = history[i - 4]
        curr = history[i]
        if abs(prior) < 1e-9:
            continue
        rates.append((curr - prior) / abs(prior))
    return rates[-max_quarters:]


def _is_yoy_accelerating(yoy_history: list[float]) -> bool:
    """True when the most recent YoY growth rate exceeds the prior quarter's rate."""
    if len(yoy_history) < 2:
        return False
    return yoy_history[-1] > yoy_history[-2]


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

    Fetches at least `periods` quarters (default 12, ≥ 8 required for YoY metrics).
    YoY same-quarter comparisons are computed automatically when sufficient history
    is available, eliminating the seasonal bias in QoQ comparisons.

    Args:
        ticker:  Ticker symbol (e.g. "AAPL").
        metrics: Subset of DEFAULT_METRICS to fetch (None = all).
        periods: Number of quarters to retrieve (default 12, min 8 for YoY).

    Returns:
        FundamentalSnapshot with latest values, QoQ growth rates, and YoY metrics.
    """
    if metrics is None:
        metrics = DEFAULT_METRICS

    # Ensure enough history for at least one YoY comparison (need 5+ for one YoY point)
    effective_periods = max(periods, 8)

    cik = lookup_cik(ticker)
    url = _FACTS_URL.format(cik=cik)
    resp = requests.get(url, headers=_HEADERS, timeout=60)
    resp.raise_for_status()

    facts = resp.json().get("facts", {})
    snap = FundamentalSnapshot(ticker=ticker, cik=cik, as_of=date.today())

    if "revenue" in metrics:
        h = _extract_periods(facts, "revenue", effective_periods)
        if h:
            snap.revenue = h[-1]
            snap.revenue_qoq_growth = _qoq_growth(h)
            snap.revenue_yoy_history = _yoy_growth_series(h)
            snap.revenue_yoy_pct = (
                snap.revenue_yoy_history[-1] if snap.revenue_yoy_history else None
            )

    if "net_income" in metrics:
        h = _extract_periods(facts, "net_income", effective_periods)
        if h:
            snap.net_income = h[-1]
            snap.net_income_history = h
            snap.net_income_qoq_growth = _qoq_growth(h)

    if "eps_diluted" in metrics:
        h = _extract_periods(facts, "eps_diluted", effective_periods)
        if h:
            snap.eps_diluted = h[-1]
            snap.eps_history = h
            snap.eps_qoq_growth = _qoq_growth(h)
            snap.eps_yoy_history = _yoy_growth_series(h)
            snap.eps_yoy_pct = (
                snap.eps_yoy_history[-1] if snap.eps_yoy_history else None
            )

    if "total_assets" in metrics:
        h = _extract_periods(facts, "total_assets", effective_periods)
        if h:
            snap.total_assets = h[-1]

    if "total_debt" in metrics:
        h = _extract_periods(facts, "total_debt", effective_periods)
        if h:
            snap.total_debt = h[-1]

    if "operating_cashflow" in metrics:
        h = _extract_periods(facts, "operating_cashflow", effective_periods)
        if h:
            snap.operating_cashflow = h[-1]

    if "capex" in metrics:
        h = _extract_periods(facts, "capex", effective_periods)
        if h:
            snap.capex = h[-1]

    if "shares_out" in metrics:
        h = _extract_periods(facts, "shares_out", effective_periods)
        if h:
            snap.shares_out = h[-1]

    # is_accelerating: both revenue AND eps YoY growth rates increasing for 2+ quarters.
    # Falls back to net_income YoY when eps data is unavailable.
    _eps_yoy = snap.eps_yoy_history or _yoy_growth_series(snap.net_income_history)
    _rev_yoy = snap.revenue_yoy_history
    snap.is_accelerating = (
        bool(_rev_yoy) and bool(_eps_yoy)
        and _is_yoy_accelerating(_rev_yoy)
        and _is_yoy_accelerating(_eps_yoy)
    )

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

# ── Earnings calendar helpers ─────────────────────────────────────────────────

def count_trading_days(start: date, end: date) -> int:
    """Count Mon–Fri trading days from start (exclusive) to end (inclusive)."""
    if end <= start:
        return 0
    count = 0
    current = start + timedelta(days=1)
    while current <= end:
        if current.weekday() < 5:
            count += 1
        current += timedelta(days=1)
    return count


def _ensure_earnings_calendar_table(con) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS earnings_calendar (
            symbol VARCHAR PRIMARY KEY,
            last_earnings_date DATE,
            next_earnings_date DATE,
            was_beat BOOLEAN,
            fetch_date DATE
        )
    """)


def _load_earnings_calendar_cache(
    symbol: str, max_age_days: int = 7
) -> Optional[tuple]:
    """Return (last_earnings_date, next_earnings_date, was_beat) if fresh, else None."""
    try:
        import duckdb
        from quantlab.storage import DB_PATH

        cutoff = (date.today() - timedelta(days=max_age_days)).isoformat()
        con = duckdb.connect(str(DB_PATH))
        _ensure_earnings_calendar_table(con)
        row = con.execute(
            """
            SELECT last_earnings_date, next_earnings_date, was_beat
            FROM earnings_calendar
            WHERE symbol = ? AND fetch_date >= ?
            """,
            [symbol, cutoff],
        ).fetchone()
        con.close()
        if row is None:
            return None
        last_d = date.fromisoformat(str(row[0])) if row[0] else None
        next_d = date.fromisoformat(str(row[1])) if row[1] else None
        was_beat = bool(row[2]) if row[2] is not None else None
        return (last_d, next_d, was_beat)
    except Exception as exc:
        logger.debug("earnings_calendar cache lookup failed for %s: %s", symbol, exc)
        return None


def _save_earnings_calendar_cache(
    symbol: str,
    last_earnings_date: Optional[date],
    next_earnings_date: Optional[date],
    was_beat: Optional[bool],
) -> None:
    try:
        import duckdb
        from quantlab.storage import DB_PATH

        con = duckdb.connect(str(DB_PATH))
        _ensure_earnings_calendar_table(con)
        con.execute(
            """
            INSERT OR REPLACE INTO earnings_calendar
                (symbol, last_earnings_date, next_earnings_date, was_beat, fetch_date)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                symbol,
                last_earnings_date.isoformat() if last_earnings_date else None,
                next_earnings_date.isoformat() if next_earnings_date else None,
                was_beat,
                date.today().isoformat(),
            ],
        )
        con.close()
    except Exception as exc:
        logger.warning("earnings_calendar cache write failed for %s: %s", symbol, exc)


def _fetch_quarterly_filing_dates(cik: str, limit: int = 6) -> list[date]:
    """Fetch dates of recent 10-Q and 10-K filings from SEC EDGAR submissions."""
    url = _SUBMISSIONS_URL.format(cik=cik)
    resp = requests.get(url, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    filed_dates = recent.get("filingDate", [])

    dates: list[date] = []
    for form, filed in zip(forms, filed_dates):
        if form in ("10-Q", "10-K"):
            try:
                dates.append(date.fromisoformat(filed))
            except Exception:
                pass

    return sorted(set(dates), reverse=True)[:limit]


def get_next_earnings_date(
    symbol: str, max_age_days: int = 7
) -> Optional[tuple[date, int]]:
    """
    Estimate next earnings date from SEC EDGAR 10-Q/10-K filing history.

    Uses the quarterly cadence of recent 10-Q/10-K filings to project the
    next reporting date. Results are cached in DuckDB for max_age_days.

    Returns:
        (estimated_date, trading_days_until) or None when unavailable.
    """
    cached = _load_earnings_calendar_cache(symbol, max_age_days)
    if cached is not None:
        _, next_d, _ = cached
        if next_d is not None:
            return (next_d, count_trading_days(date.today(), next_d))
        return None

    try:
        cik = lookup_cik(symbol)
        filing_dates = _fetch_quarterly_filing_dates(cik)

        if not filing_dates:
            return None

        # Average interval from recent consecutive filings
        if len(filing_dates) >= 2:
            intervals = [
                (filing_dates[i] - filing_dates[i + 1]).days
                for i in range(min(3, len(filing_dates) - 1))
            ]
            avg_interval = round(sum(intervals) / len(intervals))
        else:
            avg_interval = 91

        last_filing = filing_dates[0]
        next_date = last_filing + timedelta(days=avg_interval)
        today = date.today()

        # Advance until the projected date is in the future
        while next_date < today:
            next_date += timedelta(days=avg_interval)

        # Determine was_beat from EPS/NI history for the shared cache entry
        was_beat: Optional[bool] = None
        try:
            snap = fetch_fundamentals(symbol, metrics=["eps_diluted", "net_income"])
            history = snap.eps_history if snap.eps_history else snap.net_income_history
            if len(history) >= 2:
                was_beat = history[-1] > history[-2]
        except Exception:
            pass

        _save_earnings_calendar_cache(symbol, last_filing, next_date, was_beat)
        logger.debug(
            "%s: next_earnings=%s  was_beat=%s  interval=%dd",
            symbol, next_date, was_beat, avg_interval,
        )
        return (next_date, count_trading_days(today, next_date))

    except Exception as exc:
        logger.debug("get_next_earnings_date failed for %s: %s", symbol, exc)
        return None


def get_last_earnings_result(
    symbol: str, max_age_days: int = 7
) -> Optional[tuple[date, bool]]:
    """
    Return (last_earnings_date, was_beat) from most recent SEC EDGAR 10-Q/10-K.

    Shares the earnings_calendar DuckDB cache with get_next_earnings_date —
    calling either function first populates the cache for the other.

    Returns:
        (last_earnings_date, was_beat) or None when unavailable.
    """
    cached = _load_earnings_calendar_cache(symbol, max_age_days)
    if cached is not None:
        last_d, _, was_beat = cached
        if last_d is not None and was_beat is not None:
            return (last_d, was_beat)

    try:
        cik = lookup_cik(symbol)
        filing_dates = _fetch_quarterly_filing_dates(cik, limit=2)

        if not filing_dates:
            return None

        last_filing = filing_dates[0]

        was_beat = None
        try:
            snap = fetch_fundamentals(symbol, metrics=["eps_diluted", "net_income"])
            history = snap.eps_history if snap.eps_history else snap.net_income_history
            if len(history) >= 2:
                was_beat = history[-1] > history[-2]
        except Exception:
            pass

        if was_beat is None:
            return None

        # Estimate next date for cache completeness
        next_date = last_filing + timedelta(days=91)
        _save_earnings_calendar_cache(symbol, last_filing, next_date, was_beat)
        return (last_filing, was_beat)

    except Exception as exc:
        logger.debug("get_last_earnings_result failed for %s: %s", symbol, exc)
        return None


def format_yoy_summary(snap: FundamentalSnapshot, score: float) -> str:
    """Return a one-line YoY summary for display in scripts and demo output.

    Example: "AAPL: revenue_yoy=+17% eps_yoy=+22% accelerating=True score=0.48"
    """
    def _pct(v: Optional[float]) -> str:
        return f"{v * 100:+.0f}%" if v is not None else "N/A"

    return (
        f"{snap.ticker}: "
        f"revenue_yoy={_pct(snap.revenue_yoy_pct)} "
        f"eps_yoy={_pct(snap.eps_yoy_pct)} "
        f"accelerating={snap.is_accelerating} "
        f"score={score:.2f}"
    )


def compute_earnings_acceleration(snap: FundamentalSnapshot) -> float:
    """
    Score 0.0–1.0 reflecting year-over-year earnings growth and acceleration trend.

    Uses YoY same-quarter comparison (eliminates seasonal QoQ bias).
    Falls back to the legacy QoQ acceleration method when fewer than 5 quarters
    of history are available (e.g. recently-listed companies).

    Scoring (YoY path) — aligned with O'Neil's research showing 70%+ EPS growth
    precedes the biggest stock market winners:
        base from YoY magnitude:
            growth ≤ 0         →  0.0  (shrinking or flat)
            0  < growth < 20%  →  0.1  (not enough — insufficient earnings driver)
            20% ≤ growth < 50% →  0.3  (modest — below O'Neil minimum)
            50% ≤ growth < 70% →  0.6  (strong)
            70% ≤ growth < 100% → 0.8  (very strong — O'Neil 70% threshold)
            growth ≥ 100%      →  1.0  (explosive — highest conviction tier)
        acceleration bonus: +0.10 when snap.is_accelerating is True
                            (latest YoY rate > prior quarter YoY rate,
                            for BOTH revenue AND eps)
        result clamped to [0.0, 1.0]

    Interpretation:
        0.0   — negative or zero YoY growth
        0.1   — below 20% YoY — weak fundamental driver
        0.3   — 20–50% YoY — moderate, below the O'Neil threshold
        0.6   — 50–70% YoY — strong growth
        0.8   — 70–100% YoY — O'Neil-grade growth (historically leads big winners)
        1.0   — > 100% YoY — explosive hypergrowth
        +0.10 — acceleration trend on top of the above band
    """
    # ── Ensure EPS YoY is computed when not pre-filled ────────────────────────
    # eps_yoy_history is empty when < 5 EDGAR periods are available OR when the
    # GAAP field was not found on the first fetch pass.  If eps_history has 5+
    # values, compute a direct same-quarter YoY: current vs 4 periods back.
    # Write the result back to snap.eps_yoy_pct so _save_edgar_cache persists it.
    if not snap.eps_yoy_history and len(snap.eps_history) >= 5:
        _curr  = snap.eps_history[-1]
        _prior = snap.eps_history[-5]   # same fiscal quarter, prior year
        if abs(_prior) >= 1e-9:
            _eps_yoy_direct = round((_curr - _prior) / abs(_prior), 6)
            snap.eps_yoy_history = [_eps_yoy_direct]
            if snap.eps_yoy_pct is None:
                snap.eps_yoy_pct = _eps_yoy_direct

    # ── YoY path (preferred when ≥ 5 quarters of data available) ─────────────
    # Prioritise EPS YoY; fall back to revenue YoY only when EPS is unavailable.
    yoy_history = snap.eps_yoy_history or snap.revenue_yoy_history
    if yoy_history:
        latest_yoy = yoy_history[-1]
        if latest_yoy <= 0:
            base = 0.0
        elif latest_yoy < 0.20:
            base = 0.1   # 0–20%  — below O'Neil minimum
        elif latest_yoy < 0.50:
            base = 0.3   # 20–50% — modest
        elif latest_yoy < 0.70:
            base = 0.6   # 50–70% — strong
        elif latest_yoy < 1.00:
            base = 0.8   # 70–100% — O'Neil threshold
        else:
            base = 1.0   # >100%  — explosive
        bonus = 0.10 if snap.is_accelerating else 0.0
        return round(min(1.0, max(0.0, base + bonus)), 4)

    # ── Legacy QoQ fallback (< 5 quarters of data) ───────────────────────────
    history = snap.eps_history if len(snap.eps_history) >= 3 else snap.net_income_history
    if len(history) < 3:
        return 0.5

    a, b, c = history[-3], history[-2], history[-1]
    if abs(a) < 1e-9 or abs(b) < 1e-9:
        return 0.5

    prior_growth = (b - a) / abs(a)
    recent_growth = (c - b) / abs(b)
    acceleration = recent_growth - prior_growth

    clipped = max(-2.0, min(2.0, acceleration))
    return round((clipped + 2.0) / 4.0, 4)


# ── PEG ratio scoring (Boucher) ───────────────────────────────────────────────

def peg_ratio_score(
    forward_pe: Optional[float],
    eps_growth_pct: Optional[float],
) -> float:
    """
    Score 0.0–1.0 based on PEG ratio (Boucher filter).

    PEG = forward_pe / annual_eps_growth_rate_pct.

    A PEG below 1.0 indicates the stock is undervalued relative to its
    growth rate — the core insight from Boucher's PEG methodology.

    Args:
        forward_pe:      Forward (or trailing) price-to-earnings ratio.
        eps_growth_pct:  Annual EPS growth rate as a percentage (e.g. 25.0 for 25%).

    Returns:
        1.0 — PEG < 0.5  (deeply undervalued relative to growth)
        0.7 — PEG 0.5–1.0 (fairly valued — good entry zone)
        0.4 — PEG 1.0–1.5 (slightly expensive)
        0.0 — PEG > 1.5  (overvalued relative to growth)
        0.5 — neutral    (data unavailable or growth ≤ 0)
    """
    if forward_pe is None or eps_growth_pct is None:
        return 0.5
    if forward_pe <= 0 or eps_growth_pct <= 0:
        return 0.5   # negative P/E or negative growth → indeterminate

    peg = forward_pe / eps_growth_pct
    if peg < 0.5:
        return 1.0
    elif peg < 1.0:
        return 0.7
    elif peg < 1.5:
        return 0.4
    else:
        return 0.0


def get_edgar_peg_score(
    symbol: str,
    entry_close: float,
    max_age_days: int = 7,
) -> float:
    """
    Return an approximate PEG score using EDGAR-cached quarterly EPS data.

    Computes trailing P/E = entry_close / (eps_diluted_quarterly × 4) and
    divides by the cached YoY EPS growth rate (eps_growth column).  Returns
    0.0 when the cache is stale, data is absent, or EPS is non-positive.

    Does not make a network request — reads from the DuckDB edgar_fundamentals
    cache populated by get_edgar_acceleration().
    """
    try:
        import duckdb
        from quantlab.storage import DB_PATH

        cutoff = (date.today() - timedelta(days=max_age_days)).isoformat()
        con = duckdb.connect(str(DB_PATH))
        _ensure_edgar_table(con)
        row = con.execute(
            """
            SELECT eps_growth, eps_diluted FROM edgar_fundamentals
            WHERE symbol = ? AND fetch_date >= ?
            ORDER BY fetch_date DESC LIMIT 1
            """,
            [symbol, cutoff],
        ).fetchone()
        con.close()

        if row is None or row[0] is None or row[1] is None:
            return 0.0

        eps_yoy_decimal = float(row[0])   # stored as decimal, e.g. 0.47 = 47%
        eps_diluted_q   = float(row[1])   # most recent quarterly EPS

        if eps_yoy_decimal <= 0 or eps_diluted_q <= 0:
            return 0.0

        annual_eps  = eps_diluted_q * 4
        trailing_pe = entry_close / annual_eps
        growth_pct  = eps_yoy_decimal * 100   # convert to percentage for peg_ratio_score

        return peg_ratio_score(trailing_pe, growth_pct)

    except Exception as exc:
        logger.debug("get_edgar_peg_score failed for %s: %s", symbol, exc)
        return 0.0
