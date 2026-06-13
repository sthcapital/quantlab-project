"""SEC EDGAR fundamentals — no API key required."""

from __future__ import annotations

import functools
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_HEADERS = {"User-Agent": "QuantLab Research quantlab@sthcapital.com"}

# ── Rate limiter — max 8 req/sec per SEC EDGAR fair-use policy ────────────────
_EDGAR_LOCK = threading.Lock()
_EDGAR_LAST_REQ: float = 0.0
_EDGAR_MIN_INTERVAL: float = 1.0 / 8  # 125 ms between requests


def _edgar_get(url: str, timeout: int = 30) -> requests.Response:
    """Rate-limited GET for SEC EDGAR — enforces ≤ 8 req/sec globally."""
    global _EDGAR_LAST_REQ
    with _EDGAR_LOCK:
        elapsed = time.monotonic() - _EDGAR_LAST_REQ
        wait = _EDGAR_MIN_INTERVAL - elapsed
        if wait > 0:
            time.sleep(wait)
        _EDGAR_LAST_REQ = time.monotonic()
    resp = requests.get(url, headers=_HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp


DEFAULT_METRICS = [
    "revenue", "net_income", "eps_diluted", "total_assets",
    "total_debt", "operating_cashflow", "capex", "shares_out",
    "gross_profit",
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
    "gross_profit": [
        "GrossProfit",
        "GrossProfitLoss",
    ],
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
    # Negative→positive EPS transition on the latest quarter: a max-strength
    # earnings event stored as NULL% + this flag (a percentage of a negative
    # base is meaningless — SNDK -0.30 → +23.41 is the canonical case)
    eps_turned_positive: bool = False
    # Adjusted (non-GAAP) EPS from most recent 8-K press release (Exhibit 99.1)
    adj_eps: Optional[float] = None
    adj_eps_yoy_pct: Optional[float] = None   # (current - prior) / abs(prior)
    eps_surprise_pct: Optional[float] = None  # (adj_eps - consensus) / abs(consensus)
    # Gross margin trend: (latest_gm - gm_4q_ago); positive = expanding margins
    gross_margin_trend: Optional[float] = None


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
    # Auto-migration: add columns absent in earlier schema versions
    try:
        cols = {r[1] for r in con.execute("PRAGMA table_info(edgar_fundamentals)").fetchall()}
        if "eps_diluted" not in cols:
            con.execute("ALTER TABLE edgar_fundamentals ADD COLUMN eps_diluted DOUBLE")
        if "adj_eps" not in cols:
            con.execute("ALTER TABLE edgar_fundamentals ADD COLUMN adj_eps DOUBLE")
        if "adj_eps_yoy_pct" not in cols:
            con.execute("ALTER TABLE edgar_fundamentals ADD COLUMN adj_eps_yoy_pct DOUBLE")
        if "eps_surprise_pct" not in cols:
            con.execute("ALTER TABLE edgar_fundamentals ADD COLUMN eps_surprise_pct DOUBLE")
        if "gross_margin" not in cols:
            con.execute("ALTER TABLE edgar_fundamentals ADD COLUMN gross_margin DOUBLE")
        if "eps_turned_positive" not in cols:
            con.execute("ALTER TABLE edgar_fundamentals ADD COLUMN eps_turned_positive BOOLEAN")
        # Growth-filter inputs (2026-06-13): the acceleration qualifier needs
        # the YoY *trend* (latest rate > prior quarter's rate), and the IPO
        # path needs the raw quarter count.  Persisted so the growth universe
        # build stays a pure DuckDB read.  NULL = not computed (MISSING ≠ ZERO).
        if "rev_yoy_accel" not in cols:
            con.execute("ALTER TABLE edgar_fundamentals ADD COLUMN rev_yoy_accel BOOLEAN")
        if "eps_yoy_accel" not in cols:
            con.execute("ALTER TABLE edgar_fundamentals ADD COLUMN eps_yoy_accel BOOLEAN")
        if "shares_out" not in cols:
            con.execute("ALTER TABLE edgar_fundamentals ADD COLUMN shares_out DOUBLE")
        if "n_quarters" not in cols:
            con.execute("ALTER TABLE edgar_fundamentals ADD COLUMN n_quarters INTEGER")
    except Exception:
        pass


def _load_edgar_cache(symbol: str, max_age_days: int) -> tuple[bool, Optional[float]]:
    """Return (cache_hit, acceleration_score) for a fresh cache row.

    A hit with a NULL score means the last fetch found no usable fundamentals
    (e.g. a 20-F/40-F foreign filer) — callers must treat that as "unavailable"
    without re-fetching every scan.
    """
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
        if row is None:
            return False, None
        return True, (float(row[0]) if row[0] is not None else None)
    except Exception as exc:
        logger.debug("EDGAR cache lookup failed for %s: %s", symbol, exc)
        return False, None


def _save_edgar_cache(
    symbol: str,
    snap: FundamentalSnapshot,
    acceleration_score: Optional[float],
    consecutive_beats: int,
) -> None:
    """Write EDGAR-derived metrics to the edgar_fundamentals cache table. Non-fatal.

    MISSING ≠ ZERO: growth rates that could not be computed (e.g. 20-F/40-F
    foreign filers whose facts never appear under 10-K/10-Q forms) are stored
    as NULL, never 0.0 — a literal 0.0 means measured zero growth and would
    poison both display ("+0.0%") and scoring.
    """
    try:
        import duckdb
        from quantlab.storage import DB_PATH

        con = duckdb.connect(str(DB_PATH))
        _ensure_edgar_table(con)
        # Growth-filter inputs derived from the snapshot's YoY history.
        rev_accel = _is_yoy_accelerating(snap.revenue_yoy_history)
        eps_accel = _is_yoy_accelerating(snap.eps_yoy_history)
        n_quarters = len(snap.eps_history or snap.net_income_history) or None
        # RAW period-matched YoY only, uncapped — NULL means not computable
        # (quarantined / turned_positive / no base).  The old QoQ fallback is
        # gone: a Q4→Q1 sequential rate in a column consumers read as YoY was
        # itself a silent lie, and it backfilled quarantined values.
        con.execute(
            """
            INSERT OR REPLACE INTO edgar_fundamentals
                (symbol, fetch_date, acceleration_score, revenue_growth,
                 eps_growth, consecutive_beats, eps_diluted,
                 adj_eps, adj_eps_yoy_pct, eps_surprise_pct, gross_margin,
                 eps_turned_positive, rev_yoy_accel, eps_yoy_accel,
                 shares_out, n_quarters)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                symbol,
                date.today().isoformat(),
                acceleration_score,
                snap.revenue_yoy_pct,
                snap.eps_yoy_pct,
                consecutive_beats,
                snap.eps_diluted,
                snap.adj_eps,
                snap.adj_eps_yoy_pct,
                snap.eps_surprise_pct,
                snap.gross_margin_trend,
                snap.eps_turned_positive,
                rev_accel,
                eps_accel,
                snap.shares_out,
                n_quarters,
            ],
        )
        con.close()
    except Exception as exc:
        logger.warning("edgar_fundamentals cache write failed for %s: %s", symbol, exc)


# ── Fundamental data fetching ─────────────────────────────────────────────────

# Discrete fiscal quarter: a fact spanning ~3 months.  XBRL quarters run
# 84–98 days in practice (4-4-5 calendars included); the window tolerates
# leap/holiday drift without admitting 6-month YTD spans (~181 days).
_QTR_DUR_MIN = 80
_QTR_DUR_MAX = 100


def _extract_quarterly_dated(
    facts: dict, metric: str, periods: int,
) -> list[tuple[date, float]]:
    """
    Extract up to ``periods`` DISCRETE quarterly values as (period_end, value),
    oldest first.

    Selection is by explicit duration, not by exclusion: a quarterly value
    must come from a fact whose span is ~3 months (80–100 days).  Where a
    quarter exists only inside YTD cumulatives — Q2/Q3 in many 10-Qs, and Q4
    which most 10-Ks report only as the full-year total — the discrete
    quarter is DERIVED by subtracting successive same-fiscal-year YTD facts
    (facts sharing the same period start), accepting the difference only
    when the implied span is itself ~3 months.  Grouping by exact start date
    makes fiscal years that don't align to calendar quarters (CRDO: April
    FYE; UNFI: early-August FYE) work without special cases.

    2026-06-12 incident: the old end-date-keyed, position-aligned pipeline
    silently dropped Q4s (no discrete fact in 10-Ks) and then compared
    history[i] vs history[i-4] BY POSITION — with a quarter missing, "4 back"
    is not the same fiscal quarter, producing AEIS Rev +167.5% (actual +26%)
    and the wall of saturated EPS scores.
    """
    gaap_fields = _GAAP_FIELDS.get(metric, [])
    us_gaap = facts.get("us-gaap", {})

    # Build a candidate series per GAAP tag, then choose the FRESHEST one —
    # companies switch tags over time (UNFI: "Revenues" died in 2019, current
    # data lives under RevenueFromContractWithCustomerExcludingAssessedTax);
    # taking the first tag with any data can return a years-stale history.
    best: list[tuple[date, float]] = []
    for field_name in gaap_fields:
        if field_name not in us_gaap:
            continue
        units = us_gaap[field_name].get("units", {})
        unit_data = (
            units.get("USD")
            or units.get("USD/shares")
            or units.get("shares")
            or units.get("pure")
        )
        if not unit_data:
            continue

        # Parse + dedupe by exact (start, end) span, keeping the latest filing
        spans: dict[tuple[date, date], float] = {}
        for obs in sorted(unit_data, key=lambda x: x.get("filed", "")):
            if obs.get("form") not in ("10-Q", "10-K"):
                continue
            if not obs.get("start") or not obs.get("end"):
                continue
            try:
                s = date.fromisoformat(obs["start"])
                e = date.fromisoformat(obs["end"])
            except ValueError:
                continue
            if (e - s).days > 372:
                continue   # multi-year span — never useful here
            spans[(s, e)] = float(obs["val"])

        if not spans:
            continue

        # 1) Direct discrete quarters (~3-month spans)
        quarters: dict[date, float] = {}
        for (s, e), v in spans.items():
            if _QTR_DUR_MIN <= (e - s).days <= _QTR_DUR_MAX:
                quarters[e] = v

        # 2) Derive missing quarters from same-start YTD ladders:
        #    Q2 = 6mo−Q1(3mo facts share the FY start), Q3 = 9mo−6mo,
        #    Q4 = FY−9mo.  Same start date ⇒ same fiscal year by construction.
        by_start: dict[date, list[tuple[date, float]]] = {}
        for (s, e), v in spans.items():
            by_start.setdefault(s, []).append((e, v))
        for s, ladder in by_start.items():
            ladder.sort()
            for (e1, v1), (e2, v2) in zip(ladder, ladder[1:]):
                if e2 in quarters:
                    continue   # discrete fact exists — always preferred
                if _QTR_DUR_MIN <= (e2 - e1).days <= _QTR_DUR_MAX:
                    quarters[e2] = v2 - v1

        if not quarters:
            continue

        dated = sorted(quarters.items())
        if not best or dated[-1][0] > best[-1][0] or (
            dated[-1][0] == best[-1][0] and len(dated) > len(best)
        ):
            best = dated

    return best[-periods:]


def _extract_periods(facts: dict, metric: str, periods: int) -> list[float]:
    """Extract up to `periods` quarterly values for a metric. Returns oldest-first.

    Duration metrics (revenue, EPS, income…) go through the duration-explicit
    quarterly extraction (see _extract_quarterly_dated).  Instant balance-sheet
    metrics (assets, shares outstanding…) have no period start and are keyed
    by end date as before.
    """
    dated = _extract_quarterly_dated(facts, metric, periods)
    if dated:
        return [v for _, v in dated]

    # Instant-fact fallback (no "start" field): assets, shares outstanding…
    gaap_fields = _GAAP_FIELDS.get(metric, [])
    us_gaap = facts.get("us-gaap", {})
    for field_name in gaap_fields:
        if field_name not in us_gaap:
            continue
        units = us_gaap[field_name].get("units", {})
        unit_data = (
            units.get("USD")
            or units.get("USD/shares")
            or units.get("shares")
            or units.get("pure")
        )
        if not unit_data:
            continue
        seen: dict[str, float] = {}
        for obs in sorted(unit_data, key=lambda x: (x.get("end", ""), x.get("filed", ""))):
            if obs.get("form") not in ("10-Q", "10-K"):
                continue
            if not obs.get("end") or obs.get("start"):
                continue
            seen[obs["end"]] = float(obs["val"])
        if seen:
            return [seen[k] for k in sorted(seen)][-periods:]
    return []


def _remove_annual_outliers(values: list[float]) -> list[float]:
    """Remove values that are >2.5x all neighbors — catches annual totals
    slipping through the duration filter when EDGAR XBRL start-date
    metadata is absent or mis-tagged.

    Interior values are compared against both adjacent values.  Endpoint
    values (first/last) have only one adjacent value, so they are compared
    against their three nearest neighbors to resist seasonal quarterly
    variation (annual totals are ~4x quarterly; seasonal spikes rarely
    exceed 2x the surrounding quarters).
    """
    if len(values) < 3:
        return values
    to_remove: set[int] = set()
    for i in range(1, len(values) - 1):
        prev_abs = abs(values[i - 1])
        curr_abs = abs(values[i])
        next_abs = abs(values[i + 1])
        if prev_abs < 1e-9 or next_abs < 1e-9:
            continue
        if curr_abs > 2.5 * prev_abs and curr_abs > 2.5 * next_abs:
            to_remove.add(i)
    # Endpoint check: annual totals that land at the head or tail of the
    # sorted array have only one adjacent value and are missed by the loop
    # above.  Compare each endpoint against its three nearest neighbors.
    if len(values) >= 4:
        n = len(values)
        for ep_idx, ref_range in (
            (n - 1, range(n - 4, n - 1)),  # last value vs. 3 preceding
            (0,     range(1, 4)),           # first value vs. 3 following
        ):
            ep_abs = abs(values[ep_idx])
            refs = [abs(values[j]) for j in ref_range if 0 <= j < n]
            valid = [r for r in refs if r > 1e-9]
            if valid and ep_abs > 2.5 * max(valid):
                to_remove.add(ep_idx)
    return [v for i, v in enumerate(values) if i not in to_remove]


def _qoq_growth(history: list[float]) -> Optional[float]:
    if len(history) < 2:
        return None
    prev, curr = history[-2], history[-1]
    if abs(prev) < 1e-9:
        return None
    return round((curr - prev) / abs(prev), 6)


# YoY outlier bounds — shared by the report display cap and diagnostic logging.
# Upside: growth beyond +200% is suppressed from display as implausible.
# Downside: YoY revenue cannot mathematically fall below −100%, so a literal
# ±200% bound would never fire on declines — the artifact zone instead starts
# at −90%.  Real businesses almost never lose >90% of revenue YoY; such values
# are usually period-mismatch bugs (annual vs quarterly) or one-off prior-year
# items, so both tails are capped and the raw inputs logged for diagnosis.
YOY_OUTLIER_HI = 2.0
YOY_OUTLIER_LO = -0.90

# Quarantine bound for residual suspect YoY values: beyond ±1000% the inputs
# are almost certainly a data artifact (period mismatch survivor, restated
# base, near-zero denominator).  Quarantined values are stored as NULL with
# the raw inputs logged — clamping would convert "data problem" into a
# plausible lie.
YOY_QUARANTINE = 10.0

# Materiality floors for the YoY denominator: a base below these cannot
# support a growth-rate claim (quarantined, not computed).
_EPS_MIN_BASE = 0.05       # $0.05/share
_REV_MIN_BASE = 1_000_000  # $1M quarterly revenue


def winsorize_yoy(rate: Optional[float], cap: Optional[float] = None) -> Optional[float]:
    """
    Winsorize a raw YoY rate for SCORING consumption (None → scanner config
    ``yoy_winsorize``, default 3.0 = ±300%).  Storage stays raw/uncapped;
    only score inputs are clamped so one extreme print cannot dominate a
    composite.  None passes through (MISSING ≠ ZERO).
    """
    if rate is None:
        return None
    if cap is None:
        try:
            from quantlab.utils import get_config
            cap = float(get_config("scanner").get("yoy_winsorize", 3.0))
        except Exception:
            cap = 3.0
    return max(-cap, min(cap, rate))


def _dated_yoy_series(
    dated: list[tuple[date, float]],
    max_quarters: int = 4,
    label: str = "",
    min_base: float = 0.0,
) -> tuple[list[float], Optional[float], bool]:
    """
    Period-matched year-over-year growth from a dated quarterly series.

    For each quarter the comparison base is the entry whose period end is
    330–400 days earlier (closest to 365) — the SAME fiscal quarter one year
    before, regardless of gaps in the series.  Position-based alignment
    (history[i] vs history[i-4]) is exactly what corrupted AEIS/UNFI when a
    quarter was missing.

    Semantics:
        base >  min_base            → raw uncapped rate
        base ≤ 0, current > 0       → None + turned_positive (max-strength
                                       earnings event, not a percentage)
        base ≤ 0, current ≤ 0       → None (sign math is meaningless)
        0 < base < min_base         → None, raw inputs logged (quarantine —
                                       denominator too small to trust)
        |rate| > YOY_QUARANTINE     → None, raw inputs logged (quarantine)

    Returns:
        (rates, latest_rate, latest_turned_positive) — ``rates`` contains the
        computable rates oldest-first (Nones omitted) for trend/acceleration
        use; ``latest_rate`` is the most recent quarter's rate (None when
        not computable); ``latest_turned_positive`` flags a negative→positive
        transition on the most recent quarter.
    """
    rates: list[float] = []
    latest_rate: Optional[float] = None
    latest_tp = False

    for i, (end_i, curr) in enumerate(dated):
        base = None
        best_gap = None
        for end_j, prior in dated[:i]:
            gap = (end_i - end_j).days
            if 330 <= gap <= 400 and (best_gap is None or abs(gap - 365) < abs(best_gap - 365)):
                base, best_gap = prior, gap
        is_latest = i == len(dated) - 1

        if base is None:
            continue
        if base <= 0:
            if is_latest:
                latest_rate = None
                latest_tp = curr > 0
            continue
        if base < min_base:
            logger.warning(
                "YoY quarantine %s: base %.4f below materiality floor "
                "(current=%.4f end=%s) — stored NULL",
                label or "(unlabelled)", base, curr, end_i,
            )
            if is_latest:
                latest_rate, latest_tp = None, False
            continue

        rate = (curr - base) / base
        if abs(rate) > YOY_QUARANTINE:
            logger.warning(
                "YoY quarantine %s: rate=%+.0f%% current=%s prior_year=%s "
                "end=%s — stored NULL (suspect inputs, not clamped)",
                label or "(unlabelled)", rate * 100, curr, base, end_i,
            )
            if is_latest:
                latest_rate, latest_tp = None, False
            continue

        if rate > YOY_OUTLIER_HI or rate < YOY_OUTLIER_LO:
            logger.info(
                "YoY large move %s: rate=%+.1f%% current=%s prior_year=%s end=%s",
                label or "(unlabelled)", rate * 100, curr, base, end_i,
            )
        rates.append(rate)
        if is_latest:
            latest_rate = rate

    return rates[-max_quarters:], latest_rate, latest_tp


def _yoy_growth_series(
    history: list[float],
    max_quarters: int = 4,
    label: str = "",
) -> list[float]:
    """
    Compute year-over-year (same-quarter) growth rates from quarterly history.

    Requires at least 5 data points (current quarter + same quarter prior year).
    Skips quarters where the year-ago value is near zero (unreliable denominator).

    Args:
        history:      Quarterly values oldest-first (from _extract_periods).
        max_quarters: Maximum number of YoY rates to return.
        label:        "TICKER.metric" tag for outlier diagnostics (optional).

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
        rate = (curr - prior) / abs(prior)
        if rate > YOY_OUTLIER_HI or rate < YOY_OUTLIER_LO:
            # Log the raw period values so real collapses can be told apart
            # from period-mismatch artifacts (annual totals vs quarterly).
            logger.warning(
                "YoY outlier %s: rate=%+.1f%%  current=%s  prior_year=%s "
                "(check for period mismatch)",
                label or "(unlabelled)", rate * 100, curr, prior,
            )
        rates.append(rate)
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

    _rev_history: list[float] = []
    if "revenue" in metrics:
        dated = _extract_quarterly_dated(facts, "revenue", effective_periods)
        h = [v for _, v in dated]
        # NOTE: _remove_annual_outliers is deliberately NOT applied — duration-
        # explicit selection makes it unnecessary, and it would delete real
        # hypergrowth quarters (SNDK revenue 3.5× its neighbors is genuine).
        if h:
            _rev_history = h
            snap.revenue = h[-1]
            snap.revenue_qoq_growth = _qoq_growth(h)
            rates, latest, _ = _dated_yoy_series(
                dated, label=f"{ticker}.revenue", min_base=_REV_MIN_BASE,
            )
            snap.revenue_yoy_history = rates
            snap.revenue_yoy_pct = latest

    _ni_dated: list[tuple[date, float]] = []
    if "net_income" in metrics:
        _ni_dated = _extract_quarterly_dated(facts, "net_income", effective_periods)
        h = [v for _, v in _ni_dated]
        if h:
            snap.net_income = h[-1]
            snap.net_income_history = h
            snap.net_income_qoq_growth = _qoq_growth(h)

    if "eps_diluted" in metrics:
        dated = _extract_quarterly_dated(facts, "eps_diluted", effective_periods)
        h = [v for _, v in dated]
        if h:
            snap.eps_diluted = h[-1]
            snap.eps_history = h
            snap.eps_qoq_growth = _qoq_growth(h)
            rates, latest, tp = _dated_yoy_series(
                dated, label=f"{ticker}.eps", min_base=_EPS_MIN_BASE,
            )
            snap.eps_yoy_history = rates
            snap.eps_yoy_pct = latest
            snap.eps_turned_positive = tp

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

    if "gross_profit" in metrics:
        gp_h = _extract_periods(facts, "gross_profit", effective_periods)
        if gp_h and _rev_history:
            snap.gross_margin_trend = _compute_gross_margin_trend(gp_h, _rev_history)

    # is_accelerating: both revenue AND eps YoY growth rates increasing for 2+ quarters.
    # Falls back to net_income YoY when eps data is unavailable.
    _eps_yoy = snap.eps_yoy_history or _dated_yoy_series(
        _ni_dated, label=f"{ticker}.net_income"
    )[0]
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
    hit, cached = _load_edgar_cache(symbol, max_age_days)
    if hit:
        # A cached NULL means the last fetch found no usable fundamentals
        # (foreign filer / no quarterly facts) — honour it without re-fetching.
        if cached is None:
            logger.debug("%s: EDGAR acceleration cached as unavailable", symbol)
        else:
            logger.debug("%s: EDGAR acceleration from cache: %.4f", symbol, cached)
        return cached

    try:
        snap = fetch_fundamentals(symbol, metrics=["eps_diluted", "net_income", "revenue"])
        score = compute_earnings_acceleration(snap)
        consecutive = _count_consecutive_beats(snap.eps_history or snap.net_income_history)
        _save_edgar_cache(symbol, snap, score, consecutive)
        logger.debug(
            "%s: EDGAR acceleration fetched: %s  consecutive_beats=%d",
            symbol, f"{score:.4f}" if score is not None else "unavailable", consecutive,
        )
        return score
    except Exception as exc:
        logger.debug("EDGAR acceleration unavailable for %s: %s", symbol, exc)
        return None


def get_edgar_eps_growth(symbol: str, max_age_days: int = 7) -> Optional[float]:
    """Return cached YoY EPS growth rate for symbol from edgar_fundamentals, or None."""
    try:
        import duckdb
        from quantlab.storage import DB_PATH

        cutoff = (date.today() - timedelta(days=max_age_days)).isoformat()
        con = duckdb.connect(str(DB_PATH))
        _ensure_edgar_table(con)
        row = con.execute(
            "SELECT eps_growth FROM edgar_fundamentals "
            "WHERE symbol = ? AND fetch_date >= ? "
            "ORDER BY fetch_date DESC LIMIT 1",
            [symbol, cutoff],
        ).fetchone()
        con.close()
        return float(row[0]) if row is not None and row[0] is not None else None
    except Exception:
        return None


def get_edgar_revenue_growth(symbol: str, max_age_days: int = 7) -> Optional[float]:
    """Return cached YoY revenue growth rate for symbol from edgar_fundamentals, or None."""
    try:
        import duckdb
        from quantlab.storage import DB_PATH

        cutoff = (date.today() - timedelta(days=max_age_days)).isoformat()
        con = duckdb.connect(str(DB_PATH))
        _ensure_edgar_table(con)
        row = con.execute(
            "SELECT revenue_growth FROM edgar_fundamentals "
            "WHERE symbol = ? AND fetch_date >= ? "
            "ORDER BY fetch_date DESC LIMIT 1",
            [symbol, cutoff],
        ).fetchone()
        con.close()
        return float(row[0]) if row is not None and row[0] is not None else None
    except Exception:
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


# ── Adjusted EPS from 8-K press releases ─────────────────────────────────────

_ADJ_EPS_PATTERNS = [
    re.compile(r'[Aa]djusted\s+(?:diluted\s+)?EPS\s+of\s+\$(\d+\.\d+)'),
    re.compile(r'[Aa]djusted\s+(?:diluted\s+)?earnings\s+per\s+share\s+of\s+\$(\d+\.\d+)'),
    re.compile(r'\$(\d+\.\d+)\s+(?:adjusted|non-GAAP)\s+(?:diluted\s+)?EPS'),
]

# Combined: captures current and prior-year EPS in one match
# Handles "Adjusted EPS of $X.XX increased from $Y.YY" and "compared to $Y.YY"
_ADJ_EPS_WITH_PRIOR_PATTERN = re.compile(
    r'[Aa]djusted\s+(?:diluted\s+)?EPS\s+of\s+\$(\d+\.\d+).{0,80}?'
    r'(?:from|compared to)\s+\$(\d+\.\d+)',
    re.DOTALL,
)


def _strip_html(html: str) -> str:
    """Strip HTML tags and decode common entities (no external dependency)."""
    text = re.sub(r'<[^>]+>', ' ', html)
    for entity, char in (('&amp;', '&'), ('&nbsp;', ' '), ('&lt;', '<'), ('&gt;', '>')):
        text = text.replace(entity, char)
    return re.sub(r'\s+', ' ', text)


# EPS surprise — consensus estimate captured directly from the text
_EPS_ESTIMATE_PATTERNS = [
    re.compile(r'(?:versus|vs\.?)\s+(?:consensus|estimates?)\s+of\s+\$(\d+\.\d+)', re.I),
    re.compile(r'above\s+(?:the\s+)?consensus\s+estimate\s+of\s+\$(\d+\.\d+)', re.I),
    re.compile(r'compared?\s+to\s+(?:the\s+)?(?:consensus|estimate)\s+of\s+\$(\d+\.\d+)', re.I),
]

# EPS surprise — beat/miss *amount* captured (estimate = actual - beat_amount)
_EPS_BEAT_AMOUNT_PATTERNS = [
    re.compile(r'beat\s+(?:the\s+)?(?:estimates?|consensus)\s+by\s+\$(\d+\.\d+)', re.I),
]


def _parse_adjusted_eps(text: str) -> tuple[Optional[float], Optional[float]]:
    """Extract (current_adj_eps, prior_adj_eps) from press release text."""
    m = _ADJ_EPS_WITH_PRIOR_PATTERN.search(text)
    if m:
        try:
            return float(m.group(1)), float(m.group(2))
        except (ValueError, IndexError):
            pass

    for pattern in _ADJ_EPS_PATTERNS:
        m = pattern.search(text)
        if m:
            try:
                return float(m.group(1)), None
            except (ValueError, IndexError):
                continue

    return (None, None)


def _parse_eps_surprise(text: str, actual_eps: float) -> Optional[float]:
    """
    Parse EPS beat/miss vs. consensus from press release text.

    Returns surprise_pct = (actual - estimate) / abs(estimate), or None.

    Handles two text forms:
    - Estimate given directly: "vs. consensus of $X.XX"
    - Beat amount given: "beat estimates by $X.XX" → estimate = actual - beat
    """
    for pat in _EPS_ESTIMATE_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                estimate = float(m.group(1))
                if abs(estimate) >= 0.01:
                    return round((actual_eps - estimate) / abs(estimate), 6)
            except (ValueError, IndexError):
                continue

    for pat in _EPS_BEAT_AMOUNT_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                beat = float(m.group(1))
                estimate = actual_eps - beat
                if abs(estimate) >= 0.01:
                    return round(beat / abs(estimate), 6)
            except (ValueError, IndexError):
                continue

    return None


def fetch_adjusted_eps_from_8k(
    cik: str, symbol: str
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Parse adjusted/non-GAAP EPS and EPS surprise from the most recent EDGAR
    8-K Exhibit 99.1 (earnings press release).

    Args:
        cik:    Zero-padded 10-digit CIK string (e.g. "0000064803" for CVS).
        symbol: Ticker symbol — used for logging only.

    Returns:
        (current_adj_eps, prior_adj_eps, surprise_pct) or (None, None, None).
        surprise_pct = (actual - consensus) / abs(consensus); None when the
        press release contains no consensus comparison.
    """
    cutoff = date.today() - timedelta(days=90)
    cik_int = str(int(cik))  # strip leading zeros for EDGAR archive URLs

    # Step 1: submissions API — find most recent 8-K within 90 days
    try:
        data = _edgar_get(_SUBMISSIONS_URL.format(cik=cik)).json()
    except Exception as exc:
        logger.debug("%s: fetch_adjusted_eps_from_8k submissions failed: %s", symbol, exc)
        return (None, None, None)

    recent       = data.get("filings", {}).get("recent", {})
    forms        = recent.get("form", [])
    filing_dates = recent.get("filingDate", [])
    acc_numbers  = recent.get("accessionNumber", [])
    # items: e.g. "2.02,9.01" for earnings release; absent/empty for older filings
    items_list   = recent.get("items", [""] * len(forms))

    target_acc: Optional[str] = None
    for form, filed_str, acc, items in zip(forms, filing_dates, acc_numbers, items_list):
        if form not in ("8-K", "8-K/A"):
            continue
        # Skip non-earnings 8-Ks when items info is available.
        # Item 2.02 = Results of Operations; 7.01 = Regulation FD disclosure.
        items_str = str(items)
        if items_str and not any(x in items_str for x in ("2.02", "7.01")):
            continue
        try:
            filed = date.fromisoformat(filed_str)
        except Exception:
            continue
        if filed < cutoff:
            break  # list is newest-first; past the 90-day window
        target_acc = acc
        break  # first qualifying 8-K is the most recent one

    if target_acc is None:
        logger.debug("%s: no earnings 8-K found within 90 days", symbol)
        return (None, None, None)

    # Step 2: filing index HTML — locate Exhibit 99.1 filename
    acc_nodashes = target_acc.replace("-", "")
    index_url = (
        f"https://www.sec.gov/Archives/edgar/data/{cik_int}"
        f"/{acc_nodashes}/{target_acc}-index.html"
    )
    try:
        idx_html = _edgar_get(index_url).text
    except Exception as exc:
        logger.debug("%s: 8-K index fetch failed: %s", symbol, exc)
        return (None, None, None)

    # Parse index HTML: find the row with type EX-99 or EX-99.1 and extract the
    # direct archive href (not wrapped in XBRL viewer /ix?doc= prefix).
    ex991_url: Optional[str] = None
    m_ex = re.search(
        r'>EX-99(?:\.1)?</td>\s*<td[^>]*>\s*<a\s+href="(/Archives/edgar/data/[^"]+\.htm)"',
        idx_html, re.I,
    )
    if m_ex:
        ex991_url = f"https://www.sec.gov{m_ex.group(1)}"

    if ex991_url is None:
        logger.debug("%s: Exhibit 99.1 not found in 8-K filing index", symbol)
        return (None, None, None)

    # Step 3: fetch press release and parse adjusted EPS + surprise vs. consensus
    try:
        pr_text = _edgar_get(ex991_url, timeout=60).text
    except Exception as exc:
        logger.debug("%s: Exhibit 99.1 fetch failed: %s", symbol, exc)
        return (None, None, None)

    text = _strip_html(pr_text)
    adj_eps, prior_adj_eps = _parse_adjusted_eps(text)
    if adj_eps is not None:
        surprise_pct = _parse_eps_surprise(text, adj_eps)
        logger.debug(
            "%s: adjusted EPS from 8-K: current=%.2f  prior=%s  surprise=%s",
            symbol, adj_eps, prior_adj_eps, surprise_pct,
        )
        return (adj_eps, prior_adj_eps, surprise_pct)
    return (None, None, None)


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


def _compute_gross_margin_trend(
    gp_history: list[float],
    rev_history: list[float],
) -> Optional[float]:
    """YoY gross margin trend: (latest_gm - gm_4q_ago). Requires ≥5 aligned quarters."""
    n = min(len(gp_history), len(rev_history))
    if n < 5:
        return None
    gp = gp_history[-n:]
    rv = rev_history[-n:]
    margins: list[Optional[float]] = []
    for gp_q, rv_q in zip(gp, rv):
        margins.append(gp_q / rv_q if abs(rv_q) > 1e-9 else None)
    if margins[-1] is None or margins[-5] is None:
        return None
    return round(margins[-1] - margins[-5], 6)  # type: ignore[operator]


def _gross_margin_modifier(gm_trend: Optional[float]) -> float:
    """Gross-margin trend adjustment applied on top of the EPS-based score.

    Expanding margins confirm quality; contracting margins signal cost pressure.
    """
    if gm_trend is None:
        return 0.0
    if gm_trend > 0.02:
        return 0.05
    if gm_trend < -0.05:
        return -0.20
    if gm_trend < -0.02:
        return -0.10
    return 0.0


def _revenue_quality_modifier(revenue_yoy_pct: Optional[float]) -> float:
    """
    Revenue-quality adjustment applied on top of the EPS-based score.

    Rewards EPS growth confirmed by strong revenue; penalises EPS growth
    unsupported (or contradicted) by revenue trends.

        revenue growing  ≥ 10%              → +0.05  (confirms EPS strength)
        revenue growing   0–10%             →  0.00  (neutral)
        revenue declining 0–10%  (≥ −10%)  → −0.10  (mild concern)
        revenue declining 10–25% (≥ −25%)  → −0.20  (EPS may not be sustainable)
        revenue declining  >25%  (< −25%)  → −0.30  (serious red flag)
        revenue_yoy_pct is None             →  0.00  (no data — no penalty)
    """
    if revenue_yoy_pct is None:
        return 0.0
    if revenue_yoy_pct >= 0.10:
        return 0.05
    if revenue_yoy_pct >= 0.0:
        return 0.0
    if revenue_yoy_pct >= -0.10:
        return -0.10
    if revenue_yoy_pct >= -0.25:
        return -0.20
    return -0.30


def compute_earnings_acceleration(snap: FundamentalSnapshot) -> Optional[float]:
    """
    Score 0.0–1.0 reflecting year-over-year earnings growth and acceleration trend.
    Returns None when the snapshot contains no usable earnings history at all
    (MISSING ≠ ZERO — e.g. foreign 20-F/40-F filers with no 10-K/10-Q facts).

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
        revenue quality modifier (snap.revenue_yoy_pct):
            ≥ 10%       → +0.05  (revenue confirms EPS strength)
            0–10%       →  0.00  (neutral)
            declining 0–10%    → −0.10  (mild concern)
            declining 10–25%   → −0.20  (EPS growth may not be sustainable)
            declining  >25%    → −0.30  (serious red flag)
            None        →  0.00  (no data — no penalty)
        result clamped to [0.0, 1.0]

    Interpretation:
        0.0   — negative or zero YoY growth
        0.1   — below 20% YoY — weak fundamental driver
        0.3   — 20–50% YoY — moderate, below the O'Neil threshold
        0.6   — 50–70% YoY — strong growth
        0.8   — 70–100% YoY — O'Neil-grade growth (historically leads big winners)
        1.0   — > 100% YoY — explosive hypergrowth
        +0.10 — acceleration trend on top of the above band
        ±0.05/0.10/0.20/0.30 — revenue quality modifier
    """
    # NOTE: the old positional backfill (eps_history[-1] vs eps_history[-5])
    # is gone — "5 back" in a list with missing quarters is NOT the same
    # fiscal quarter (the 2026-06-12 AEIS/UNFI corruption).  YoY now comes
    # exclusively from the period-matched _dated_yoy_series in fetch.

    # ── Turned-positive: negative→positive EPS is a max-strength earnings
    # event (stored as NULL% — a rate off a negative base is meaningless).
    # It qualifies as the top band wherever "EPS YoY ≥ X%" is evaluated.
    if snap.eps_turned_positive:
        bonus = 0.10 if snap.is_accelerating else 0.0
        rev_mod = _revenue_quality_modifier(winsorize_yoy(snap.revenue_yoy_pct))
        gm_mod  = _gross_margin_modifier(snap.gross_margin_trend)
        return round(min(1.0, max(0.0, 1.0 + bonus + rev_mod + gm_mod)), 4)

    # ── YoY path (preferred when ≥ 5 quarters of data available) ─────────────
    # Prefer adjusted EPS YoY (non-GAAP, from 8-K press release) over GAAP eps_yoy;
    # fall back to revenue_yoy when neither is available.
    if snap.adj_eps_yoy_pct is not None:
        yoy_history: list[float] = [snap.adj_eps_yoy_pct]
    else:
        yoy_history = snap.eps_yoy_history or snap.revenue_yoy_history
    if yoy_history:
        # Winsorized at scoring time (config yoy_winsorize, default ±300%);
        # the stored value stays raw
        latest_yoy = winsorize_yoy(yoy_history[-1])
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
        rev_mod = _revenue_quality_modifier(winsorize_yoy(snap.revenue_yoy_pct))
        gm_mod  = _gross_margin_modifier(snap.gross_margin_trend)
        return round(min(1.0, max(0.0, base + bonus + rev_mod + gm_mod)), 4)

    # ── Legacy QoQ fallback (< 5 quarters of data) ───────────────────────────
    history = snap.eps_history if len(snap.eps_history) >= 3 else snap.net_income_history
    if len(history) < 3:
        # Not enough data to measure anything — unavailable, not neutral.
        return None

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
) -> Optional[float]:
    """
    Return an approximate PEG score using EDGAR-cached quarterly EPS data.

    Computes trailing P/E = entry_close / (eps_diluted_quarterly × 4) and
    divides by the cached YoY EPS growth rate (eps_growth column).  Returns
    None when the PEG is not computable — cache stale, data absent, or EPS /
    growth non-positive (MISSING ≠ ZERO: a real 0.0 means PEG > 1.5, i.e.
    measured-overvalued, and must stay distinct from "no data").

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
            return None

        eps_yoy_decimal = float(row[0])   # stored as decimal, e.g. 0.47 = 47%
        eps_diluted_q   = float(row[1])   # most recent quarterly EPS

        if eps_yoy_decimal <= 0 or eps_diluted_q <= 0:
            return None   # PEG indeterminate for negative EPS or shrinking earnings

        annual_eps  = eps_diluted_q * 4
        trailing_pe = entry_close / annual_eps
        growth_pct  = eps_yoy_decimal * 100   # convert to percentage for peg_ratio_score

        return peg_ratio_score(trailing_pe, growth_pct)

    except Exception as exc:
        logger.debug("get_edgar_peg_score failed for %s: %s", symbol, exc)
        return None
