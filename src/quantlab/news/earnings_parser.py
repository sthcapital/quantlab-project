"""
quantlab.news.earnings_parser — real-time beat/miss detection from IBKR press releases.

Parses earnings press release headlines to extract EPS actuals, consensus estimates,
revenue figures, and beat/miss signals without needing full article text.

Common formats handled:
    "Reports Q2 EPS $2.01 vs $1.88 Estimate"
    "Q3 Earnings: EPS $1.52 Beats $1.44 Estimate"
    "Revenue $94.9B vs $94.1B Expected"
    "Reports fiscal Q2 2026 results: EPS $2.01, Revenue $94.9B"
    "NVDA Q4: EPS $0.52 Beats $0.45 Estimate; Revenue $22.1B vs $20.6B Expected"
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


# ── Earnings headline indicators ───────────────────────────────────────────────

_EARNINGS_PHRASES: tuple[str, ...] = (
    "quarterly results",
    "per share",
    "fiscal q",
    "q1 earnings", "q2 earnings", "q3 earnings", "q4 earnings",
    "beats estimates", "misses estimates",
    "beats consensus", "misses consensus",
    "reports quarterly", "reports fiscal",
    "reports q1", "reports q2", "reports q3", "reports q4",
    "quarterly earnings",
    "beats expectations", "misses expectations",
)


# ── Regex building blocks ──────────────────────────────────────────────────────

_NUM = r'\d+(?:,\d+)*(?:\.\d+)?'
_NUM_GRP = r'(' + _NUM + r')'
_SUFFIX_GRP = r'([BMK])?'

# EPS actual: "EPS $2.01" or "EPS of $2.01"
_EPS_ACTUAL_RE = re.compile(
    r'\bEPS\s+(?:of\s+)?-?\$\s*' + _NUM_GRP,
    re.IGNORECASE,
)
# "$2.01 per share" — fallback when no EPS prefix
_PER_SHARE_RE = re.compile(
    r'-?\$\s*' + _NUM_GRP + r'\s+per\s+share',
    re.IGNORECASE,
)

# Estimate comparisons (searched in the text after the EPS actual)
_VS_RE = re.compile(
    r'\b(?:vs\.?|versus)\s+-?\$\s*' + _NUM_GRP,
    re.IGNORECASE,
)
_BEATS_RE = re.compile(
    r'\bbeats?\s+-?\$\s*' + _NUM_GRP,
    re.IGNORECASE,
)
_MISSES_RE = re.compile(
    r'\bmisses?\s+-?\$\s*' + _NUM_GRP,
    re.IGNORECASE,
)
_EST_OF_RE = re.compile(
    r'\bestimate(?:d)?\s+(?:of\s+)?-?\$\s*' + _NUM_GRP,
    re.IGNORECASE,
)

# Revenue with vs-estimate: "Revenue $94.9B vs $94.1B [Expected|Estimate]"
# Groups: (actual_val, actual_suffix, estimate_val, estimate_suffix)
_REVENUE_VS_RE = re.compile(
    r'\b(?:revenue|sales)\b[^$]*-?\$\s*' + _NUM_GRP + r'\s*' + _SUFFIX_GRP
    + r'\s+(?:vs\.?|versus|compared\s+to)\s+-?\$\s*' + _NUM_GRP + r'\s*' + _SUFFIX_GRP,
    re.IGNORECASE,
)
# Revenue only (no comparison): "Revenue $94.9B"
# Groups: (val, suffix)
_REVENUE_ACTUAL_RE = re.compile(
    r'\b(?:revenue|sales)\b[^$]*-?\$\s*' + _NUM_GRP + r'\s*' + _SUFFIX_GRP,
    re.IGNORECASE,
)

_QUARTER_RE = re.compile(r'\b(?:fiscal\s+)?[Qq]([1-4])\b')
_YEAR_RE = re.compile(r'\b(20\d{2})\b')

# Bare dollar immediately before "beats" / "misses" (no EPS keyword)
_PREV_DOLLAR_RE = re.compile(r'\$\s*(' + _NUM + r')\s*$')


# ── Helpers ────────────────────────────────────────────────────────────────────

def _to_float(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def _scale_to_millions(value: float | None, suffix: str | None) -> float | None:
    """Scale a revenue value to millions using B/M/K suffix."""
    if value is None:
        return None
    if not suffix:
        return value  # assume raw millions when no suffix
    s = suffix.upper()
    if s == "B":
        return round(value * 1_000, 4)
    if s == "K":
        return round(value / 1_000, 4)
    return value   # M → already millions


# ── Public API ─────────────────────────────────────────────────────────────────

def is_earnings_headline(headline: str) -> bool:
    """
    Return True when the headline is likely an earnings press release.

    Checks for earnings indicators:
        "reports", "quarterly results", "Q1/Q2/Q3/Q4 earnings",
        "fiscal", "EPS", "per share", and related beat/miss phrases.
    """
    low = " " + headline.lower() + " "   # pad for word-boundary checks

    for phrase in _EARNINGS_PHRASES:
        if phrase in low:
            return True

    # Standalone EPS as a word
    if re.search(r'\beps\b', low):
        return True

    return False


@dataclass
class ParsedEarnings:
    """Fields extracted from a single earnings press release headline."""

    eps_actual: float | None = None
    eps_estimate: float | None = None
    revenue_actual: float | None = None     # in millions
    revenue_estimate: float | None = None   # in millions
    quarter: str | None = None
    fiscal_year: int | None = None
    eps_beat: bool | None = None
    revenue_beat: bool | None = None


def parse_earnings_headline(headline: str) -> ParsedEarnings:
    """
    Extract EPS, revenue, quarter, fiscal year, and beat/miss flags from a headline.

    Returns a ParsedEarnings with None for any field not found in the text.
    Revenue values are normalised to millions (B suffix → ×1 000, K → ÷1 000).

    Beat/miss flags are set to None when no consensus estimate is present.
    """
    result = ParsedEarnings()

    # ── Quarter ────────────────────────────────────────────────────────────────
    qm = _QUARTER_RE.search(headline)
    if qm:
        result.quarter = f"Q{qm.group(1)}"

    # ── Fiscal year ────────────────────────────────────────────────────────────
    ym = _YEAR_RE.search(headline)
    if ym:
        result.fiscal_year = int(ym.group(1))

    # ── EPS ────────────────────────────────────────────────────────────────────
    eps_m = _EPS_ACTUAL_RE.search(headline)
    if not eps_m:
        eps_m = _PER_SHARE_RE.search(headline)

    if eps_m:
        result.eps_actual = _to_float(eps_m.group(1))
        rest = headline[eps_m.end():]

        # Priority: explicit beats/misses > "vs" comparison > "estimate of"
        bm = _BEATS_RE.search(rest)
        mm = _MISSES_RE.search(rest)
        vm = _VS_RE.search(rest)
        em = _EST_OF_RE.search(rest)

        if bm:
            result.eps_estimate = _to_float(bm.group(1))
            result.eps_beat = True
        elif mm:
            result.eps_estimate = _to_float(mm.group(1))
            result.eps_beat = False
        elif vm:
            result.eps_estimate = _to_float(vm.group(1))
            if result.eps_actual is not None and result.eps_estimate is not None:
                result.eps_beat = result.eps_actual > result.eps_estimate
        elif em:
            result.eps_estimate = _to_float(em.group(1))
            if result.eps_actual is not None and result.eps_estimate is not None:
                result.eps_beat = result.eps_actual > result.eps_estimate

    # Fallback: explicit "beats $X" / "misses $X" without an EPS prefix
    if result.eps_beat is None:
        bm = _BEATS_RE.search(headline)
        mm = _MISSES_RE.search(headline)
        if bm and result.eps_actual is None:
            before = headline[:bm.start()].rstrip()
            pdm = _PREV_DOLLAR_RE.search(before)
            if pdm:
                result.eps_actual = _to_float(pdm.group(1))
                result.eps_estimate = _to_float(bm.group(1))
                result.eps_beat = True
        elif mm and result.eps_actual is None:
            before = headline[:mm.start()].rstrip()
            pdm = _PREV_DOLLAR_RE.search(before)
            if pdm:
                result.eps_actual = _to_float(pdm.group(1))
                result.eps_estimate = _to_float(mm.group(1))
                result.eps_beat = False

    # ── Revenue ────────────────────────────────────────────────────────────────
    rvm = _REVENUE_VS_RE.search(headline)
    if rvm:
        result.revenue_actual = _scale_to_millions(
            _to_float(rvm.group(1)), rvm.group(2)
        )
        result.revenue_estimate = _scale_to_millions(
            _to_float(rvm.group(3)), rvm.group(4)
        )
        if result.revenue_actual is not None and result.revenue_estimate is not None:
            result.revenue_beat = result.revenue_actual > result.revenue_estimate
    else:
        rm = _REVENUE_ACTUAL_RE.search(headline)
        if rm:
            result.revenue_actual = _scale_to_millions(
                _to_float(rm.group(1)), rm.group(2)
            )

    return result


def compute_beat_score(
    eps_beat: bool | None,
    revenue_beat: bool | None,
) -> float:
    """
    Return a 0.0–1.0 score reflecting overall earnings beat/miss outcome.

    Scoring table:
        both beat               → 1.0
        eps beat, revenue N/A   → 0.7
        revenue beat, eps N/A   → 0.5
        one beat, one miss      → 0.3
        eps miss, revenue N/A   → 0.3
        revenue miss, eps N/A   → 0.3
        both miss               → 0.0
        neither known           → 0.5  (neutral — insufficient data)
    """
    if eps_beat is None and revenue_beat is None:
        return 0.5  # insufficient data

    if eps_beat is not None and revenue_beat is not None:
        if eps_beat and revenue_beat:
            return 1.0
        if not eps_beat and not revenue_beat:
            return 0.0
        return 0.3  # mixed: one beat, one miss

    # Only one metric available
    if eps_beat is not None:
        return 0.7 if eps_beat else 0.3
    # revenue_beat is not None, eps unknown
    return 0.5 if revenue_beat else 0.3


@dataclass
class EarningsResult:
    """
    Parsed and scored earnings result from a press release headline.
    Persisted to DuckDB for real-time conviction adjustment in the scanner.
    """

    symbol: str
    report_date: str            # YYYY-MM-DD
    quarter: str | None
    fiscal_year: int | None
    eps_actual: float | None
    eps_estimate: float | None
    eps_beat: bool | None
    revenue_actual: float | None    # millions
    revenue_estimate: float | None  # millions
    revenue_beat: bool | None
    beat_score: float               # 0.0–1.0 from compute_beat_score()
    headline_source: str            # original headline (truncated to 500 chars)


def make_earnings_result(
    symbol: str,
    headline: str,
    report_date: str | None = None,
) -> EarningsResult | None:
    """
    Parse a headline and build an EarningsResult.

    Returns None when is_earnings_headline() is False (not an earnings release).
    report_date defaults to today when not provided.
    """
    if not is_earnings_headline(headline):
        return None
    parsed = parse_earnings_headline(headline)
    score = compute_beat_score(parsed.eps_beat, parsed.revenue_beat)
    return EarningsResult(
        symbol=symbol,
        report_date=report_date or date.today().isoformat(),
        quarter=parsed.quarter,
        fiscal_year=parsed.fiscal_year,
        eps_actual=parsed.eps_actual,
        eps_estimate=parsed.eps_estimate,
        eps_beat=parsed.eps_beat,
        revenue_actual=parsed.revenue_actual,
        revenue_estimate=parsed.revenue_estimate,
        revenue_beat=parsed.revenue_beat,
        beat_score=score,
        headline_source=headline[:500],
    )


# ── DuckDB persistence ─────────────────────────────────────────────────────────

def _ensure_earnings_results_table(con) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS earnings_results (
            symbol            VARCHAR,
            report_date       DATE,
            quarter           VARCHAR,
            fiscal_year       INTEGER,
            eps_actual        DOUBLE,
            eps_estimate      DOUBLE,
            eps_beat          BOOLEAN,
            revenue_actual    DOUBLE,
            revenue_estimate  DOUBLE,
            revenue_beat      BOOLEAN,
            beat_score        DOUBLE,
            headline_source   VARCHAR,
            PRIMARY KEY (symbol, report_date)
        )
    """)


def store_earnings_result(result: EarningsResult) -> None:
    """
    Persist an EarningsResult to the DuckDB earnings_results table.
    Non-fatal — logs a warning on any failure.
    """
    try:
        import duckdb
        from quantlab.storage import DB_PATH

        con = duckdb.connect(str(DB_PATH))
        _ensure_earnings_results_table(con)
        con.execute(
            """
            INSERT OR REPLACE INTO earnings_results
                (symbol, report_date, quarter, fiscal_year,
                 eps_actual, eps_estimate, eps_beat,
                 revenue_actual, revenue_estimate, revenue_beat,
                 beat_score, headline_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                result.symbol,
                result.report_date,
                result.quarter,
                result.fiscal_year,
                result.eps_actual,
                result.eps_estimate,
                result.eps_beat,
                result.revenue_actual,
                result.revenue_estimate,
                result.revenue_beat,
                result.beat_score,
                result.headline_source,
            ],
        )
        con.close()
        logger.debug(
            "earnings_results: stored %s %s  beat_score=%.2f",
            result.symbol, result.report_date, result.beat_score,
        )
    except Exception as exc:
        logger.warning(
            "earnings_results: write failed for %s: %s", result.symbol, exc
        )


def get_recent_earnings_result(
    symbol: str,
    max_days: int = 5,
) -> EarningsResult | None:
    """
    Return the most recent EarningsResult for symbol within max_days trading days.

    Uses a calendar-day pre-filter (max_days × 3) to handle weekends and holidays,
    then validates with an exact trading-day count before returning.

    Returns None when no result is found or DuckDB is unavailable.
    """
    try:
        import duckdb
        from quantlab.storage import DB_PATH
        from quantlab.providers.edgar import count_trading_days

        # Calendar pre-filter: 3× multiplier safely covers weekends / holidays
        cutoff = (date.today() - timedelta(days=max_days * 3)).isoformat()

        con = duckdb.connect(str(DB_PATH))
        _ensure_earnings_results_table(con)
        row = con.execute(
            """
            SELECT symbol, report_date, quarter, fiscal_year,
                   eps_actual, eps_estimate, eps_beat,
                   revenue_actual, revenue_estimate, revenue_beat,
                   beat_score, headline_source
            FROM earnings_results
            WHERE symbol = ? AND report_date >= ?
            ORDER BY report_date DESC LIMIT 1
            """,
            [symbol, cutoff],
        ).fetchone()
        con.close()

        if row is None:
            return None

        # Exact trading-day check
        result_date = date.fromisoformat(str(row[1]))
        if count_trading_days(result_date, date.today()) > max_days:
            return None

        return EarningsResult(
            symbol=row[0],
            report_date=str(row[1]),
            quarter=row[2],
            fiscal_year=row[3],
            eps_actual=float(row[4]) if row[4] is not None else None,
            eps_estimate=float(row[5]) if row[5] is not None else None,
            eps_beat=bool(row[6]) if row[6] is not None else None,
            revenue_actual=float(row[7]) if row[7] is not None else None,
            revenue_estimate=float(row[8]) if row[8] is not None else None,
            revenue_beat=bool(row[9]) if row[9] is not None else None,
            beat_score=float(row[10]) if row[10] is not None else 0.5,
            headline_source=row[11] or "",
        )

    except Exception as exc:
        logger.debug(
            "get_recent_earnings_result failed for %s: %s", symbol, exc
        )
        return None
