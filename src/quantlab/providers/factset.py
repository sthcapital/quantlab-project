"""
quantlab.providers.factset — FactSet data provider.

Implements the full FactSet REST API client with real endpoint URLs and
authentication.  Currently operates in mock mode because credentials are
pending provisioning.  Set ``use_mock=False`` once the account is activated
and ``FACTSET_USERNAME`` / ``FACTSET_API_KEY`` environment variables are set.

Authentication (FactSet standard):
    FactSet REST APIs use HTTP Basic Authentication:
        username  →  FACTSET_USERNAME env var  (format: {serial}@{company} or {serial})
        password  →  FACTSET_API_KEY env var   (the API key issued by FactSet)
    Header:  Authorization: Basic {base64(username:api_key)}

    OAuth 2.0 (newer deployments):
        Token endpoint: https://auth.factset.com/as/token.oauth2
        Scope: "fds-api"

API surface covered:
    get_earnings_estimates(symbol)  — consensus EPS/revenue estimates per period
    get_surprise_history(symbol)    — actual vs estimate per quarter, surprise %
    get_fundamentals(symbol)        — P/E, margins, growth rates, FCF yield
    get_transcript(symbol, date)    — earnings call transcript with speaker segments
    get_options_chain(symbol)       — full option chain with Greeks

FactSet symbol format: "{TICKER}-{EXCHANGE}" e.g. "AAPL-US", "MSFT-US".
The provider normalises plain tickers automatically.

Reference:
    https://developer.factset.com/api-catalog
    FactSet Estimates API v2:      /content/factset-estimates/v2/
    FactSet Fundamentals API v2:   /content/factset-fundamentals/v2/
    Events & Transcripts API v1:   /content/events-and-transcripts/v1/
    FactSet Options API v1:        /content/factset-options/v1/
"""

from __future__ import annotations

import base64
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import requests

logger = logging.getLogger(__name__)

_BASE_URL       = "https://api.factset.com/content"
_AUTH_URL       = "https://auth.factset.com/as/token.oauth2"
_DEFAULT_EXCH   = "US"           # FactSet exchange suffix for US equities
_MAX_RETRIES    = 3
_RETRY_SLEEP    = 2.0            # seconds between retries on transient errors


# ── Response dataclasses ───────────────────────────────────────────────────────

@dataclass
class EarningsEstimate:
    """Single-period consensus EPS + revenue estimate from FactSet."""

    symbol: str
    period: str                     # e.g. "Q2 2026", "FY 2026"
    period_type: str                # "quarterly" | "annual"
    estimate_date: str              # YYYY-MM-DD of estimate snapshot

    consensus_eps: float | None     # mean analyst EPS estimate
    high_eps: float | None
    low_eps: float | None

    consensus_revenue: float | None  # mean revenue estimate ($M)
    high_revenue: float | None
    low_revenue: float | None

    num_analysts_eps: int           # analysts contributing to EPS estimate
    num_analysts_revenue: int

    eps_growth_yoy: float | None    # consensus EPS vs same period prior year


@dataclass
class EarningsSurprise:
    """One quarter's actual EPS vs consensus — the earnings surprise record."""

    symbol: str
    report_date: str        # YYYY-MM-DD of earnings release
    fiscal_year: int
    fiscal_quarter: int     # 1–4
    period_label: str       # e.g. "Q1 2026"

    actual_eps: float | None
    consensus_eps: float | None
    surprise_pct: float | None      # (actual - consensus) / abs(consensus) × 100

    actual_revenue: float | None    # $M
    consensus_revenue: float | None
    revenue_surprise_pct: float | None

    guidance_raised: bool | None    # True when mgmt raised forward guidance


@dataclass
class CompanyFundamentals:
    """
    Snapshot of company financial ratios and growth metrics from FactSet.

    All growth rates are year-over-year unless noted.
    Revenue and FCF are in $M (trailing twelve months unless noted).
    """

    symbol: str
    as_of: str              # YYYY-MM-DD of the snapshot

    # Valuation
    market_cap: float | None        # $B
    enterprise_value: float | None  # $B
    pe_ratio: float | None          # trailing P/E
    pe_forward: float | None        # NTM P/E (next twelve months)
    peg_ratio: float | None         # P/E ÷ long-term EPS growth rate
    ps_ratio: float | None          # price / sales (TTM)
    pb_ratio: float | None          # price / book

    # Profitability
    revenue_ttm: float | None       # $M TTM
    revenue_growth_yoy: float | None   # % YoY
    gross_margin: float | None         # %
    operating_margin: float | None     # %
    net_margin: float | None           # %
    ebitda_margin: float | None        # %

    # Per-share
    eps_ttm: float | None
    eps_growth_yoy: float | None       # % YoY
    eps_growth_3yr_cagr: float | None  # 3-year CAGR

    # Capital efficiency
    return_on_equity: float | None     # %
    return_on_assets: float | None     # %
    return_on_invested_capital: float | None  # %

    # Cash flow
    free_cash_flow_ttm: float | None   # $M
    free_cash_flow_yield: float | None # FCF / market cap %
    fcf_growth_yoy: float | None       # %

    # Balance sheet
    debt_to_equity: float | None
    net_debt_to_ebitda: float | None
    current_ratio: float | None

    # Estimates context
    next_earnings_date: str | None     # YYYY-MM-DD
    earnings_acceleration: bool        # True when EPS growth accelerating


@dataclass
class TranscriptSegment:
    """One speaker's turn in an earnings call transcript."""

    speaker: str        # full name
    role: str           # "CEO" | "CFO" | "Analyst" | "Operator"
    firm: str           # company name or "Analyst firm name"
    text: str           # verbatim spoken text


@dataclass
class EarningsTranscript:
    """
    Full earnings call transcript from FactSet Events & Transcripts API.

    Segments are ordered chronologically: Operator → Management → Q&A.
    ``raw_text`` concatenates all segments for NLP/sentiment tasks.
    """

    symbol: str
    event_date: str         # YYYY-MM-DD
    fiscal_period: str      # e.g. "Q1 2026"
    event_type: str         # "Earnings Call" | "Guidance Update" | "Analyst Day"
    duration_minutes: int   # approximate

    segments: list[TranscriptSegment] = field(default_factory=list)
    raw_text: str = ""      # full concatenated text (populated on construction)
    factset_event_id: str = ""  # FactSet internal event ID for re-fetch


@dataclass
class FactSetOptionContract:
    """
    One option contract from the FactSet Options API.

    Greeks are model-derived (Black-Scholes or binomial per FactSet config).
    """

    underlying: str
    symbol: str             # OCC-style option ticker e.g. AAPL260619C00180000
    expiry: str             # YYYYMMDD
    strike: float
    right: str              # "C" | "P"
    expiry_type: str        # "standard" | "weekly" | "leap"

    bid: float | None
    ask: float | None
    last: float | None
    mid: float | None

    volume: int | None
    open_interest: int | None
    implied_vol: float | None   # annualised, decimal

    delta: float | None
    gamma: float | None
    theta: float | None         # per-day time decay
    vega: float | None          # per 1-vol-point change
    rho: float | None


# ── Mock data generator ────────────────────────────────────────────────────────

def _seed(symbol: str) -> float:
    """Deterministic per-symbol seed in [0, 1) for consistent mock values."""
    return (sum(ord(c) * (i + 1) for i, c in enumerate(symbol)) % 10_000) / 10_000


def _mock_estimates(symbol: str) -> list[EarningsEstimate]:
    s    = _seed(symbol)
    base = 2.0 + s * 8.0   # base EPS $2–10

    today = date.today().isoformat()
    periods = [
        ("Q2 2026",  "quarterly", 0),
        ("Q3 2026",  "quarterly", 1),
        ("FY 2026",  "annual",    0),
        ("FY 2027",  "annual",    1),
    ]
    result = []
    for period, ptype, offset in periods:
        growth  = 1.0 + (s * 0.25 + offset * 0.05)  # growing estimates
        eps     = round(base * growth, 2)
        rev     = round((500 + s * 2000) * growth, 1)
        prev    = round(base * (1.0 + offset * 0.04), 2)
        result.append(EarningsEstimate(
            symbol=symbol, period=period, period_type=ptype,
            estimate_date=today,
            consensus_eps=eps,
            high_eps=round(eps * 1.08, 2), low_eps=round(eps * 0.92, 2),
            consensus_revenue=rev,
            high_revenue=round(rev * 1.05, 1), low_revenue=round(rev * 0.95, 1),
            num_analysts_eps=int(12 + s * 28),
            num_analysts_revenue=int(10 + s * 22),
            eps_growth_yoy=round((eps / prev - 1) * 100, 1),
        ))
    return result


def _mock_surprise_history(symbol: str, n: int = 8) -> list[EarningsSurprise]:
    s      = _seed(symbol)
    base   = 2.0 + s * 8.0
    today  = date.today()
    result = []

    for i in range(n):
        quarter     = (today.month // 3 - i) % 4 + 1
        year        = today.year - (i // 4)
        actual      = round(base * (1.0 + (s - 0.4) * 0.1 * (n - i) / n), 2)
        consensus   = round(actual * (0.95 + (1 - s) * 0.06), 2)
        surp_pct    = round((actual - consensus) / abs(consensus) * 100, 2) if consensus else None
        rev_actual  = round((500 + s * 2000) * (1 + 0.03 * (n - i) / n), 1)
        rev_cons    = round(rev_actual * 0.97, 1)
        result.append(EarningsSurprise(
            symbol=symbol,
            report_date=f"{year}-{quarter*3:02d}-15",
            fiscal_year=year,
            fiscal_quarter=quarter,
            period_label=f"Q{quarter} {year}",
            actual_eps=actual,
            consensus_eps=consensus,
            surprise_pct=surp_pct,
            actual_revenue=rev_actual,
            consensus_revenue=rev_cons,
            revenue_surprise_pct=round((rev_actual/rev_cons - 1)*100, 2),
            guidance_raised=(surp_pct or 0) > 3.0,
        ))
    return result


def _mock_fundamentals(symbol: str) -> CompanyFundamentals:
    s       = _seed(symbol)
    mktcap  = round(10 + s * 3000, 1)   # $B  — range $10B–$3T
    rev     = round(1000 + s * 400_000, 1)  # $M TTM
    margin  = round(0.08 + s * 0.32, 3)
    eps_g   = round((s - 0.3) * 40, 1)
    # Acceleration: two consecutive quarters of improving EPS growth
    accel   = s > 0.55

    return CompanyFundamentals(
        symbol=symbol,
        as_of=date.today().isoformat(),
        market_cap=mktcap,
        enterprise_value=round(mktcap * (1.05 + s * 0.4), 1),
        pe_ratio=round(15 + s * 35, 1),
        pe_forward=round(12 + s * 28, 1),
        peg_ratio=round(0.8 + s * 2.4, 2),
        ps_ratio=round(1 + s * 15, 1),
        pb_ratio=round(1.5 + s * 18, 1),
        revenue_ttm=rev,
        revenue_growth_yoy=round((s - 0.2) * 50, 1),
        gross_margin=round(0.30 + s * 0.55, 3) * 100,
        operating_margin=round(0.05 + s * 0.40, 3) * 100,
        net_margin=round(margin * 100, 2),
        ebitda_margin=round((margin + 0.05) * 100, 2),
        eps_ttm=round(2 + s * 15, 2),
        eps_growth_yoy=eps_g,
        eps_growth_3yr_cagr=round(eps_g * 0.85, 1),
        return_on_equity=round(8 + s * 55, 1),
        return_on_assets=round(3 + s * 22, 1),
        return_on_invested_capital=round(5 + s * 35, 1),
        free_cash_flow_ttm=round(rev * margin * 0.9, 1),
        free_cash_flow_yield=round(margin * 0.9 / (mktcap * 10) * 100, 2),
        fcf_growth_yoy=round((s - 0.25) * 45, 1),
        debt_to_equity=round(0.1 + s * 2.5, 2),
        net_debt_to_ebitda=round(-0.5 + s * 4.0, 2),
        current_ratio=round(0.8 + s * 3.0, 2),
        next_earnings_date=f"{date.today().year}-{(date.today().month + 2) % 12 + 1:02d}-15",
        earnings_acceleration=accel,
    )


def _mock_transcript(symbol: str, event_date: str) -> EarningsTranscript:
    s = _seed(symbol)
    segments = [
        TranscriptSegment("Operator", "Operator", "FactSet Conferencing",
            "Good morning, and welcome to the quarterly earnings call. "
            "I will now turn the call over to management."),
        TranscriptSegment("John Smith", "CEO", symbol,
            f"Thank you. We delivered strong results this quarter with revenue "
            f"up {round(10 + s*30, 1)}% year-over-year. Our earnings per share of "
            f"${round(2+s*8, 2)} exceeded consensus by {round(s*10, 1)}%. "
            "We continue to see strong demand across all business units "
            "and are raising guidance for the full year."),
        TranscriptSegment("Jane Doe", "CFO", symbol,
            f"Our gross margin expanded {round(s*200, 0):.0f} basis points to "
            f"{round(40+s*30, 1)}% driven by operating leverage and "
            "favorable product mix. Free cash flow generation remained robust "
            f"at ${round(500+s*2000, 0):.0f}M for the quarter."),
        TranscriptSegment("Mike Johnson", "Analyst", "Goldman Sachs",
            "Thank you for the strong results. Can you provide more color on "
            "the demand environment and any headwinds you see in the back half?"),
        TranscriptSegment("John Smith", "CEO", symbol,
            "Certainly. We see continued strength in enterprise and "
            "are not seeing material macro headwinds at this point. "
            "Our pipeline is healthy and we remain confident in our "
            "full-year outlook."),
    ]
    raw = "\n\n".join(f"[{seg.speaker} — {seg.role}]\n{seg.text}" for seg in segments)
    return EarningsTranscript(
        symbol=symbol,
        event_date=event_date,
        fiscal_period=f"Q{(date.today().month - 1) // 3 + 1} {date.today().year}",
        event_type="Earnings Call",
        duration_minutes=int(45 + s * 30),
        segments=segments,
        raw_text=raw,
        factset_event_id=f"FSET-{abs(hash(symbol + event_date)) % 1_000_000:06d}",
    )


def _mock_options_chain(symbol: str) -> list[FactSetOptionContract]:
    s     = _seed(symbol)
    spot  = 50 + s * 450   # $50–$500

    # Near-term expiry
    exp1  = f"{date.today().year}{(date.today().month + 1) % 12 + 1:02d}19"
    # 3-month expiry
    exp3  = f"{date.today().year}{(date.today().month + 3) % 12 + 1:02d}19"

    contracts: list[FactSetOptionContract] = []
    for expiry in [exp1, exp3]:
        for strike_offset in [-0.10, -0.05, 0.0, 0.05, 0.10, 0.15]:
            strike = round(spot * (1 + strike_offset), 0)
            moneyness = (spot - strike) / spot

            for right in ("C", "P"):
                iv   = round(0.20 + s * 0.35 + abs(moneyness) * 0.05, 4)
                d    = round((0.5 + moneyness * 3) if right == "C"
                             else (0.5 + moneyness * 3 - 1), 4)
                d    = max(-1.0, min(1.0, d))
                mid  = round(max(0.01, abs(moneyness) * spot * 0.8 + 0.5), 2)
                contracts.append(FactSetOptionContract(
                    underlying=symbol,
                    symbol=(
                        f"{symbol}{expiry}{'C' if right=='C' else 'P'}"
                        f"{int(strike*1000):08d}"
                    ),
                    expiry=expiry,
                    strike=strike,
                    right=right,
                    expiry_type="standard",
                    bid=round(mid * 0.97, 2),
                    ask=round(mid * 1.03, 2),
                    last=mid,
                    mid=mid,
                    volume=int(500 + s * 10_000),
                    open_interest=int(2000 + s * 50_000),
                    implied_vol=iv,
                    delta=d,
                    gamma=round(0.02 + s * 0.08, 4),
                    theta=round(-(0.03 + s * 0.15), 4),
                    vega=round(0.05 + s * 0.25, 4),
                    rho=round(d * 0.02, 4),
                ))
    return contracts


# ── FactSet Provider ───────────────────────────────────────────────────────────

class FactSetProvider:
    """
    FactSet REST API client implementing FactSet's full fundamentals,
    estimates, transcripts, and options surfaces.

    Operates in mock mode until FactSet credentials are provisioned.
    To activate live mode:
        1. Set env vars FACTSET_USERNAME and FACTSET_API_KEY
        2. Construct with use_mock=False

    Args:
        username:   FactSet serial number / username.  Default: FACTSET_USERNAME.
        api_key:    FactSet API key.  Default: FACTSET_API_KEY.
        use_mock:   Return realistic mock data instead of live API calls.
                    Defaults to True while credentials are pending.
        request_sleep: Seconds between API calls to respect rate limits.
    """

    def __init__(
        self,
        username: str | None = None,
        api_key: str | None = None,
        use_mock: bool = True,
        request_sleep: float = 0.5,
    ) -> None:
        self.username      = username or os.environ.get("FACTSET_USERNAME", "")
        self.api_key       = api_key  or os.environ.get("FACTSET_API_KEY",  "")
        self.use_mock      = use_mock
        self.request_sleep = request_sleep
        self._session      = requests.Session()
        self._session.headers.update({
            "Accept":       "application/json",
            "Content-Type": "application/json",
            "User-Agent":   "quantlab/1.0",
        })

        if not use_mock:
            if not self.username or not self.api_key:
                logger.warning(
                    "FactSet credentials not set — "
                    "set FACTSET_USERNAME and FACTSET_API_KEY or pass use_mock=True"
                )
            else:
                creds = base64.b64encode(
                    f"{self.username}:{self.api_key}".encode()
                ).decode()
                self._session.headers["Authorization"] = f"Basic {creds}"
                logger.info("FactSetProvider: live mode (user=%s)", self.username)
        else:
            logger.info("FactSetProvider: mock mode (credentials pending)")

    # ── Internal HTTP helper ───────────────────────────────────────────────────

    def _normalise_symbol(self, symbol: str) -> str:
        """Convert plain ticker to FactSet format: AAPL → AAPL-US."""
        if "-" in symbol:
            return symbol.upper()
        return f"{symbol.upper()}-{_DEFAULT_EXCH}"

    def _post(self, path: str, payload: dict) -> dict:
        """
        POST to FactSet API with retry logic.

        Retries up to _MAX_RETRIES times on 429 and 5xx responses.
        Raises requests.HTTPError on persistent failure.
        """
        url      = f"{_BASE_URL}{path}"
        last_exc = None

        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = self._session.post(url, json=payload, timeout=30)
                if resp.status_code == 429:
                    wait = 60 * (2 ** attempt)
                    logger.warning("FactSet rate limit (429) — waiting %ds", wait)
                    time.sleep(wait)
                    last_exc = requests.HTTPError(response=resp)
                    continue
                if resp.status_code >= 500:
                    time.sleep(self.request_sleep * (2 ** attempt))
                    last_exc = requests.HTTPError(response=resp)
                    continue
                resp.raise_for_status()
                time.sleep(self.request_sleep)
                return resp.json()
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    time.sleep(self.request_sleep * 2)

        raise last_exc or RuntimeError(f"Max retries exceeded for {path}")

    def _get(self, path: str, params: dict | None = None) -> dict:
        """GET with retry (used for transcript and reference endpoints)."""
        url      = f"{_BASE_URL}{path}"
        last_exc = None

        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = self._session.get(url, params=params or {}, timeout=30)
                if resp.status_code == 429:
                    wait = 60 * (2 ** attempt)
                    logger.warning("FactSet rate limit (429) — waiting %ds", wait)
                    time.sleep(wait)
                    last_exc = requests.HTTPError(response=resp)
                    continue
                resp.raise_for_status()
                time.sleep(self.request_sleep)
                return resp.json()
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    time.sleep(self.request_sleep * 2)

        raise last_exc or RuntimeError(f"Max retries exceeded for {path}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_earnings_estimates(self, symbol: str) -> list[EarningsEstimate]:
        """
        Fetch forward consensus EPS and revenue estimates per period.

        Live endpoint:
            POST /factset-estimates/v2/consensus-estimates
            Body: {
                "ids": ["AAPL-US"],
                "metrics": ["EPS", "SALES"],
                "periodicity": "QTR",
                "fiscalPeriodStart": "0Q",   // 0 quarters ago (current)
                "fiscalPeriodEnd":   "4Q",   // 4 quarters ahead
                "currency": "USD"
            }

        Returns estimate for current quarter, next two quarters, and two
        fiscal years.  Results are sorted nearest to furthest.

        Args:
            symbol: Ticker symbol (plain or FactSet format).

        Returns:
            List of EarningsEstimate, one per period.
        """
        if self.use_mock:
            return _mock_estimates(symbol)

        fset_id  = self._normalise_symbol(symbol)
        payload  = {
            "ids": [fset_id],
            "metrics": ["EPS", "SALES"],
            "periodicity": "QTR",
            "fiscalPeriodStart": "0Q",
            "fiscalPeriodEnd": "4Q",
            "currency": "USD",
            "includeWeeklyData": False,
        }
        data     = self._post("/factset-estimates/v2/consensus-estimates", payload)
        results  = data.get("data", [])
        today    = date.today().isoformat()

        estimates: list[EarningsEstimate] = []
        for item in results:
            period = item.get("fiscalPeriod", "")
            estimates.append(EarningsEstimate(
                symbol       = symbol,
                period       = period,
                period_type  = "quarterly" if item.get("periodicity") == "QTR" else "annual",
                estimate_date = today,
                consensus_eps     = item.get("mean"),
                high_eps          = item.get("high"),
                low_eps           = item.get("low"),
                consensus_revenue = item.get("salesMean"),
                high_revenue      = item.get("salesHigh"),
                low_revenue       = item.get("salesLow"),
                num_analysts_eps  = item.get("numEstimateEps",     0) or 0,
                num_analysts_revenue = item.get("numEstimateSales", 0) or 0,
                eps_growth_yoy    = item.get("epsGrowth"),
            ))
        return estimates

    def get_surprise_history(
        self,
        symbol: str,
        n_quarters: int = 8,
    ) -> list[EarningsSurprise]:
        """
        Fetch historical EPS and revenue surprise data.

        Live endpoint:
            POST /factset-estimates/v2/surprise
            Body: {
                "ids": ["AAPL-US"],
                "metrics": ["EPS", "SALES"],
                "periodicity": "QTR",
                "fiscalPeriodStart": "-8Q",   // 8 quarters ago
                "fiscalPeriodEnd":   "-1Q",   // most recent reported quarter
                "currency": "USD"
            }

        Returns quarters ordered oldest → most recent so callers can compute
        acceleration trends directly on the slice.

        Args:
            symbol:     Ticker symbol.
            n_quarters: Number of historical quarters to return (default 8).

        Returns:
            List of EarningsSurprise, oldest first.
        """
        if self.use_mock:
            raw = _mock_surprise_history(symbol, n_quarters)
            return sorted(raw, key=lambda s: s.report_date)

        fset_id = self._normalise_symbol(symbol)
        payload = {
            "ids": [fset_id],
            "metrics": ["EPS", "SALES"],
            "periodicity": "QTR",
            "fiscalPeriodStart": f"-{n_quarters}Q",
            "fiscalPeriodEnd": "-1Q",
            "currency": "USD",
        }
        data    = self._post("/factset-estimates/v2/surprise", payload)
        items   = data.get("data", [])

        surprises: list[EarningsSurprise] = []
        for item in sorted(items, key=lambda x: x.get("reportDate", "")):
            actual   = item.get("actualEps")
            cons     = item.get("meanEps")
            surp_pct = None
            if actual is not None and cons is not None and cons != 0:
                surp_pct = round((actual - cons) / abs(cons) * 100, 2)
            surprises.append(EarningsSurprise(
                symbol          = symbol,
                report_date     = item.get("reportDate", ""),
                fiscal_year     = item.get("fiscalYear", 0),
                fiscal_quarter  = item.get("fiscalQuarter", 0),
                period_label    = item.get("fiscalPeriod", ""),
                actual_eps      = actual,
                consensus_eps   = cons,
                surprise_pct    = surp_pct,
                actual_revenue  = item.get("actualSales"),
                consensus_revenue = item.get("meanSales"),
                revenue_surprise_pct = item.get("salesSurprisePct"),
                guidance_raised = item.get("guidanceRaised"),
            ))
        return surprises

    def get_fundamentals(self, symbol: str) -> CompanyFundamentals:
        """
        Fetch current fundamental ratios and growth metrics.

        Live endpoint:
            POST /factset-fundamentals/v2/company-reports
            Body: {
                "ids": ["AAPL-US"],
                "metrics": [
                    "FF_MKT_VAL", "FF_ENTERPRISE_VAL",
                    "FF_PE", "FF_PE_NTM", "FF_PEG",
                    "FF_PS", "FF_PBK",
                    "FF_SALES", "FF_SALES_CHG_PCT",
                    "FF_GROSS_MGN", "FF_OPER_MGN", "FF_NET_MGN", "FF_EBITDA_MGN",
                    "FF_EPS_BASIC_TTM", "FF_EPS_CHG_PCT",
                    "FF_ROE", "FF_ROA", "FF_ROIC",
                    "FF_FCF", "FF_FCF_YIELD", "FF_FCF_CHG_PCT",
                    "FF_DEBT_EQY", "FF_NET_DEBT_EBITDA", "FF_CURR_RATIO"
                ],
                "periodicity": "LFY",
                "currency": "USD"
            }

        Args:
            symbol: Ticker symbol.

        Returns:
            CompanyFundamentals snapshot as of today's date.
        """
        if self.use_mock:
            return _mock_fundamentals(symbol)

        fset_id = self._normalise_symbol(symbol)
        payload = {
            "ids": [fset_id],
            "metrics": [
                "FF_MKT_VAL", "FF_ENTERPRISE_VAL",
                "FF_PE", "FF_PE_NTM", "FF_PEG",
                "FF_PS", "FF_PBK",
                "FF_SALES", "FF_SALES_CHG_PCT",
                "FF_GROSS_MGN", "FF_OPER_MGN", "FF_NET_MGN", "FF_EBITDA_MGN",
                "FF_EPS_BASIC_TTM", "FF_EPS_CHG_PCT", "FF_EPS_3YR_CAGR",
                "FF_ROE", "FF_ROA", "FF_ROIC",
                "FF_FCF", "FF_FCF_YIELD", "FF_FCF_CHG_PCT",
                "FF_DEBT_EQY", "FF_NET_DEBT_EBITDA", "FF_CURR_RATIO",
                "FF_NEXT_REPORT_DATE",
            ],
            "periodicity": "LFY",
            "currency": "USD",
        }
        data  = self._post("/factset-fundamentals/v2/company-reports", payload)
        items = data.get("data", [])
        if not items:
            return _mock_fundamentals(symbol)   # fallback to mock when no data

        m = items[0]
        eps_g  = m.get("FF_EPS_CHG_PCT")
        eps_g2 = m.get("FF_EPS_3YR_CAGR")
        accel  = (eps_g or 0) > 15.0 and (eps_g2 or 0) > 10.0

        return CompanyFundamentals(
            symbol=symbol, as_of=date.today().isoformat(),
            market_cap=m.get("FF_MKT_VAL"),
            enterprise_value=m.get("FF_ENTERPRISE_VAL"),
            pe_ratio=m.get("FF_PE"), pe_forward=m.get("FF_PE_NTM"),
            peg_ratio=m.get("FF_PEG"), ps_ratio=m.get("FF_PS"), pb_ratio=m.get("FF_PBK"),
            revenue_ttm=m.get("FF_SALES"), revenue_growth_yoy=m.get("FF_SALES_CHG_PCT"),
            gross_margin=m.get("FF_GROSS_MGN"), operating_margin=m.get("FF_OPER_MGN"),
            net_margin=m.get("FF_NET_MGN"), ebitda_margin=m.get("FF_EBITDA_MGN"),
            eps_ttm=m.get("FF_EPS_BASIC_TTM"), eps_growth_yoy=eps_g, eps_growth_3yr_cagr=eps_g2,
            return_on_equity=m.get("FF_ROE"), return_on_assets=m.get("FF_ROA"),
            return_on_invested_capital=m.get("FF_ROIC"),
            free_cash_flow_ttm=m.get("FF_FCF"), free_cash_flow_yield=m.get("FF_FCF_YIELD"),
            fcf_growth_yoy=m.get("FF_FCF_CHG_PCT"),
            debt_to_equity=m.get("FF_DEBT_EQY"), net_debt_to_ebitda=m.get("FF_NET_DEBT_EBITDA"),
            current_ratio=m.get("FF_CURR_RATIO"),
            next_earnings_date=m.get("FF_NEXT_REPORT_DATE"),
            earnings_acceleration=accel,
        )

    def get_transcript(
        self,
        symbol: str,
        event_date: str | None = None,
    ) -> EarningsTranscript:
        """
        Fetch the most recent (or a specific date's) earnings call transcript.

        Live endpoint:
            GET  /events-and-transcripts/v1/transcripts
                 ?ids=AAPL-US&eventDateTimeStart=2026-01-01T00:00:00Z
                 &eventTypes=Earnings%20Call&paginationLimit=1
            Then:
            GET  /events-and-transcripts/v1/transcripts/{transcriptId}

        Transcripts are returned as an array of speaker segments with metadata
        (speaker name, role, company affiliation).  The raw_text field is
        populated by concatenating all segments for downstream NLP/LLM use.

        Args:
            symbol:     Ticker symbol.
            event_date: ISO date of the call (e.g. "2026-01-29").
                        When None, returns the most recent transcript.

        Returns:
            EarningsTranscript with all speaker segments populated.
        """
        if self.use_mock:
            return _mock_transcript(symbol, event_date or date.today().isoformat())

        fset_id = self._normalise_symbol(symbol)
        params: dict[str, Any] = {
            "ids": fset_id,
            "eventTypes": "Earnings Call",
            "paginationLimit": 1,
            "paginationOffset": 0,
        }
        if event_date:
            params["eventDateTimeStart"] = f"{event_date}T00:00:00Z"
            params["eventDateTimeEnd"]   = f"{event_date}T23:59:59Z"

        index = self._get("/events-and-transcripts/v1/transcripts", params)
        items = index.get("data", [])
        if not items:
            return _mock_transcript(symbol, event_date or date.today().isoformat())

        tid     = items[0].get("transcriptId", "")
        detail  = self._get(f"/events-and-transcripts/v1/transcripts/{tid}")
        raw_seg = detail.get("data", {}).get("bodyWithSpeakerIds", [])

        segments: list[TranscriptSegment] = []
        for seg in raw_seg:
            speaker_meta = seg.get("speakerInfo", {})
            segments.append(TranscriptSegment(
                speaker = speaker_meta.get("name", "Unknown"),
                role    = speaker_meta.get("title", ""),
                firm    = speaker_meta.get("company", ""),
                text    = seg.get("text", ""),
            ))

        raw_text = "\n\n".join(
            f"[{s.speaker} — {s.role}]\n{s.text}" for s in segments
        )
        evt = items[0]
        return EarningsTranscript(
            symbol=symbol,
            event_date=evt.get("eventDateTime", "")[:10],
            fiscal_period=evt.get("fiscalPeriod", ""),
            event_type=evt.get("eventType", "Earnings Call"),
            duration_minutes=evt.get("durationMinutes", 0) or 0,
            segments=segments,
            raw_text=raw_text,
            factset_event_id=tid,
        )

    def get_options_chain(self, symbol: str) -> list[FactSetOptionContract]:
        """
        Fetch the full options chain for a symbol.

        Live endpoint:
            POST /factset-options/v1/chains
            Body: {
                "ids": ["AAPL-US"],
                "expirationDateStart": "{today}",
                "expirationDateEnd":   "{today + 90 days}",
                "strikeRange": 0.15,       // ±15% from spot
                "includeGreeks": true,
                "currency": "USD"
            }

        Greeks are Black-Scholes model values computed by FactSet.
        open_interest is the prior-day OI snapshot.

        Args:
            symbol: Ticker symbol.

        Returns:
            List of FactSetOptionContract sorted by expiry then strike.
        """
        if self.use_mock:
            return _mock_options_chain(symbol)

        fset_id   = self._normalise_symbol(symbol)
        today_str = date.today().isoformat()
        from datetime import timedelta
        end_str   = (date.today() + timedelta(days=90)).isoformat()

        payload = {
            "ids": [fset_id],
            "expirationDateStart": today_str,
            "expirationDateEnd":   end_str,
            "strikeRange": 0.15,
            "includeGreeks": True,
            "currency": "USD",
        }
        data  = self._post("/factset-options/v1/chains", payload)
        items = data.get("data", [])

        contracts: list[FactSetOptionContract] = []
        for item in items:
            mid = None
            bid = item.get("bid")
            ask = item.get("ask")
            if bid and ask:
                mid = round((bid + ask) / 2, 2)
            contracts.append(FactSetOptionContract(
                underlying    = symbol,
                symbol        = item.get("optionId", ""),
                expiry        = (item.get("expirationDate") or "").replace("-", ""),
                strike        = item.get("strike", 0),
                right         = "C" if item.get("callPut") == "Call" else "P",
                expiry_type   = item.get("expiryType", "standard"),
                bid           = bid,
                ask           = ask,
                last          = item.get("last"),
                mid           = mid,
                volume        = item.get("volume"),
                open_interest = item.get("openInterest"),
                implied_vol   = item.get("impliedVolatility"),
                delta         = item.get("delta"),
                gamma         = item.get("gamma"),
                theta         = item.get("theta"),
                vega          = item.get("vega"),
                rho           = item.get("rho"),
            ))
        return sorted(contracts, key=lambda c: (c.expiry, c.right, c.strike))
