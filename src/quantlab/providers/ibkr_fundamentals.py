"""
quantlab.providers.ibkr_fundamentals — IBKR fundamental data via ib_fundamental.

Wraps the ib_fundamental library to extract earnings surprise history,
revenue/EPS growth rates, beat streaks, and next earnings dates directly
from IBKR's Reuters Fundamentals feed.

IBKR subscription required:
    Reuters Fundamentals (Research Subscriptions → Reuters Fundamentals)
    or Wall Street Horizons Corporate Event Estimates.
    Without this subscription, all calls raise IbkrFundamentalsUnavailable
    (error 10358 from TWS: "Fundamentals data is not allowed").

Once the subscription is activated, set use_mock=False and the provider
connects to the live TWS at the given host/port.

Data sources (IBKR report types):
    ReportsFinSummary — quarterly EPS actuals, quarterly revenue
    RESC              — forward year EPS/revenue estimates and actuals
    ReportSnapshot    — analyst forecasts, ratios

Usage::

    from ib_async import IB
    from quantlab.providers.ibkr_fundamentals import IbkrFundamentalsProvider

    ib = IB()
    ib.connect("172.23.208.1", 7497, clientId=30)

    provider = IbkrFundamentalsProvider(ib=ib, use_mock=False)
    profile = provider.get_earnings_profile("AAPL")
    print(profile.consecutive_beats, profile.eps_growth_yoy)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

logger = logging.getLogger(__name__)


# ── Exception hierarchy ────────────────────────────────────────────────────────

class IbkrFundamentalsUnavailable(RuntimeError):
    """
    Raised when IBKR returns error 10358 ("Fundamentals data is not allowed").

    Activate a Reuters Fundamentals subscription in TWS:
        Account Management → Research Subscriptions → Reuters Fundamentals
    """


# ── Response dataclass ─────────────────────────────────────────────────────────

@dataclass
class EarningsSurpriseRecord:
    """One quarter's actual EPS vs the consensus estimate."""
    period_label: str           # e.g. "Q1 2026"
    report_date: str            # YYYY-MM-DD (as_of_date from IBKR)
    actual_eps: float | None
    estimate_eps: float | None  # from RESC fy_actuals consensus
    surprise_pct: float | None  # (actual - estimate) / abs(estimate) × 100
    beat: bool                  # True when actual > estimate


@dataclass
class FundamentalEarningsProfile:
    """
    Fundamental earnings metrics extracted from IBKR Reuters data.

    This replaces the OHLCV-inferred EarningsProfile when the Reuters
    Fundamentals subscription is active.
    """
    symbol: str
    as_of: str                           # YYYY-MM-DD of this snapshot

    # Surprise history (last 4–8 quarters, oldest first)
    surprise_history: list[EarningsSurpriseRecord] = field(default_factory=list)

    # Computed from surprise history
    consecutive_beats: int = 0           # consecutive quarters beating consensus
    positive_surprise_rate: float = 0.0  # fraction of beats in available history

    # Growth metrics (YoY, trailing four quarters)
    eps_growth_yoy: float | None = None      # % change vs year-ago quarter
    revenue_growth_yoy: float | None = None  # % change vs year-ago quarter

    # Acceleration (are recent EPS surprises larger than older ones?)
    acceleration_trend: float = 0.0      # (recent_half_avg - early_half_avg) / early
    earnings_acceleration: bool = False   # True when trend > 0.10 and beats >= 2

    # Calendar
    next_earnings_date: str | None = None

    # Data provenance
    source: str = "ibkr_fundamentals"
    n_quarters: int = 0                  # number of quarters available

    def to_ohlcv_profile(self):
        """
        Convert to a quantlab.signals.earnings.EarningsProfile so the existing
        earnings_acceleration_score() function works without changes.
        """
        from quantlab.signals.earnings import EarningsProfile
        n = len(self.surprise_history)
        last_4 = [abs(r.surprise_pct or 0) for r in self.surprise_history[-4:]]
        prior_4 = [abs(r.surprise_pct or 0) for r in self.surprise_history[-8:-4]]
        return EarningsProfile(
            symbol=self.symbol,
            earnings_dates=[r.report_date for r in self.surprise_history],
            earnings_count=n,
            earnings_frequency=4.0 if n >= 4 else float(n),
            avg_post_earnings_return=sum(last_4) / len(last_4) / 100 if last_4 else 0.0,
            avg_signed_return=0.0,  # not available from fundamental data
            positive_surprise_rate=self.positive_surprise_rate,
            acceleration_trend=self.acceleration_trend,
            last_4_avg=sum(last_4)  / len(last_4)  if last_4  else 0.0,
            prior_4_avg=sum(prior_4) / len(prior_4) if prior_4 else 0.0,
        )


# ── Mock data ─────────────────────────────────────────────────────────────────

def _seed(symbol: str) -> float:
    return (sum(ord(c) * (i + 1) for i, c in enumerate(symbol)) % 10_000) / 10_000


def _mock_profile(symbol: str) -> FundamentalEarningsProfile:
    """Realistic mock profile seeded by symbol (deterministic)."""
    from datetime import timedelta
    s = _seed(symbol)

    today = date.today()
    history: list[EarningsSurpriseRecord] = []

    # Build 8 quarters of history, oldest first
    base_eps = 1.50 + s * 8.0
    for i in range(8):
        quarter_offset = 8 - i      # 8 = oldest, 1 = most recent
        q_date = today - timedelta(days=quarter_offset * 91)
        quarter = ((q_date.month - 1) // 3) + 1
        year    = q_date.year

        actual    = round(base_eps * (1.0 + (s - 0.3) * 0.05 * (9 - quarter_offset)), 2)
        estimate  = round(actual * (0.95 + (1 - s) * 0.04), 2)
        surp_pct  = round((actual - estimate) / abs(estimate) * 100, 2) if estimate else None
        beat      = (surp_pct or 0) > 0

        history.append(EarningsSurpriseRecord(
            period_label = f"Q{quarter} {year}",
            report_date  = q_date.isoformat(),
            actual_eps   = actual,
            estimate_eps = estimate,
            surprise_pct = surp_pct,
            beat         = beat,
        ))

    beats   = sum(1 for r in history if r.beat)
    pos_rate = beats / len(history) if history else 0.0

    # Consecutive beats from most recent backwards
    consec = 0
    for r in reversed(history):
        if r.beat:
            consec += 1
        else:
            break

    # Acceleration: compare magnitude of last 4 vs prior 4
    recent_surps = [abs(r.surprise_pct or 0) for r in history[-4:]]
    prior_surps  = [abs(r.surprise_pct or 0) for r in history[:4]]
    recent_avg = sum(recent_surps) / len(recent_surps) if recent_surps else 0
    prior_avg  = sum(prior_surps)  / len(prior_surps)  if prior_surps  else 0.001
    trend = round((recent_avg - prior_avg) / prior_avg, 4)

    next_ed = (today + timedelta(days=int(30 + s * 60))).isoformat()

    return FundamentalEarningsProfile(
        symbol               = symbol,
        as_of                = today.isoformat(),
        surprise_history     = history,
        consecutive_beats    = consec,
        positive_surprise_rate = pos_rate,
        eps_growth_yoy       = round((s - 0.3) * 40, 1),
        revenue_growth_yoy   = round((s - 0.25) * 30, 1),
        acceleration_trend   = trend,
        earnings_acceleration = trend > 0.10 and consec >= 2,
        next_earnings_date   = next_ed,
        source               = "mock",
        n_quarters           = len(history),
    )


# ── Provider ───────────────────────────────────────────────────────────────────

class IbkrFundamentalsProvider:
    """
    IBKR fundamental data provider using ib_fundamental.

    Args:
        ib:        Connected IB() instance (ib_async).  Ignored in mock mode.
        use_mock:  Return deterministic mock data.  Defaults to True until
                   the Reuters Fundamentals subscription is activated.
        n_quarters: Number of historical quarters to retrieve (default 8).
    """

    def __init__(
        self,
        ib=None,
        use_mock: bool = True,
        n_quarters: int = 8,
    ) -> None:
        self._ib        = ib
        self.use_mock   = use_mock
        self.n_quarters = n_quarters

        if not use_mock and ib is None:
            raise ValueError("ib (IB() instance) required when use_mock=False")

        if not use_mock:
            try:
                from ib_fundamental.fundamental import FundamentalData  # noqa: F401
            except ImportError:
                raise ImportError(
                    "ib_fundamental not installed. Run: pip install ib_fundamental"
                )

        mode = "mock" if use_mock else "live IBKR"
        logger.info("IbkrFundamentalsProvider: %s mode", mode)

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_earnings_profile(self, symbol: str) -> FundamentalEarningsProfile:
        """
        Fetch fundamental earnings profile for a symbol.

        Returns:
            FundamentalEarningsProfile with surprise history, beat streak,
            EPS/revenue growth, and next earnings date.

        Raises:
            IbkrFundamentalsUnavailable: when IBKR returns error 10358.
        """
        if self.use_mock:
            return _mock_profile(symbol)
        return self._fetch_live(symbol)

    # ── Live fetcher ───────────────────────────────────────────────────────────

    def _fetch_live(self, symbol: str) -> FundamentalEarningsProfile:
        """Fetch from IBKR via ib_fundamental.  Raises on subscription errors."""
        from ib_fundamental.fundamental import FundamentalData

        try:
            fd = FundamentalData(ib=self._ib, symbol=symbol)
            return self._parse(symbol, fd)
        except Exception as exc:
            msg = str(exc)
            if "10358" in msg or "not allowed" in msg.lower():
                raise IbkrFundamentalsUnavailable(
                    f"IBKR Fundamentals subscription required for {symbol}. "
                    "Activate 'Reuters Fundamentals' in TWS → Account Management "
                    "→ Research Subscriptions.  Error from IBKR: " + msg
                ) from exc
            raise

    def _parse(self, symbol: str, fd) -> FundamentalEarningsProfile:
        """Extract metrics from a live FundamentalData object."""
        today = date.today().isoformat()

        # ── EPS actuals (ReportsFinSummary) ────────────────────────────────────
        eps_actuals: list = []
        try:
            eps_actuals = fd.eps_q or []
        except Exception as exc:
            logger.debug("%s: eps_q failed: %s", symbol, exc)

        # ── Revenue actuals ────────────────────────────────────────────────────
        rev_actuals: list = []
        try:
            rev_actuals = fd.revenue_q or []
        except Exception as exc:
            logger.debug("%s: revenue_q failed: %s", symbol, exc)

        # ── Forward year estimates vs actuals (RESC) ───────────────────────────
        fy_actuals: list = []
        fy_estimates: list = []
        try:
            fy_actuals   = fd.fy_actuals   or []
            fy_estimates = fd.fy_estimates  or []
        except Exception as exc:
            logger.debug("%s: fy_actuals/fy_estimates failed: %s", symbol, exc)

        # ── Build surprise history from RESC ───────────────────────────────────
        # fy_actuals has type="Actual", item="EPS", period_type="Q"
        # fy_estimates has type="Estimate", item="EPS", est_type="Mean", period_type="Q"
        eps_act_q = sorted(
            [a for a in fy_actuals if a.item == "EPS" and a.period_type == "Q"],
            key=lambda x: (x.fyear, x.end_month),
        )
        eps_est_q = {
            (e.fyear, e.end_month): e.value
            for e in fy_estimates
            if e.item == "EPS" and e.period_type == "Q" and e.est_type == "Mean"
        }

        history: list[EarningsSurpriseRecord] = []
        for act in eps_act_q[-self.n_quarters:]:
            key = (act.fyear, act.end_month)
            est  = eps_est_q.get(key)
            surp = None
            if est is not None and est != 0:
                surp = round((act.value - est) / abs(est) * 100, 2)
            beat = (surp or 0) > 0
            history.append(EarningsSurpriseRecord(
                period_label = f"Q{(act.end_month // 3)} {act.fyear}",
                report_date  = (act.updated.date().isoformat()
                                if act.updated else f"{act.fyear}-{act.end_month:02d}-01"),
                actual_eps   = act.value,
                estimate_eps = est,
                surprise_pct = surp,
                beat         = beat,
            ))

        # Fallback: use ReportsFinSummary eps_q when RESC unavailable
        if not history and eps_actuals:
            sorted_eps = sorted(eps_actuals, key=lambda x: x.as_of_date)
            for ep in sorted_eps[-self.n_quarters:]:
                history.append(EarningsSurpriseRecord(
                    period_label = ep.period,
                    report_date  = ep.as_of_date.date().isoformat(),
                    actual_eps   = ep.eps,
                    estimate_eps = None,
                    surprise_pct = None,
                    beat         = False,
                ))

        # ── Compute derived metrics ────────────────────────────────────────────
        beats     = sum(1 for r in history if r.beat)
        pos_rate  = beats / len(history) if history else 0.0

        consec = 0
        for r in reversed(history):
            if r.beat:
                consec += 1
            else:
                break

        # Acceleration: recent 4 vs prior 4 by absolute surprise %
        recent_surps = [abs(r.surprise_pct or 0) for r in history[-4:] if r.surprise_pct]
        prior_surps  = [abs(r.surprise_pct or 0) for r in history[-8:-4] if r.surprise_pct]
        recent_avg   = sum(recent_surps) / len(recent_surps) if recent_surps else 0.0
        prior_avg    = sum(prior_surps)  / len(prior_surps)  if prior_surps  else None
        trend = (
            round((recent_avg - prior_avg) / prior_avg, 4)
            if prior_avg and prior_avg > 0
            else 0.0
        )

        # ── YoY EPS growth from quarterly actuals ─────────────────────────────
        eps_growth = None
        if len(history) >= 5:
            curr = history[-1].actual_eps
            year_ago = history[-5].actual_eps
            if curr and year_ago and year_ago != 0:
                eps_growth = round((curr / year_ago - 1) * 100, 1)

        # ── YoY revenue growth ─────────────────────────────────────────────────
        rev_growth = None
        if len(rev_actuals) >= 5:
            sorted_rev = sorted(rev_actuals, key=lambda x: x.as_of_date)
            curr_r    = sorted_rev[-1].revenue
            year_r    = sorted_rev[-5].revenue
            if curr_r and year_r and year_r != 0:
                rev_growth = round((curr_r / year_r - 1) * 100, 1)

        # ── Next earnings date from RESC fy_estimates ─────────────────────────
        next_ed = None
        future_qs = sorted(
            [e for e in fy_estimates if e.item == "EPS" and e.period_type == "Q"
             and e.type == "Estimate"],
            key=lambda x: (x.fyear, x.end_month),
        )
        if future_qs:
            nxt = future_qs[0]
            next_ed = f"{nxt.end_cal_year}-{nxt.end_month:02d}-01"

        return FundamentalEarningsProfile(
            symbol               = symbol,
            as_of                = today,
            surprise_history     = history,
            consecutive_beats    = consec,
            positive_surprise_rate = pos_rate,
            eps_growth_yoy       = eps_growth,
            revenue_growth_yoy   = rev_growth,
            acceleration_trend   = trend,
            earnings_acceleration = trend > 0.10 and consec >= 2,
            next_earnings_date   = next_ed,
            source               = "ibkr_fundamentals",
            n_quarters           = len(history),
        )

    # ── Convenience wrappers ───────────────────────────────────────────────────

    def get_eps_surprise_history(self, symbol: str) -> list[EarningsSurpriseRecord]:
        """Return just the surprise history list."""
        return self.get_earnings_profile(symbol).surprise_history

    def get_consecutive_beats(self, symbol: str) -> int:
        """Return the number of consecutive quarters beating consensus."""
        return self.get_earnings_profile(symbol).consecutive_beats

    def get_next_earnings_date(self, symbol: str) -> str | None:
        """Return next expected earnings date as YYYY-MM-DD string or None."""
        return self.get_earnings_profile(symbol).next_earnings_date
