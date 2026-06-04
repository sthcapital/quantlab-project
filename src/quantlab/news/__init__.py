"""
Layer 2: News pipeline.

Fetch → clean → classify → extract sentiment scores → rolling feature counts.

IBKR headline format:
    {A:<conid>:L:<lang>:K:<relevance>:C:<confidence>}!Headline text here

K: field = keyword/relevance score (0.0–1.0, or n/a)
C: field = confidence score (0.0–1.0)

Both scores are extracted as numeric features. The chat noted these were
being discarded — this module captures them properly.
"""

from __future__ import annotations

import html
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Sequence

METADATA_RE = re.compile(r"\{.*?\}!?")
K_SCORE_RE = re.compile(r":K:([\d.]+)")
C_SCORE_RE = re.compile(r":C:([\d.]+)")


@dataclass
class NewsItem:
    """A single cleaned and classified news headline."""

    time: datetime
    date: str               # YYYY-MM-DD
    provider: str
    article_id: str
    category: str
    headline: str
    k_score: float | None   # IBKR relevance score (0–1)
    c_score: float | None   # IBKR confidence score (0–1)


def _extract_score(pattern: re.Pattern, raw: str) -> float | None:
    m = pattern.search(raw)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def clean_headline(raw: str) -> str:
    """Strip IBKR metadata prefix and decode HTML entities."""
    return html.unescape(METADATA_RE.sub("", raw).strip())


def classify_headline(headline: str) -> str:
    """
    Classify a cleaned headline into one of five categories:
        upgrade | downgrade | earnings | management | analyst_action | other
    """
    low = headline.lower()

    if any(k in low for k in ["upgrade", "upgraded", "raises target", "raised target", "price target increase"]):
        return "upgrade"

    if any(k in low for k in ["downgrade", "downgraded", "lowers target", "lowered target", "price target cut"]):
        return "downgrade"

    if any(k in low for k in ["ceo", "cfo", "coo", "chief executive", "president", "board member", "appoints", "resigns", "steps down", "new cto", "new coo"]):
        return "management"

    if any(k in low for k in ["earnings", "q1 ", "q2 ", "q3 ", "q4 ", "guide", "guidance", "revenue", "sales", " eps ", "beats estimates", "misses estimates", "quarterly results"]):
        return "earnings"

    if any(k in low for k in [
        "reiterated", "initiated", "resumed", "overweight", "underweight",
        "outperform", "underperform", "market perform", "neutral", "buy", "sell",
        "hold", "equal weight", "sector weight",
    ]):
        return "analyst_action"

    return "other"


def parse_news_item(item) -> NewsItem:
    """Parse a raw ib_insync NewsArticle into a NewsItem."""
    raw = item.headline
    k_score = _extract_score(K_SCORE_RE, raw)
    c_score = _extract_score(C_SCORE_RE, raw)
    cleaned = clean_headline(raw)
    return NewsItem(
        time=item.time,
        date=str(item.time)[:10],
        provider=item.providerCode,
        article_id=item.articleId,
        category=classify_headline(cleaned),
        headline=cleaned,
        k_score=k_score,
        c_score=c_score,
    )


def fetch_news(
    ib,
    contract,
    days: int = 120,
    limit: int = 100,
    start_dt: datetime | None = None,
    end_dt: datetime | None = None,
) -> list[NewsItem]:
    """
    Fetch and parse historical IBKR news headlines for a qualified contract.

    Requires a live ib_insync IB() connection.

    Args:
        ib:       Connected IB() instance.
        contract: Qualified IBKR contract (from qualifyContracts).
        days:     Calendar days back from now when start_dt is not provided.
        limit:    Max headlines to return.
        start_dt: Explicit start datetime (overrides days when provided).
        end_dt:   Explicit end datetime (overrides now when provided).

    Returns:
        List of NewsItem sorted by date descending (newest first).
    """
    if end_dt is None:
        end_dt = datetime.utcnow()
    if start_dt is None:
        start_dt = end_dt - timedelta(days=days)

    raw_headlines = ib.reqHistoricalNews(
        conId=contract.conId,
        providerCodes="BRFG+BRFUPDN+DJNL",
        startDateTime=start_dt.strftime("%Y%m%d %H:%M:%S"),
        endDateTime=end_dt.strftime("%Y%m%d %H:%M:%S"),
        totalResults=limit,
        historicalNewsOptions=[],
    )

    items = [parse_news_item(h) for h in raw_headlines]
    items.sort(key=lambda x: x.time, reverse=True)
    return items


@dataclass
class NewsFeatures:
    """Rolling news feature counts for a single signal date."""

    trade_date: str
    lookback_days: int
    total_count: int
    upgrade_count: int
    downgrade_count: int
    earnings_count: int
    management_count: int
    analyst_action_count: int
    other_count: int
    avg_k_score: float | None
    avg_c_score: float | None
    dominant_category: str

    def has_news(self) -> bool:
        return self.total_count > 0

    def bullish_score(self) -> float:
        """Simple bullish signal: upgrades + positive earnings headlines."""
        return float(self.upgrade_count + self.earnings_count)

    def bearish_score(self) -> float:
        return float(self.downgrade_count)


def compute_news_features(
    items: Sequence[NewsItem],
    trade_date: str,
    lookback_days: int = 7,
) -> NewsFeatures:
    """
    Compute rolling news feature counts for a given trade date.

    Selects all NewsItems within [trade_date - lookback_days, trade_date]
    and aggregates category counts and average sentiment scores.
    """
    end_dt = datetime.strptime(trade_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=lookback_days)

    window = [
        item for item in items
        if start_dt <= datetime.strptime(item.date, "%Y-%m-%d") <= end_dt
    ]

    counts = Counter(item.category for item in window)

    k_scores = [item.k_score for item in window if item.k_score is not None]
    c_scores = [item.c_score for item in window if item.c_score is not None]

    if not window:
        dominant = "none"
    else:
        dominant = max(counts, key=counts.get)

    return NewsFeatures(
        trade_date=trade_date,
        lookback_days=lookback_days,
        total_count=len(window),
        upgrade_count=counts.get("upgrade", 0),
        downgrade_count=counts.get("downgrade", 0),
        earnings_count=counts.get("earnings", 0),
        management_count=counts.get("management", 0),
        analyst_action_count=counts.get("analyst_action", 0),
        other_count=counts.get("other", 0),
        avg_k_score=sum(k_scores) / len(k_scores) if k_scores else None,
        avg_c_score=sum(c_scores) / len(c_scores) if c_scores else None,
        dominant_category=dominant,
    )
