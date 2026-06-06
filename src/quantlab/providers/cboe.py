"""CBOE VIX history — no API key required."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import date, datetime

import requests

_VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
_HEADERS = {"User-Agent": "QuantLab Research quantlab@sthcapital.com"}


@dataclass(frozen=True)
class VixBar:
    date: date
    open: float
    high: float
    low: float
    close: float


def fetch_vix_history(start: date, end: date) -> list[VixBar]:
    """
    Download VIX daily OHLC history from CBOE CDN.

    Args:
        start: Start date (inclusive).
        end:   End date (inclusive).

    Returns:
        List of VixBar sorted ascending by date.
    """
    resp = requests.get(_VIX_URL, headers=_HEADERS, timeout=30)
    resp.raise_for_status()

    bars: list[VixBar] = []
    reader = csv.DictReader(io.StringIO(resp.text))
    for row in reader:
        try:
            dt = datetime.strptime(row["DATE"].strip(), "%m/%d/%Y").date()
        except (KeyError, ValueError):
            continue
        if dt < start or dt > end:
            continue
        try:
            bars.append(VixBar(
                date=dt,
                open=float(row["OPEN"]),
                high=float(row["HIGH"]),
                low=float(row["LOW"]),
                close=float(row["CLOSE"]),
            ))
        except (KeyError, ValueError):
            continue

    bars.sort(key=lambda b: b.date)
    return bars


def classify_vix_regime(vix_close: float) -> tuple[str, int]:
    """
    Classify VIX into a market fear regime.

    Thresholds:
        <15  → low      (score 0): calm, risk-on
        15–25 → elevated (score 1): moderate uncertainty
        25–35 → high     (score 2): significant stress
        ≥35  → extreme  (score 3): crisis / panic

    Returns:
        (regime_label, score) where score is 0 (low) to 3 (extreme).
    """
    if vix_close < 15.0:
        return ("low", 0)
    if vix_close < 25.0:
        return ("elevated", 1)
    if vix_close < 35.0:
        return ("high", 2)
    return ("extreme", 3)
