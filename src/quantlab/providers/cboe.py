"""CBOE VIX and put/call ratio history — no API key required."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import date, datetime

import requests

_CDN = "https://cdn.cboe.com/api/global/us_indices/daily_prices"
_VIX_URL        = f"{_CDN}/VIX_History.csv"
_TOTAL_PCR_URL  = f"{_CDN}/PC_History.csv"
_EQUITY_PCR_URL = f"{_CDN}/EQUITY_PC_History.csv"
_INDEX_PCR_URL  = f"{_CDN}/INDEX_PC_History.csv"
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


@dataclass(frozen=True)
class PcrBar:
    """A single daily put/call ratio reading from CBOE."""

    date: date
    close: float   # daily PCR (put volume / call volume)


def _fetch_pcr(url: str, start: date, end: date) -> list[PcrBar]:
    """
    Generic helper: download a CBOE PCR CSV and return PcrBars in [start, end].

    CBOE PCR files use the column layout:
        DATE, CALL, PUT, TOTAL
    where TOTAL is the put/call ratio.  Falls back to CLOSE if TOTAL is absent
    (some index PCR files use a different column name).
    """
    resp = requests.get(url, headers=_HEADERS, timeout=30)
    resp.raise_for_status()

    bars: list[PcrBar] = []
    reader = csv.DictReader(io.StringIO(resp.text))
    for row in reader:
        try:
            dt = datetime.strptime(row["DATE"].strip(), "%m/%d/%Y").date()
        except (KeyError, ValueError):
            continue
        if dt < start or dt > end:
            continue
        # CBOE PCR files: TOTAL = put/call ratio; fall back to CLOSE
        raw = row.get("TOTAL") or row.get("CLOSE") or row.get("P/C Ratio", "")
        try:
            bars.append(PcrBar(date=dt, close=float(raw)))
        except (ValueError, TypeError):
            continue

    bars.sort(key=lambda b: b.date)
    return bars


def fetch_total_pcr(start: date, end: date) -> list[PcrBar]:
    """Download CBOE total (equity + index) put/call ratio history."""
    return _fetch_pcr(_TOTAL_PCR_URL, start, end)


def fetch_equity_pcr(start: date, end: date) -> list[PcrBar]:
    """Download CBOE equity-only put/call ratio history."""
    return _fetch_pcr(_EQUITY_PCR_URL, start, end)


def fetch_index_pcr(start: date, end: date) -> list[PcrBar]:
    """Download CBOE index-only put/call ratio history."""
    return _fetch_pcr(_INDEX_PCR_URL, start, end)


def classify_pcr_regime(pcr: float) -> tuple[str, int]:
    """
    Classify equity PCR into a sentiment regime.

    Thresholds (equity PCR):
        > 1.00 → extreme_fear       (score -2): contrarian bullish
        > 0.75 → fear               (score -1): bearish sentiment
        > 0.55 → neutral            (score  0): balanced
        > 0.40 → complacency        (score +1): watch for reversal
               → extreme_complacency(score +2): contrarian bearish

    Returns:
        (regime_label, score) where negative scores are contrarian-bullish signals.
    """
    if pcr > 1.00:
        return ("extreme_fear", -2)
    if pcr > 0.75:
        return ("fear", -1)
    if pcr > 0.55:
        return ("neutral", 0)
    if pcr > 0.40:
        return ("complacency", 1)
    return ("extreme_complacency", 2)


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
