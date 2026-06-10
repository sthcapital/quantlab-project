"""CBOE VIX and put/call ratio history — no API key required."""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass
from datetime import date, datetime

import requests

logger = logging.getLogger(__name__)

_CDN = "https://cdn.cboe.com/api/global/us_indices/daily_prices"
_VIX_URL        = f"{_CDN}/VIX_History.csv"
_TOTAL_PCR_URL  = f"{_CDN}/PC_History.csv"
_EQUITY_PCR_URL = f"{_CDN}/EQUITY_PC_History.csv"
_INDEX_PCR_URL  = f"{_CDN}/INDEX_PC_History.csv"
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; QuantLab/1.0; +https://sthcapital.com)"}

# CBOE index option underlyings — excluded from equity PCR, included in index PCR
_INDEX_UNDERLYINGS: frozenset[str] = frozenset({
    "SPX", "SPXW", "NDX", "NDXP", "RUT", "RUTW",
    "VIX", "VIXW", "XSP", "DJX", "OEX", "XEO", "MNX",
})


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
    Download a CBOE PCR CSV from the CDN and return PcrBars in [start, end].

    CBOE PCR files use the column layout:
        DATE, CALL, PUT, TOTAL
    where TOTAL is the put/call ratio.  Returns an empty list (never raises)
    when the CDN is unavailable — CBOE restricted PCR CSV access in 2025.
    """
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.HTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", "?")
        logger.warning(
            "CBOE PCR CDN returned %s for %s — PCR unavailable from CDN",
            status, url,
        )
        return []
    except requests.RequestException as exc:
        logger.warning("CBOE PCR request failed (%s) — PCR unavailable", exc)
        return []

    bars: list[PcrBar] = []
    reader = csv.DictReader(io.StringIO(resp.text))
    for row in reader:
        try:
            dt = datetime.strptime(row["DATE"].strip(), "%m/%d/%Y").date()
        except (KeyError, ValueError):
            continue
        if dt < start or dt > end:
            continue
        raw = row.get("TOTAL") or row.get("CLOSE") or row.get("P/C Ratio", "")
        try:
            bars.append(PcrBar(date=dt, close=float(raw)))
        except (ValueError, TypeError):
            continue

    bars.sort(key=lambda b: b.date)
    return bars


def _pcr_from_flat_file(trade_date: date) -> "tuple[float, float, float] | None":
    """
    Compute (equity_pcr, total_pcr, index_pcr) from the Massive options flat file.

    Returns None when no flat file is cached for trade_date.  Uses volume
    aggregated by option_type ('C'/'P') across all underlyings; separates
    equity vs index using _INDEX_UNDERLYINGS.
    """
    try:
        from quantlab.providers.flat_files import FlatFileProvider
        import pyarrow.parquet as pq

        path = FlatFileProvider().options_cache_path(trade_date)
        if not path.exists():
            return None

        tbl = pq.read_table(
            str(path), columns=["option_type", "volume", "underlying"]
        ).to_pydict()

        eq_puts = eq_calls = idx_puts = idx_calls = 0
        for opt_type, vol, underlying in zip(
            tbl["option_type"], tbl["volume"], tbl["underlying"]
        ):
            if not vol:
                continue
            is_index = underlying in _INDEX_UNDERLYINGS
            if opt_type == "P":
                if is_index:
                    idx_puts += vol
                else:
                    eq_puts += vol
            elif opt_type == "C":
                if is_index:
                    idx_calls += vol
                else:
                    eq_calls += vol

        eq_pcr    = eq_puts  / eq_calls   if eq_calls  > 0 else 0.0
        idx_pcr   = idx_puts / idx_calls  if idx_calls > 0 else 0.0
        total_pcr = (eq_puts + idx_puts) / (eq_calls + idx_calls) \
                    if (eq_calls + idx_calls) > 0 else 0.0
        return round(eq_pcr, 4), round(total_pcr, 4), round(idx_pcr, 4)

    except Exception as exc:
        logger.debug("Flat-file PCR computation failed for %s: %s", trade_date, exc)
        return None


def _flat_file_pcr_range(
    start: date, end: date, kind: str
) -> list[PcrBar]:
    """
    Compute PCR for each cached flat-file day in [start, end].

    kind: 'equity' | 'total' | 'index'
    """
    from datetime import timedelta

    bars: list[PcrBar] = []
    d = start
    while d <= end:
        result = _pcr_from_flat_file(d)
        if result is not None:
            eq_pcr, total_pcr, idx_pcr = result
            val = {"equity": eq_pcr, "total": total_pcr, "index": idx_pcr}.get(kind, 0.0)
            if val > 0.0:
                bars.append(PcrBar(date=d, close=val))
        d += timedelta(days=1)
    return bars


def fetch_total_pcr(start: date, end: date) -> list[PcrBar]:
    """
    Return CBOE total (equity + index) put/call ratio for [start, end].
    Tries CBOE CDN first; falls back to Massive flat-file computation.
    """
    bars = _fetch_pcr(_TOTAL_PCR_URL, start, end)
    if not bars:
        bars = _flat_file_pcr_range(start, end, "total")
    return bars


def fetch_equity_pcr(start: date, end: date) -> list[PcrBar]:
    """
    Return CBOE equity-only put/call ratio for [start, end].
    Tries CBOE CDN first; falls back to Massive flat-file computation.
    """
    bars = _fetch_pcr(_EQUITY_PCR_URL, start, end)
    if not bars:
        bars = _flat_file_pcr_range(start, end, "equity")
    return bars


def fetch_index_pcr(start: date, end: date) -> list[PcrBar]:
    """
    Return CBOE index-only put/call ratio for [start, end].
    Tries CBOE CDN first; falls back to Massive flat-file computation.
    """
    bars = _fetch_pcr(_INDEX_PCR_URL, start, end)
    if not bars:
        bars = _flat_file_pcr_range(start, end, "index")
    return bars


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
