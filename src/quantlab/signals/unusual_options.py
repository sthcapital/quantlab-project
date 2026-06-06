"""
Unusual options activity detector — mid-cap institutional signal.

Detects anomalous call volume spikes at specific strikes that indicate
directed institutional accumulation, specifically for mid-cap names where
the options market is deep enough to be informative but thin enough that
big players leave clear footprints.

Data source: Massive S3 flat files via FlatFileProvider (no API key required).
The 20-day historical baseline uses cached Parquet files only — dates that
haven't been synced are silently skipped.

Usage::

    from quantlab.providers.flat_files import FlatFileProvider
    from quantlab.signals.unusual_options import detect_unusual_activity, score_unusual_activity

    flat = FlatFileProvider()
    signals = detect_unusual_activity("CAT", date.today(), flat, spot_price=310.50)
    score   = score_unusual_activity(signals)          # 0.0–1.0
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, timedelta

logger = logging.getLogger(__name__)

# Minimum 20-day average daily volume for a strike to be considered
# "normally traded" — strikes below this are excluded because a few
# contracts look like enormous multiples of effectively-zero baseline.
_MIN_AVG_VOL: float = 10.0


# ── Data structure ─────────────────────────────────────────────────────────────

@dataclass
class UnusualOptionsSignal:
    """Unusual call activity at a single strike for one underlying."""

    symbol: str
    date: date
    strike: float
    option_type: str              # "C" for calls (only calls are emitted)
    today_volume: float
    avg_20day_volume: float       # average daily volume at this strike over 20 days
    volume_ratio: float           # today_volume / avg_20day_volume
    oi_today: float               # open interest today (0.0 when not in flat file)
    oi_change_3day: float         # OI change over last 3 days (0.0 when unavailable)
    expiry: date
    days_to_expiry: int
    otm_pct: float                # (strike − spot) / spot; positive = OTM call
    is_concentrated: bool         # True when top-3 strikes hold >60% of unusual vol
    conviction_score: float       # 0.0–1.0 from score_unusual_activity()


# ── 20-day baseline ────────────────────────────────────────────────────────────

def compute_20day_avg_volume(
    symbol: str,
    as_of: date,
    flat_file_provider,
    trading_days: int = 20,
) -> dict[tuple[float, str], float]:
    """
    Compute average daily call/put volume per strike over the last
    ``trading_days`` cached trading days for ``symbol``.

    Only reads already-cached Parquet files — never triggers a new S3 download.
    Dates without a cached options file are silently skipped.  The denominator
    is ``trading_days`` (not just active days) so rarely-traded strikes produce
    conservative (low) averages.

    Returns:
        ``{(strike, option_type): avg_daily_volume}``
    """
    volume_acc: dict[tuple[float, str], list[float]] = {}
    days_found = 0
    current = as_of - timedelta(days=1)
    lookback_limit = as_of - timedelta(days=trading_days * 3)   # buffer for holidays

    while days_found < trading_days and current >= lookback_limit:
        # Only read from cache — skip if not pre-synced
        try:
            cache_path = flat_file_provider.options_cache_path(current)
            if not cache_path.exists():
                current -= timedelta(days=1)
                continue

            records = flat_file_provider.get_options_chain_from_flatfile(symbol, current)
            if records:
                for rec in records:
                    key = (float(rec["strike"]), str(rec["option_type"]))
                    volume_acc.setdefault(key, []).append(float(rec["volume"]))
                days_found += 1
        except Exception as exc:
            logger.debug(
                "%s avg-vol %s: skipped (%s)", symbol, current, exc
            )
        current -= timedelta(days=1)

    if days_found == 0:
        return {}

    # Denominator is trading_days (not days_found) so strikes that were only
    # active on a subset of days produce conservatively low averages.
    # A strike traded on 3 of 20 days with 200 vol each → avg = 30, not 200.
    return {
        key: sum(vols) / trading_days
        for key, vols in volume_acc.items()
    }


# ── Detection ──────────────────────────────────────────────────────────────────

def detect_unusual_activity(
    symbol: str,
    as_of: date,
    flat_file_provider,
    spot_price: float,
    volume_ratio_threshold: float = 5.0,
    min_avg_volume: float = _MIN_AVG_VOL,
    otm_pct_min: float = 0.03,
    otm_pct_max: float = 0.20,
    dte_min: int = 10,
    dte_max: int = 60,
) -> list[UnusualOptionsSignal]:
    """
    Detect unusual institutional call activity for ``symbol`` on ``as_of``.

    All five filters must pass:

    1. ``option_type == "C"``                  calls only (long signals)
    2. ``volume_ratio >= volume_ratio_threshold``  today 5× the 20-day avg
    3. ``avg_20day_volume >= min_avg_volume``   strike is normally tradeable
    4. ``otm_pct in [0.03, 0.20]``             OTM positioning, not ATM hedging
    5. ``days_to_expiry in [10, 60]``           no weeklies, no LEAPs

    Concentration flag: ``is_concentrated=True`` when the top-3 strikes by
    today's volume account for > 60% of all unusual-call volume.

    Args:
        symbol:                  Underlying ticker.
        as_of:                   Scan date.
        flat_file_provider:      FlatFileProvider (cached reads only for 20-day avg).
        spot_price:              Current price for OTM % calculation.
        volume_ratio_threshold:  Minimum volume multiple (5.0 for mid-cap,
                                 3.0 for large-cap).

    Returns:
        List sorted by volume_ratio descending.  Empty when no signal passes.
    """
    if spot_price <= 0:
        return []

    # Today's chain
    try:
        today_chain = flat_file_provider.get_options_chain_from_flatfile(symbol, as_of)
    except Exception as exc:
        logger.debug("%s: today's options chain unavailable (%s)", symbol, exc)
        return []

    if not today_chain:
        return []

    # 20-day baseline (cache-only reads)
    avg_vol = compute_20day_avg_volume(symbol, as_of, flat_file_provider)

    # Apply filters
    candidates: list[dict] = []
    for rec in today_chain:
        if rec.get("option_type") != "C":
            continue

        strike = float(rec["strike"])
        today_vol = float(rec.get("volume", 0))
        expiry_str = rec.get("expiry", "")

        try:
            expiry_date = date.fromisoformat(expiry_str)
        except ValueError:
            continue

        dte = (expiry_date - as_of).days
        otm_pct = (strike - spot_price) / spot_price

        key = (strike, "C")
        avg = avg_vol.get(key, 0.0)

        if not (otm_pct_min <= otm_pct <= otm_pct_max):
            continue
        if not (dte_min <= dte <= dte_max):
            continue
        if avg < min_avg_volume:
            continue

        ratio = today_vol / avg if avg > 0 else 0.0
        if ratio < volume_ratio_threshold:
            continue

        candidates.append({
            "strike": strike,
            "today_volume": today_vol,
            "avg_20day_volume": avg,
            "volume_ratio": ratio,
            "expiry": expiry_date,
            "dte": dte,
            "otm_pct": otm_pct,
        })

    if not candidates:
        return []

    # Concentration check: top-3 strikes > 60% of total unusual volume
    total_vol = sum(c["today_volume"] for c in candidates)
    top3_vol = sum(
        c["today_volume"]
        for c in sorted(candidates, key=lambda x: x["today_volume"], reverse=True)[:3]
    )
    concentrated = total_vol > 0 and (top3_vol / total_vol) > 0.60

    signals = [
        UnusualOptionsSignal(
            symbol=symbol,
            date=as_of,
            strike=c["strike"],
            option_type="C",
            today_volume=c["today_volume"],
            avg_20day_volume=c["avg_20day_volume"],
            volume_ratio=c["volume_ratio"],
            oi_today=0.0,           # flat file has no OI column
            oi_change_3day=0.0,     # flat file has no OI column
            expiry=c["expiry"],
            days_to_expiry=c["dte"],
            otm_pct=c["otm_pct"],
            is_concentrated=concentrated,
            conviction_score=0.0,   # filled in by score_unusual_activity
        )
        for c in candidates
    ]
    signals.sort(key=lambda s: s.volume_ratio, reverse=True)
    return signals


# ── Scoring ────────────────────────────────────────────────────────────────────

def score_unusual_activity(signals: list[UnusualOptionsSignal]) -> float:
    """
    Composite 0.0–1.0 score for a list of unusual options signals.

    Components and weights:
        Volume ratio magnitude  (50%): log-normalised; 5×→0.0, 10×→0.30, 50×→1.0
        Strike concentration    (25%): top-3 > 60% of vol → 1.0, else 0.4
        DTE quality             (25%): tent function peaking at 30–45 DTE

    Returns 0.0 when ``signals`` is empty.
    """
    if not signals:
        return 0.0

    # Volume ratio: log(ratio/5) / log(10)
    #   ratio=5  → 0.00   ratio=10 → 0.30   ratio=50 → 1.00
    max_ratio = max(s.volume_ratio for s in signals)
    ratio_comp = min(1.0, math.log(max(1.0, max_ratio / 5.0)) / math.log(10.0))

    # Concentration
    conc_comp = 1.0 if any(s.is_concentrated for s in signals) else 0.4

    # DTE quality: tent peaked at 37 days (midpoint of 10–60 range biased toward 30-45)
    def _dte_q(dte: int) -> float:
        if dte <= 10 or dte >= 60:
            return 0.0
        if dte <= 37:
            return (dte - 10) / (37 - 10)
        return (60 - dte) / (60 - 37)

    dte_comp = max(_dte_q(s.days_to_expiry) for s in signals)

    raw = ratio_comp * 0.50 + conc_comp * 0.25 + dte_comp * 0.25
    final = round(min(1.0, raw), 4)

    # Back-fill conviction_score on each signal
    for s in signals:
        per_ratio = min(1.0, math.log(max(1.0, s.volume_ratio / 5.0)) / math.log(10.0))
        s.conviction_score = round(
            per_ratio * 0.50 + conc_comp * 0.25 + _dte_q(s.days_to_expiry) * 0.25, 4
        )

    return final
