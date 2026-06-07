"""
Unusual options activity detector — mid-cap institutional signal.

Detects anomalous call volume spikes at specific strikes that indicate
directed institutional accumulation, specifically for mid-cap names where
the options market is deep enough to be informative but thin enough that
big players leave clear footprints.

Data source: Massive S3 flat files via FlatFileProvider (no API key required).
The 20-day historical baseline uses cached Parquet files only — dates that
haven't been synced are silently skipped.

Key refinement (post live-validation):
    Consecutive-day filter: requires the same strike to show unusual activity
    on 2+ of the last 5 trading days.  This eliminates one-day spikes (which
    were false positives in live testing — BROS May-18) while keeping genuine
    multi-day institutional accumulation (PATH May-26, CELH May-21).

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

# Consecutive-day scoring lookup: days of unusual activity → component score.
# Today always counts as 1; prior days add to the count.
_CONSEC_SCORE: dict[int, float] = {
    1: 0.10,   # only today — weak (likely noise); usually filtered by min_consecutive_days
    2: 0.45,   # today + 1 prior day — possible accumulation
    3: 0.75,   # today + 2 prior days — probable institutional
    4: 0.90,   # strong accumulation pattern
    5: 1.00,   # maximum — strike active all 5 days
}


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
    consecutive_days: int         # unusual-activity days out of last 5 trading days
                                  # (today counts as 1; prior days add to total)
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

    Key: ``(strike, option_type)`` tuple — aggregates volume across all expiries
    at the same strike so new expiry series inherit the established baseline.

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


def _load_recent_days(
    symbol: str,
    as_of: date,
    flat_file_provider,
    n_days: int = 4,
) -> list[dict[tuple[float, str], float]]:
    """
    Load volume snapshots for the ``n_days`` cached trading days before ``as_of``.

    Returns a list (oldest first) of dicts mapping ``(strike, option_type)`` →
    volume.  Non-cached dates are silently skipped.  Used for the consecutive-day
    count without triggering new S3 downloads.
    """
    snapshots: list[dict[tuple[float, str], float]] = []
    current = as_of - timedelta(days=1)
    limit = as_of - timedelta(days=n_days * 3)   # buffer for weekends/holidays

    while len(snapshots) < n_days and current >= limit:
        try:
            cache = flat_file_provider.options_cache_path(current)
            if not cache.exists():
                current -= timedelta(days=1)
                continue
            records = flat_file_provider.get_options_chain_from_flatfile(symbol, current)
            snapshots.append({
                (float(r["strike"]), str(r["option_type"])): float(r["volume"])
                for r in records
            })
        except Exception:
            pass
        current -= timedelta(days=1)

    snapshots.reverse()   # oldest first
    return snapshots


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
    min_consecutive_days: int = 2,
) -> list[UnusualOptionsSignal]:
    """
    Detect unusual institutional call activity for ``symbol`` on ``as_of``.

    All five filters must pass today:

    1. ``option_type == "C"``                   calls only (long signals)
    2. ``volume_ratio >= volume_ratio_threshold`` today 5× the 20-day avg
       (pass 8.0 for consumer/restaurant names, 5.0 for software/biotech)
    3. ``avg_20day_volume >= min_avg_volume``    strike is normally tradeable
    4. ``otm_pct in [0.03, 0.20]``              OTM positioning, not ATM hedging
    5. ``days_to_expiry in [10, 60]``            no weeklies, no LEAPs

    Consecutive-day filter (post-validation):

        Only keeps strikes where the same unusual activity appeared on
        ``min_consecutive_days`` or more of the last 5 trading days
        (today counts as day 1).  Default min is 2 — this eliminates
        one-day spikes (false positives like BROS May-18) while keeping
        multi-day accumulation patterns (PATH May-26, CELH May-21).

    Concentration flag: ``is_concentrated=True`` when the top-3 strikes by
    today's volume account for > 60% of all unusual-call volume.

    Args:
        symbol:                  Underlying ticker.
        as_of:                   Scan date.
        flat_file_provider:      FlatFileProvider (cached reads only).
        spot_price:              Current price for OTM % calculation.
        volume_ratio_threshold:  Minimum volume multiple.  Use 8.0 for
                                 consumer/restaurant names and 5.0 for
                                 software/biotech mid-caps.
        min_consecutive_days:    Minimum unusual-activity days out of last 5
                                 (including today).  Set to 1 to disable.

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

    # Apply per-contract filters for today
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

    # ── Consecutive-day filter ─────────────────────────────────────────────────
    # Load the last 4 cached trading days (today is day 1; these are days 2–5).
    # One read per day regardless of number of candidates.
    recent_snapshots = _load_recent_days(symbol, as_of, flat_file_provider, n_days=4)

    for c in candidates:
        strike = c["strike"]
        avg = c["avg_20day_volume"]
        # Today always counts as 1
        count = 1
        for snap in recent_snapshots:
            vol = snap.get((strike, "C"), 0.0)
            if avg > 0 and vol / avg >= volume_ratio_threshold:
                count += 1
        c["consecutive_days"] = count

    if min_consecutive_days > 1:
        candidates = [c for c in candidates if c["consecutive_days"] >= min_consecutive_days]

    if not candidates:
        return []

    # ── Concentration check ────────────────────────────────────────────────────
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
            oi_today=0.0,
            oi_change_3day=0.0,
            expiry=c["expiry"],
            days_to_expiry=c["dte"],
            otm_pct=c["otm_pct"],
            is_concentrated=concentrated,
            consecutive_days=c["consecutive_days"],
            conviction_score=0.0,
        )
        for c in candidates
    ]
    signals.sort(key=lambda s: s.volume_ratio, reverse=True)
    return signals


# ── Scoring ────────────────────────────────────────────────────────────────────

def score_unusual_activity(signals: list[UnusualOptionsSignal]) -> float:
    """
    Composite 0.0–1.0 score for a list of unusual options signals.

    Components and weights (post live-validation):
        Consecutive days    (35%): persistence is the primary institutional tell
        Volume ratio        (35%): magnitude of the spike above baseline
        Strike concentration(15%): top-3 > 60% of vol → focused positioning
        DTE quality         (15%): tent function peaking at 30–45 DTE

    Hard boost: 3+ consecutive days adds +0.15 to the final score (capped at 1.0).

    Returns 0.0 when ``signals`` is empty.
    """
    if not signals:
        return 0.0

    # Consecutive days component
    max_consec = max(s.consecutive_days for s in signals)
    consec_comp = _CONSEC_SCORE.get(max_consec, 1.0)

    # Volume ratio: log-normalised (5×→0.0, 10×→0.30, 50×→1.0)
    max_ratio = max(s.volume_ratio for s in signals)
    ratio_comp = min(1.0, math.log(max(1.0, max_ratio / 5.0)) / math.log(10.0))

    # Concentration
    conc_comp = 1.0 if any(s.is_concentrated for s in signals) else 0.4

    # DTE quality: tent peaked at 37 days
    def _dte_q(dte: int) -> float:
        if dte <= 10 or dte >= 60:
            return 0.0
        if dte <= 37:
            return (dte - 10) / (37 - 10)
        return (60 - dte) / (60 - 37)

    dte_comp = max(_dte_q(s.days_to_expiry) for s in signals)

    raw = (
        consec_comp * 0.35
        + ratio_comp  * 0.35
        + conc_comp   * 0.15
        + dte_comp    * 0.15
    )

    # Hard boost for strong multi-day accumulation
    if max_consec >= 3:
        raw = min(1.0, raw + 0.15)

    final = round(min(1.0, raw), 4)

    # Back-fill per-signal conviction_score
    for s in signals:
        per_consec = _CONSEC_SCORE.get(s.consecutive_days, 1.0)
        per_ratio = min(1.0, math.log(max(1.0, s.volume_ratio / 5.0)) / math.log(10.0))
        base = (
            per_consec * 0.35
            + per_ratio  * 0.35
            + conc_comp  * 0.15
            + _dte_q(s.days_to_expiry) * 0.15
        )
        if s.consecutive_days >= 3:
            base = min(1.0, base + 0.15)
        s.conviction_score = round(min(1.0, base), 4)

    return final
