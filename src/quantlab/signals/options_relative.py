"""
quantlab.signals.options_relative — relative unusual-options scoring.

Replaces the absolute-threshold compute_options_score path, which saturated:
on 2026-06-10/11 the within-chain "unusual calls" check (max strike volume vs
the chain's own average) fired on ~97% of monitored symbols and PCR < 0.5 on
~76%, so 81% of the watchlist scored ≥ 0.6.  Both components compare a symbol
against itself *today* or against fixed absolute cutoffs; neither asks the
only question that matters: is today unusual *for this symbol*?

Two-stage design:

1. Per-symbol baseline — today's total call volume vs the symbol's OWN
   trailing 20-session history → z-score.  A name doing 3× its own normal
   call volume is a signal; a name with high absolute volume is just a
   liquid name.

2. Cross-sectional gate — after every monitored symbol is scored, "unusual"
   means top decile of the day's scores (percentile configurable, default
   p90).  This caps the daily flag rate at ~10% by construction: the signal
   was predictive on mid-caps precisely because it was rare.

MISSING ≠ ZERO: a symbol without enough baseline history gets score None
(excluded from the gate), never 0.0.

All functions here are pure — no I/O, no provider dependencies.
"""

from __future__ import annotations

import statistics
from typing import Mapping, Sequence

# Baseline shorter than this cannot support an "unusual for this symbol" claim.
MIN_BASELINE_SESSIONS = 10

# Z-scores are capped here so a zero-variance baseline (or one absurd day)
# cannot produce an unbounded value.
ZSCORE_CAP = 10.0

# Materiality floor: below this many contracts today, the day is "not unusual"
# (z = 0.0) no matter the ratio — 8 contracts over a 2-contract baseline is
# noise, not accumulation (same rationale as _MIN_AVG_VOL in unusual_options).
# This is a real claim, not missing data, so 0.0 — not None — is correct.
MIN_TODAY_CALL_VOLUME = 100.0

# Flag eligibility: a symbol must be at least this many σ above its own
# baseline to be flaggable.  The percentile gate is a CAP on the daily rate,
# not a quota — on a quiet day, without this floor, the gate would fill its
# decile with non-anomalies (e.g. FITB on 2026-06-11: z = −0.72, volume BELOW
# its own baseline, carried into the top decile by IV skew alone).
MIN_FLAG_ZSCORE = 2.0

# Liquidity floor for FLAG eligibility: a 20-session baseline average below
# this many contracts cannot support an accumulation claim — EG flagged at
# z = 10 on 9,043 contracts vs a 24-contract baseline (2026-06-11), where one
# hedger rolling a position is indistinguishable from accumulation.  Symbols
# below the floor are still scored, displayed, and persisted — they just
# cannot receive the unusual flag / gate credit.
MIN_BASELINE_CONTRACTS = 75.0

# Direction ceiling for FLAG eligibility: a session PCR above this is
# put-dominated flow — not accumulation evidence, whatever the call-volume z
# (HST on 2026-06-11: z = 10 with PCR 6.25).  Still scored/persisted, and
# callers tag such rows put_dominated in options_snapshots: they are future
# short-side signal data (SHORT_SIGNAL_ENABLED is False, the record exists).
MAX_GATE_PCR = 1.5

# Z-score at which the volume component saturates at 1.0.
_Z_SATURATION = 4.0

# Component weights (re-normalised when pcr / iv_skew are unavailable).
_W_VOLUME = 0.55
_W_PCR = 0.25
_W_SKEW = 0.20


# ── Per-symbol z-score ─────────────────────────────────────────────────────────

def volume_zscore(
    today_volume: float | None,
    baseline: Sequence[float],
    min_sessions: int = MIN_BASELINE_SESSIONS,
    min_today_volume: float = MIN_TODAY_CALL_VOLUME,
) -> float | None:
    """
    Z-score of today's volume against the symbol's own trailing baseline.

    z = (today − mean(baseline)) / sample_std(baseline), capped to ±ZSCORE_CAP.

    Returns None (MISSING ≠ ZERO) when today_volume is None or the baseline
    has fewer than ``min_sessions`` observations — without history there is
    no basis for an "unusual for this symbol" claim.

    Returns 0.0 when today_volume is below ``min_today_volume``: immaterial
    activity is affirmatively "not unusual" regardless of the ratio to a
    near-zero baseline.

    Zero-variance baseline: 0.0 when today equals the constant level,
    ±ZSCORE_CAP when above/below it.
    """
    if today_volume is None or len(baseline) < min_sessions:
        return None
    if today_volume < min_today_volume:
        return 0.0

    mean = statistics.mean(baseline)
    std = statistics.stdev(baseline)

    if std == 0.0:
        if today_volume == mean:
            return 0.0
        return ZSCORE_CAP if today_volume > mean else -ZSCORE_CAP

    z = (today_volume - mean) / std
    return max(-ZSCORE_CAP, min(ZSCORE_CAP, z))


# ── Per-symbol composite score ─────────────────────────────────────────────────

def relative_options_score(
    vol_zscore: float | None,
    pcr: float | None = None,
    iv_skew: float | None = None,
) -> float | None:
    """
    Composite 0.0–1.0 score from the volume z-score plus optional PCR and
    IV-skew tilts.  Continuous everywhere — no step thresholds — so the
    cross-sectional gate has real rank information to work with.

    Components:
        volume z (55%): z / 4 clamped to [0, 1] — 4σ above own baseline = max
        PCR      (25%): 1 / (1 + pcr) — continuous bullishness, 0 puts → 1.0
        IV skew  (20%): already 0–1 from the provider

    Weights are re-normalised over the available components when pcr or
    iv_skew is None.  Returns None when vol_zscore is None: the per-symbol
    volume anomaly IS the signal; without it no "unusual" claim is possible
    (MISSING ≠ ZERO).
    """
    if vol_zscore is None:
        return None

    vol_comp = max(0.0, min(1.0, vol_zscore / _Z_SATURATION))
    parts: list[tuple[float, float]] = [(vol_comp, _W_VOLUME)]

    if pcr is not None:
        parts.append((1.0 / (1.0 + max(0.0, pcr)), _W_PCR))
    if iv_skew is not None:
        parts.append((max(0.0, min(1.0, iv_skew)), _W_SKEW))

    total_weight = sum(w for _, w in parts)
    score = sum(v * w for v, w in parts) / total_weight
    return round(score, 4)


# ── Cross-sectional gate ───────────────────────────────────────────────────────

def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolation percentile (p in [0, 100]) of a non-empty sequence."""
    if not values:
        raise ValueError("percentile() of empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def cross_sectional_flags(
    scores: Mapping[str, float | None],
    percentile_cut: float = 90.0,
    min_universe: int = 10,
    zscores: Mapping[str, float | None] | None = None,
    min_zscore: float = MIN_FLAG_ZSCORE,
    baseline_means: Mapping[str, float | None] | None = None,
    min_baseline: float = MIN_BASELINE_CONTRACTS,
    pcrs: Mapping[str, float | None] | None = None,
    max_pcr: float = MAX_GATE_PCR,
) -> set[str]:
    """
    Flag the top tail of the day's scores: symbols whose score is strictly
    above the ``percentile_cut`` percentile of all scored symbols.

    Strict ``>`` means the daily flag rate is capped at ~(100 − p)% by
    construction, and a degenerate day where most scores tie at the threshold
    flags nothing rather than everything.

    The percentile is a CAP, not a quota: when ``zscores`` is provided, a
    symbol must also be at least ``min_zscore`` σ above its own baseline to
    flag.  On a quiet day the gate flags fewer than its decile rather than
    filling it with non-anomalies.  (The threshold is still computed over all
    scored symbols so the cut stays a stable day-level statistic.)

    Liquidity floor: when ``baseline_means`` is provided, a symbol whose
    trailing baseline average is below ``min_baseline`` contracts cannot
    flag — at a 24-contract baseline a z = 10 spike is one hedger rolling a
    position, not accumulation.  Such symbols are still scored and persisted;
    they only lose gate credit.

    Direction ceiling: when ``pcrs`` is provided, a symbol whose measured
    session PCR exceeds ``max_pcr`` cannot flag — put-dominated flow is not
    LONG-accumulation evidence regardless of call-volume z (HST 2026-06-11:
    z = 10, PCR 6.25).  A None PCR passes: unknown direction is not evidence
    of put domination.

    Symbols with score None (no baseline) are excluded from both the
    percentile computation and the flags.  Returns the empty set when fewer
    than ``min_universe`` symbols are scored — a percentile over a handful
    of names is noise.
    """
    scored = {sym: s for sym, s in scores.items() if s is not None}
    if len(scored) < min_universe:
        return set()

    threshold = percentile(list(scored.values()), percentile_cut)
    flagged = {sym for sym, s in scored.items() if s > threshold}

    if zscores is not None:
        flagged = {
            sym for sym in flagged
            if zscores.get(sym) is not None and zscores[sym] >= min_zscore
        }
    if baseline_means is not None:
        flagged = {
            sym for sym in flagged
            if baseline_means.get(sym) is not None
            and baseline_means[sym] >= min_baseline
        }
    if pcrs is not None:
        flagged = {
            sym for sym in flagged
            if pcrs.get(sym) is None or pcrs[sym] <= max_pcr
        }
    return flagged
