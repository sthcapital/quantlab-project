# QuantLab Strategy — Core Algorithm Thesis

## The Pattern

The system targets one repeatable market pattern: a stock that has spent months
quietly building institutional ownership, whose business fundamentals are
visibly improving, and whose price is coiling just beneath a resistance level —
then breaks out on a meaningful catalyst with conviction volume.

This is not a momentum-chasing system. The edge is in identifying the *setup
before* the move, not reacting to price action that has already happened.

### The 6-Stage Sequence

```
1. CONSOLIDATION BASE
   Stock spends 3–6+ months trading in a tight range.
   Price volatility (ATR) contracts. Volume dries up.
   Relative strength vs market holds or improves quietly.

2. EARNINGS ACCELERATION
   Underlying business begins improving:
   - EPS growth rate turns positive or accelerates
   - Revenue growth inflecting upward
   - Guidance raised, estimates revised higher
   This is the fuel. Without it, a breakout is noise.

3. INSTITUTIONAL ACCUMULATION
   Large buyers absorb supply inside the base.
   Signature: above-average volume on up days,
   below-average volume on down days.
   The base "holds" not because sellers are absent
   but because buyers are absorbing every sale.

4. CATALYST APPEARANCE
   An earnings beat, analyst upgrade, or management
   event coincides with — or precedes — the breakout.
   The catalyst is the trigger, not the reason to own.

5. BREAKOUT CONFIRMATION
   Price clears the base highs on volume > 1.5× average.
   Entry is on the breakout bar or first pullback to the
   breakout level — not early in the base formation.

6. HOLD / STOP MANAGEMENT
   Stop placed below the base (ATR-based).
   Exit on signal failure or first sign of distribution.
```

---

## What the Algorithm Does Today

The scanner runs daily across the SP500 sample universe and scores each symbol
on available confirmation layers:

| Layer | Weight | Status |
|---|---|---|
| Price breakout (N-day high) | 0.30 | **Live** |
| Market regime (SPY > 200 SMA) | 0.20 | **Live** |
| News catalyst quality | 0.00–0.20 | **Live** (IBKR headlines) |
| Relative volume ≥ 1.5× | 0.10 | **Live** |
| News confidence score | 0.10 | **Live** (IBKR C: score) |
| Historical OOS edge | — | **Live** (DuckDB query) |
| Base detection | — | **Not built** |
| Earnings acceleration | — | **Not built** (needs fundamentals) |
| Institutional volume signature | — | **Not built** |

The `score_conviction()` function weights news categories from real observed
trade outcomes: earnings/management catalysts showed +0.32–0.55% average trade
returns vs breakouts without news; downgrade news showed −0.17%, and now
reduces conviction rather than adding to it.

---

## Signals to Build — Priority Order

### 1. Base Detection
**What it is:** Identify stocks in late-stage consolidation — the coiling before
the spring.

**Implementation:**
```python
def base_quality_score(bars, lookback_weeks=20) -> float:
    """
    Score 0.0–1.0 measuring how tight the consolidation base is.

    Metrics:
    - ATR decline: current 14-day ATR vs ATR 8 weeks ago
      (declining ATR = contracting volatility = base forming)
    - Price range compression: (high - low) / low over lookback window
      as a % of price. Target: < 15% range for a tight base.
    - Proximity to highs: close / max(high over lookback) > 0.85
      (price holding near the top of the base, not drifting down)
    - Duration: number of weeks price has stayed within the base range
      (longer base = stronger coil; minimum 12 weeks preferred)

    Returns 1.0 for an ideal tight, long, high-proximity base.
    Returns 0.0 for wide, short, or price-drifting-lower patterns.
    """
```

**Data needed:** OHLCV bars only (already available via IBKR).

**Where it fits:** `src/quantlab/signals/__init__.py` → `base_quality_score()`.
Wire into `scan_symbol()` and add 0.15 weight to `score_conviction()`.

---

### 2. Earnings Acceleration
**What it is:** Confirm the business is improving, not just the stock price.

**Metrics needed:**
- EPS growth rate: current quarter vs same quarter prior year (YoY)
- Revenue growth rate: same basis
- Earnings surprise: actual vs consensus estimate
- Estimate revision trend: are analysts raising or cutting forward estimates?

**Data source required:** IBKR does not provide fundamental data.
Options in priority order:
1. **Polygon.io** — `GET /v2/reference/financials/{ticker}` — most complete,
   reasonable pricing, good Python SDK
2. **Alpha Vantage** — free tier available, rate-limited, EPS endpoint exists
3. **SEC EDGAR XBRL API** — free, delayed ~2 days, no rate limit

**Implementation sketch:**
```python
@dataclass
class EarningsProfile:
    symbol: str
    eps_growth_yoy: float | None      # % change vs year-ago quarter
    revenue_growth_yoy: float | None
    last_eps_surprise_pct: float | None  # (actual - estimate) / abs(estimate)
    estimate_revision_trend: str       # "rising" | "falling" | "flat" | "unknown"
    accelerating: bool                 # True when eps_growth is positive and
                                       # increasing for 2+ consecutive quarters

def fetch_earnings_profile(symbol: str, provider: str = "polygon") -> EarningsProfile:
    ...
```

**Where it fits:** `src/quantlab/fundamentals/__init__.py` (new Layer 3 module).
Add `accelerating_earnings: bool = False` field to `ScanResult`.
Score boost: +0.20 when `accelerating=True` (this is the most important filter).

**Note:** Without earnings acceleration, the breakout scanner catches too many
false starts. This is the single highest-value signal gap.

---

### 3. Institutional Volume Signature
**What it is:** Detect whether the base is being accumulated by large buyers or
simply going sideways with no interest.

**The signature:**
- On up-days inside the base: volume above N-day average (accumulation)
- On down-days inside the base: volume below N-day average (lack of distribution)
- Net "accumulation days" vs "distribution days" over the base period

**Implementation:**
```python
def accumulation_score(bars, lookback: int = 60) -> float:
    """
    Score 0.0–1.0 measuring institutional accumulation inside a base.

    For each bar in the lookback window:
      - "Accumulation bar": close > open AND volume > avg_volume_20d
      - "Distribution bar": close < open AND volume > avg_volume_20d

    Score = (accumulation_bars - distribution_bars) / lookback
    Normalised to [0.0, 1.0].

    A score above 0.6 indicates net institutional buying.
    A score below 0.4 indicates net distribution — avoid.
    """
```

**Data needed:** OHLCV bars only (already available).

**Where it fits:** `src/quantlab/signals/__init__.py` → `accumulation_score()`.
Add 0.10 weight in `score_conviction()`.

---

### 4. Breakout Quality Score
**What it is:** Not all breakouts are equal. This scores the quality of the
breakout bar itself.

**Dimensions:**
```
volume_ratio    = today_volume / avg_volume_20d
                  Target: > 1.5× (strong), > 2.5× (exceptional)

distance_from_base = (close - base_high) / base_high
                     Close proximity (< 3%) = cleaner entry
                     Too extended (> 8%) = chase risk, skip

catalyst_present = news_count > 0 AND news_category in
                   {"earnings", "upgrade", "management"}

range_expansion  = (high - low) / atr_14
                   Expanding range on breakout day confirms conviction
```

**Implementation:**
```python
@dataclass
class BreakoutQuality:
    volume_ratio: float
    distance_from_base_pct: float
    catalyst_present: bool
    range_expansion: float
    score: float   # composite 0.0–1.0
```

**Where it fits:** Computed inside `scan_symbol()`, added to `ScanResult`.
The `score_conviction()` function already has a `rel_volume` layer; this
replaces it with the richer breakout quality score.

---

## Data Sources Summary

| Data type | Current source | Gap |
|---|---|---|
| Daily OHLCV bars | IBKR TWS | None — working |
| News headlines | IBKR TWS | None — working |
| Market regime | IBKR TWS (SPY bars) | None — working |
| EPS / revenue / estimates | **None** | Critical gap |
| Options flow / unusual activity | None | Phase 5+ |
| Short interest | None | Phase 5+ |

The most important next integration is a fundamentals provider.
Polygon.io's free tier covers historical financials; the paid tier adds
real-time estimates and revision tracking, which is what the accelerating
earnings filter ultimately needs.

---

## What "Historical Winners" Looks Like in the Data

From the 2024–2025 IBKR live run (breakout lookback=5, 10 bps cost):

**Top performers — common characteristics:**
- CAT: industrial cyclical, earnings recovery, institutional base
- AAPL: defensive growth, tight consolidation, frequent analyst upgrades
- LLY: weight-loss drug earnings acceleration — textbook fundamental catalyst
- NVDA: AI infrastructure earnings acceleration — multi-quarter EPS beat streak
- ABBV: pharmaceutical pipeline — management/analyst catalyst pattern

**Underperformers — what they share:**
- PG, KO, PEP: consumer staples in a rising rate environment — earnings
  deceleration, no institutional catalyst, low volatility makes breakouts
  mean-revert quickly
- CVX, NEE: energy/utilities — macro-driven, not earnings-driven; the breakout
  signal fires on noise, not accumulation
- META, AMZN: high-beta tech where breakouts in 2024 were frequently
  followed by sharp reversals; high vol means ATR stop is wide, drawdowns large

**The pattern confirms:** earnings acceleration is the separating variable.
LLY and NVDA had the most obvious multi-quarter EPS acceleration of any name
in the universe. Their strong performance is not a coincidence — it is the
signal the system needs to learn to detect before the breakout.

---

## Implementation Roadmap

```
Phase 5 (next):
  [ ] Polygon.io fundamentals client (Layer 3)
  [ ] EarningsProfile dataclass + fetch_earnings_profile()
  [ ] base_quality_score() in signals
  [ ] accumulation_score() in signals
  [ ] Wire all four new signals into score_conviction()

Phase 6:
  [ ] Live paper trading via IBKR order submission
  [ ] Position sizing module (Kelly / fixed fractional)
  [ ] Portfolio-level risk limits (max concentration, sector limits)
  [ ] Daily pre-market automated scan + alert delivery

Phase 7:
  [ ] Walk-forward re-optimisation on rolling 6-month windows
  [ ] Strategy degradation monitor (alert when live Sharpe < IS baseline)
  [ ] Options flow overlay (unusual call buying as confirmation layer)
```
