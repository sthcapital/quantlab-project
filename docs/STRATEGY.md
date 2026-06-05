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
  [ ] Wire Wyckoff scores into score_conviction()
  [ ] base_quality_score() validated against known historical bases

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

---

## Wyckoff Accumulation / Distribution Framework

Richard Wyckoff's method describes how institutional operators (the
"composite operator") accumulate or distribute large positions without
moving price against themselves. Recognising which phase the stock is in —
accumulation, markup, distribution, or markdown — is the most important
structural filter in the system.

### Why Wyckoff Belongs Here

The current scanner fires a breakout signal without knowing whether the
base represents genuine accumulation (institutions absorbing supply) or a
distribution top (institutions offloading to retail). The difference in
expected outcome is large. A breakout from a Wyckoff accumulation base
tends to have follow-through; a breakout from a distribution top fails
and reverses sharply.

---

### Accumulation Signatures (Bullish — Find These)

**1. Absorption**
Price is declining or testing lows on high volume but *not making new lows*.
High volume without price progression means buyers are absorbing every share
offered. The market refuses to go down despite heavy supply.

```
Detection: over a rolling N-bar window, find bars where:
  - volume > avg_volume_20d * 1.3  (above-average supply)
  - low >= prior_N_bar_low * 0.99  (price NOT making new lows)
Absorption score = proportion of high-volume bars that are non-declining
```

**2. Cause Being Built (Tight Range + Declining Volume)**
After absorption, price enters a tight trading range. Volume contracts
further. This is the "cause" phase — the coiling of potential energy.
The longer and tighter this phase, the larger the eventual move.

```
Detection:
  - (N-week high - N-week low) / N-week low < 0.12  (< 12% range)
  - 14-day ATR now < ATR 8 weeks ago * 0.75  (volatility declining)
  - Duration of this condition >= min_weeks (12 preferred)
```

**3. Volume Character — Up Days vs Down Days**
Inside a Wyckoff accumulation base, institutions absorb on weakness and
let price rise on their own buying. The fingerprint: above-average volume
on up-days, below-average on down-days.

```
For each bar in the base window:
  accumulation_bar: close > open AND volume > avg_volume_20d
  distribution_bar: close < open AND volume > avg_volume_20d

volume_character_score = (accumulation_bars - distribution_bars)
                         / total_bars_in_window
Clipped to [0.0, 1.0]. Score > 0.55 = net accumulation.
```

**4. The Spring (Shakeout)**
One of the most reliable Wyckoff signatures: price briefly undercuts
the base lows (triggering stops and retail selling), then reverses sharply
back above support within 1–3 bars. The false breakdown is engineered by
operators to shake out weak hands before the markup begins.

```
Detection (over recent N bars):
  - At least one bar where low < min_low_of_base * (1 - threshold)
    (undercuts the base, e.g. 1.5% below support)
  - That bar (or within 2 bars) closes back above the base low
  - Volume on the spring bar is elevated (confirms shakeout, not breakdown)
```

**5. Breakout Volume Supremacy**
The breakout bar's volume must be the highest single-day volume in the
entire base period. This confirms that the move has institutional
sponsorship and is not a low-liquidity false break.

```
Detection: bars[-1].volume == max(b.volume for b in base_window + [bars[-1]])
```

---

### Distribution Signatures (Bearish — Filter These Out)

These indicate the composite operator is *selling* into strength.
A breakout from a distribution top is a trap.

**Supply Meeting Demand**
Price is rising on high volume but *not making new highs*.
Sellers are matching every buyer. The market refuses to go up
despite apparent buying pressure.

```
Detection: bars where volume > avg * 1.3 AND high <= prior_N_bar_high * 1.01
```

**Reversed Volume Character**
Up-days on below-average volume (no conviction behind rallies),
down-days on above-average volume (heavy supply on weakness).
This is the mirror image of accumulation. `volume_character_score < 0.45`.

**Failed Tests of Prior Highs**
Price approaches a resistance level repeatedly but closes below it
each time. The market is testing whether supply has been absorbed
(in accumulation) or whether it is still present (distribution).
Three failed close-above-high attempts in a tight window = distribution flag.

---

### Order Flow Confirmation (Phase 5+ Data Sources)

These signals require data beyond OHLCV bars. Listed here to define
what to build when the data is available.

**Options Market Positioning**
- Unusual call OI building at strikes above current price in the weeks
  before a breakout: institutions buying calls to lever their position
- Put/call ratio declining while price is flat: the options market is
  quietly positioning for upside while price shows no movement
- Call spread volume increasing: directional bet without headline risk

*Data source: IBKR options chain (already wired in fetch_ibkr_option_chain.py),
Unusual Whales API, or Market Chameleon for historical OI series.*

**Dark Pool / Block Trade Activity**
- Dark pool prints at or above the ask (aggressive buying, not passive)
- Block trades (≥ 10,000 shares) printed on uptick during base formation
- Dark pool volume as % of total volume trending upward during base

*Data source: Quod Financial, Cboe LiveVol, or FINRA TRF data via Polygon.io*

**Integration Plan**
```python
@dataclass
class OrderFlowFeatures:
    unusual_call_oi: bool        # notable call OI buildup at OTM strikes
    put_call_ratio_trend: str    # "declining" | "rising" | "flat"
    dark_pool_aggressive: bool   # dark pool prints at/above ask
    block_trade_count: int       # block trades during base window
    order_flow_score: float      # composite 0.0–1.0
```
Add `order_flow_score` to `ScanResult` and weight +0.15 in `score_conviction()`
when available, 0.0 when data is absent (graceful degradation).

---

### Wyckoff Signal Integration into score_conviction()

When the Wyckoff module is complete, conviction scoring becomes:

```
Signal fired (breakout)        : 0.25  (reduced — Wyckoff quality replaces some weight)
Market regime (SPY > 200 SMA)  : 0.15
Wyckoff absorption score       : 0.10
Base quality (tight + long)    : 0.10
Volume character score         : 0.10
News catalyst quality          : 0.00–0.20  (earnings/management > upgrade > other)
Relative volume at breakout    : 0.05
Historical OOS edge (DuckDB)   : 0.05  (normalised from walk_forward_windows)
Order flow confirmation        : 0.00–0.15  (when data available)
                                 ────
Maximum score                  : 1.00
```

A Wyckoff spring present → automatic +0.10 bonus on top of `base_quality_score`.
A distribution signature detected → score capped at 0.30 regardless of other layers
(hard veto, same logic as `downgrade` news category).

---

### What a Perfect Setup Looks Like in Code

```python
# Fully confirmed Wyckoff breakout — all layers aligned
result = scan_symbol("AAPL", bars, signal_type="breakout", lookback=5)

# Expected state:
assert result.signal is True
assert result.regime_bullish is True
assert wyckoff.absorption_score(base_bars) > 0.65
assert wyckoff.base_quality_score(base_bars, min_weeks=12) > 0.70
assert wyckoff.volume_character_score(base_bars) > 0.55
assert wyckoff.is_wyckoff_spring(base_bars) is True  # bonus
assert result.rel_volume > 2.0                        # highest in base
assert result.news_category in {"earnings", "upgrade"}
assert result.conviction_score > 0.80
```

This is the entry checklist. Every layer that fails is a reason to reduce
size or pass entirely.

---

## Short Selling Framework

> **Status: not yet active.** `SHORT_SIGNAL_ENABLED = False` in `DEFAULT_CONFIG`.
> Build sequence is documented here so the architecture is decided before code
> is written. Activate only after the long side has been validated in paper
> trading for 3+ months with consistent positive expectancy.

Short selling is the mirror of the accumulation-breakout playbook. The edge
comes from the same source — identifying institutional positioning before the
move — but the direction is reversed: large holders *exiting* rather than
*entering*, and the price failing rather than succeeding.

---

### Short Signal Criteria

Five conditions must align. Any single condition alone is insufficient; the
high-conviction setup requires at least three, with the volume signature and
relative weakness as non-negotiable anchors.

**1. Breakdown below N-day low on above-average volume**

The structural entry trigger. Price violates the N-day range low (same
lookback as the long-side breakout, default 20 days) with volume exceeding
the 20-day average by at least 1.5×. A low-volume breakdown is a trap —
no conviction, likely to reverse. High-volume breakdown means institutional
selling, not just retail stops being hit.

```python
def breakdown_signal(bars, symbol, lookback=20, min_rel_volume=1.5):
    """Mirror of breakout_signal() — fires when price closes below N-day low
    on volume >= min_rel_volume * avg_volume_20d."""
    n_day_low = min(b.low for b in bars[-(lookback + 1):-1])
    rel_vol   = bars[-1].volume / mean(b.volume for b in bars[-21:-1])
    signal    = bars[-1].close < n_day_low and rel_vol >= min_rel_volume
    return SignalResult(signal=signal, signal_type="breakdown", ...)
```

**2. Wyckoff distribution signature**

Price rising on declining volume — the opposite of accumulation. The
composite operator is *offloading* into retail buying. Key indicators:

- Up-days on below-average volume (rallies have no institutional backing)
- Down-days on above-average volume (heavy supply on every pullback)
- `volume_character_score() < 0.40` — sustained over 40–60 bar window
- Multiple failed tests of the same resistance level (supply never absorbed)
- `absorption_score() < 0.40` on high-volume bars making new highs

A distribution top looks like a breakout at first. The tell is what happens
to volume *after* the breakout attempt: it collapses if real, or
the high-volume bars print on down-days if distribution.

**3. Earnings deceleration**

The fundamental anchor for a short thesis. Without deteriorating business
fundamentals, a declining stock can recover rapidly on any positive
re-rating. Required signals:

- Two or more consecutive EPS misses vs consensus estimate
- Revenue growth decelerating — not just slower, but trending toward zero
- Forward guidance cut — management lowering its own expectations
- Margin compression — gross or operating margin declining YoY
- Estimate revision trend: analysts *cutting* forward estimates, not raising

```python
@dataclass
class EarningsDeceleration:
    consecutive_misses:   int         # >= 2 required for short thesis
    revenue_growth_trend: str         # "decelerating" | "flat" | "negative"
    guidance_cut:         bool        # management lowered guidance
    margin_trend:         str         # "compressing" | "stable" | "expanding"
    estimate_revisions:   str         # "falling" | "flat" | "rising"
    deceleration_score:   float       # 0.0–1.0 composite
```

Data source: same as earnings acceleration — Polygon.io financials endpoint
or FactSet fundamentals. **This data is not yet available in the system.**
Without it, short signals carry significantly higher false-positive risk.

**4. Relative weakness vs SPY**

A stock must be *underperforming the market* to merit a short. Shorting a
stock that is merely not keeping up with a raging bull market is a losing
strategy — broad market strength can overwhelm even poor fundamentals.

- 63-day relative return: `stock_return_63d - spy_return_63d < -0.05`
  (underperforming SPY by at least 5% over the prior quarter)
- 126-day relative return: `stock_return_126d - spy_return_126d < -0.10`
  (underperforming SPY by at least 10% over the prior half-year)
- `rs_score() < 0.35` — sustained relative weakness, not a single bad week

Both timeframes must confirm. A stock weak over 63 days that is starting to
outperform over 126 days is potentially bottoming, not a short setup.

**5. Analyst downgrades with high IBKR C: confidence scores**

Analyst downgrades are already in the news pipeline (`classify_headline()`
→ `"downgrade"`). For the short side, high-conviction downgrades add weight:

- `news_category == "downgrade"` within the last 7 days
- `news_c_score >= 0.75` — IBKR's internal confidence score for the article
- Multiple downgrades from different firms within 14 days: sector rotation
  signal, not idiosyncratic noise
- Target price cuts alongside rating cuts: the analyst has recalibrated the
  fundamental model, not just reacted to price

Downgrades *already reduce* long-side conviction in `score_conviction()`.
On the short side, they add conviction rather than reduce it.

---

### Options-Based Institutional Exit Signals

Institutions cannot exit large positions by simply selling shares without
moving the market against themselves. They use options to hedge, accelerate,
or partially monetise their exposure. These signals are detectable in the
options market days or weeks before the price breakdown becomes obvious in
OHLCV bars.

**1. Deep ITM call selling (delta > 0.80)**

A large holder selling deep in-the-money calls is executing a synthetic
short hedge. The delta > 0.80 calls behave almost identically to short stock
— collecting premium while capping upside. This is the options market
fingerprint of an institution protecting a large long position from downside
without triggering a public 13F disclosure of a sale.

```
Signal: unusual OI or volume at call strikes where delta > 0.80
        (strikes significantly below current price for calls)
Threshold: OI > 5× the 20-day average OI at that strike
Interpretation: large holder hedging downside / synthetically exiting
```

Note: deep ITM *put* buying is a direct hedge. Deep ITM *call selling* is
the more subtle and information-rich signal because it requires the seller to
hold the underlying and is typically done by a large long reducing risk.

**2. Negative call skew (deep ITM call IV > ATM call IV)**

Normal call skew is negative: ATM calls trade at higher IV than deep ITM
calls because ATM options embed the most uncertainty. When this skew
*inverts* — deep ITM call IV exceeds ATM call IV — it signals that someone
is paying a premium to hedge existing long exposure at the deep ITM strikes.

```
Negative call skew = IV(delta=0.85 call) > IV(delta=0.50 call)
Normal skew:    IV(0.85) < IV(0.50)   → no institutional hedging signal
Inverted skew:  IV(0.85) > IV(0.50)   → paid hedging, exit signal
```

This is the options market equivalent of distribution in the OHLCV data —
the institutional fingerprint of a large holder protecting gains without
visibly selling stock.

**3. OI accumulation at deep ITM call strikes**

A single day of unusual volume at deep ITM calls can be noise — a covered
call writer, a structured product unwind. Sustained OI *building over
multiple sessions* at the same strikes is the institutional signal:

```
Signal: OI at delta > 0.80 call strikes growing over 5+ consecutive sessions
        without corresponding price decline (hedging, not closing)
Threshold: OI growth > 20% over 5 sessions at the target strike cluster
Interpretation: sustained hedging campaign, not a one-day event
```

OI that builds then disappears in a single session is likely market-maker
hedging of a customer order. OI that accumulates and persists is a strategic
position.

**4. Combined high-conviction short setup**

The highest-conviction short signal requires all three options layers to
confirm simultaneously with the fundamental and price signals:

```
DISTRIBUTION VOLUME      → volume_character_score() < 0.40
  +
DEEP ITM CALL SELLING   → unusual OI at delta > 0.80 calls, sustained 5+ days
  +
EARNINGS DECELERATION   → consecutive_misses >= 2, guidance cut = True
  +
RELATIVE WEAKNESS       → rs_score() < 0.35, underperforming SPY 63d and 126d
─────────────────────────────────────────────────────────────────────────────
= HIGH CONVICTION SHORT SETUP (target conviction_score_short >= 0.70)
```

No single factor is sufficient. The convergence of price structure
(distribution), positioning (institutional hedging via deep ITM calls), and
fundamentals (earnings deceleration) is what separates a tradeable short
from a stock that is merely having a bad week.

**5. Data requirements**

The full options signal stack requires historical options data that the
current IBKR integration does not provide at scale:

- **FactSet Options API** — real-time and historical Greeks, OI, IV surface.
  Required for the IV skew calculation and sustained OI monitoring.
- **Theta Data** — tick-level options data with historical OI series.
  Lower cost than FactSet, adequate for backtesting the deep ITM signal.
- **IBKR options chain** (`fetch_ibkr_option_chain.py`) — already wired for
  single-symbol snapshots. Not sufficient for universe-scale screening.

Until FactSet or Theta Data is integrated, the deep ITM call signals cannot
be computed. The framework is documented here to define what to build.

---

### Regime-Aware Shorting

The tape regime (from `signals/breadth.py`) gates which direction the
scanner operates in:

| Tape | Long side | Short side |
|------|-----------|------------|
| BULL | Scan longs, full universe | Long-only mode — no shorts |
| NEUTRAL | Scan longs, reduced universe | Short candidates added alongside longs |
| BEAR | Selective longs (conviction >= 0.80) | Short candidates are primary focus |

In BULL tape, adding short exposure is fighting the market's primary trend.
The expected value of a short in a broad bull market is negative even when
the individual stock thesis is correct — rising tide lifts most boats and
short squeezes are frequent.

In BEAR tape (McClellan oscillator < −100, or fewer than 200 stocks in the
universe at 52-week highs), the scanner flips to adding short candidates
alongside any remaining high-conviction longs:

```python
# Pseudocode — not yet implemented (SHORT_SIGNAL_ENABLED = False)
if breadth_snap.tape == "BEAR" and SHORT_SIGNAL_ENABLED:
    short_candidates = scan_for_breakdowns(
        symbols, min_conviction=0.65,
        require_earnings_decel=True,
        require_relative_weakness=True,
    )
    results = long_candidates + short_candidates
```

The `SHORT_SIGNAL_ENABLED` flag in `DEFAULT_CONFIG["scanner"]` is the
master switch. It stays `False` until the long side is validated and the
short infrastructure (earnings deceleration data, options OI monitoring)
is in place.

---

### Risk Differences vs Long Positions

Short selling carries structurally different risks that require separate
risk management rules. The long-side kill switches and position sizing
module must be fully operational before any short is placed.

**Asymmetric risk**
A long position can lose at most 100% of capital invested (the stock goes to
zero). A short position has theoretically unlimited loss — a stock can rise
10×, 50×, or more. This is not a theoretical risk; it has destroyed accounts
in NVDA (2023), GME (2021), and dozens of other squeeze events.
Risk per short trade must be sized at 50% of the equivalent long position
size until the short side is validated with a minimum 30-trade sample.

**Borrow availability and cost**
Short selling requires borrowing shares from a broker. Hard-to-borrow stocks
carry daily fees that can exceed the trade's expected return. IBKR provides
borrow rate data via the Short Stock (SLB) indicator. Before entering any
short, confirm:
- Borrow is available (not "HTB" — hard to borrow)
- Borrow rate < 5% annualised (above this, the carry cost erodes the edge)
- Float is large enough that a recall is unlikely within the holding period

**The uptick rule**
SEC Rule 10a-1 (modified uptick rule) restricts short sales when a stock has
declined more than 10% in a single day. This can prevent entry at the
intended price on a strong breakdown day — the most attractive short entry
point. Build execution logic to check for uptick rule restrictions before
submitting short orders.

**Margin requirements**
Reg T requires 150% of the short position value to be held in the account
(100% proceeds + 50% margin). Portfolio margin accounts reduce this, but
during high-volatility periods brokers can increase margin requirements
without notice. The position sizing module must account for the higher
capital reservation per short.

**Never short into earnings**
Earnings are the highest-risk event for a short position. A company with
every short signal aligned can report one better-than-feared quarter and
gap up 15–25% overnight, hitting the stop at a loss equivalent to several
months of expected edge. The rule is absolute:
- Check earnings date before entering any short
- Do not initiate a short position within 5 trading days of a scheduled
  earnings release
- If already short, reduce to half-size or exit before the announcement

---

### Build Sequence

Short selling is Phase 8+, after the long side is operationally mature.
Each prerequisite must be complete before proceeding to the next:

```
Prerequisites (must all be green before writing short-selling code):

  [Phase 5] Polygon.io fundamentals client — EarningsProfile data available
  [Phase 6] Live paper trading (longs) — IBKR order submission working
  [Phase 6] Position sizing module — Kelly / fixed fractional implemented
  [Phase 6] Portfolio-level risk limits — max drawdown kill switch live
  [Phase 7] 3+ months paper trading (longs) — positive expectancy confirmed
             Minimum 30 completed long trades with documented outcomes

Short selling build sequence (Phase 8):

  [ ] breakdown_signal() — mirror of breakout_signal(), price + volume
  [ ] EarningsDeceleration dataclass — fetch from Polygon/FactSet
  [ ] rs_score() short threshold — rs_score() < 0.35 filter
  [ ] Borrow availability check — IBKR SLB rate query before entry
  [ ] Uptick rule detection — check SSR flag on entry bar
  [ ] Short-side conviction scorer — separate weights from long scorer
  [ ] Regime gate — SHORT_SIGNAL_ENABLED = True only when BEAR tape

Short options signals (Phase 9 — requires FactSet or Theta Data):

  [ ] IV surface loader — historical Greeks per strike per expiry
  [ ] Deep ITM call OI monitor — flag sustained OI buildup at delta > 0.80
  [ ] Call skew inversion detector — IV(0.85Δ) > IV(0.50Δ)
  [ ] Combined signal aggregator — distribution + deep ITM calls + decel
```

The key principle: **prove the long side works first**. Short selling
amplifies both the return and the operational complexity. A system that
cannot consistently identify long-side breakouts will not consistently
identify short-side breakdowns. The signals are mirrors of each other;
the edge must be demonstrated in one direction before being extended to both.

