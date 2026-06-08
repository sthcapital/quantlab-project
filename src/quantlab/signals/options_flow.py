"""
quantlab.signals.options_flow — Options market conviction signals.

Detects institutional positioning from option chain data without tick-level
order flow access.  All analytical functions accept a ChainData object so
they are pure and fully testable without an IBKR connection.

Data flow:
    1. fetch_chain_data(ib, symbol, spot)  →  ChainData
    2. put_call_ratio(chain)               →  float
       unusual_call_activity(chain)        →  (bool, float)
       iv_skew_score(chain)                →  float
    3. compute_options_score(chain)        →  float  (0.0–1.0)
    4. options_conviction_score(sym, bars, ib)  →  float  (convenience wrapper)

Notes on data availability:
    - Volume (optVolume):  available via reqTickers with delayed data.
    - Open interest:       requires specific generic tick subscription
                           (tick type 101); we fall back to volume where OI
                           is absent, since daily volume is a good proxy for
                           short-term activity.
    - Implied volatility:  from modelGreeks; available with delayed data (type 3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from math import fabs
from typing import Sequence


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class OptionContract:
    """A single option contract with market data."""

    strike: float
    right: str               # "C" (call) or "P" (put)
    expiry: str              # YYYYMMDD

    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    mid: float | None = None
    volume: float | None = None        # today's optVolume; proxy for OI when absent
    open_interest: float | None = None # tick-type 101 if subscribed; else None
    implied_vol: float | None = None   # from modelGreeks.impliedVol
    delta: float | None = None

    @property
    def activity(self) -> float:
        """Best available measure of market activity for this contract."""
        return self.open_interest or self.volume or 0.0


@dataclass
class ChainData:
    """Snapshot of an option chain for a single underlying."""

    symbol: str
    spot: float
    expiry: str
    contracts: list[OptionContract] = field(default_factory=list)

    @property
    def calls(self) -> list[OptionContract]:
        return [c for c in self.contracts if c.right == "C"]

    @property
    def puts(self) -> list[OptionContract]:
        return [c for c in self.contracts if c.right == "P"]


# ── 1. Put/call ratio ──────────────────────────────────────────────────────────

def put_call_ratio(
    chain: ChainData,
    atm_band_pct: float = 0.05,
) -> float:
    """
    Put activity / call activity at ATM strikes (within ±atm_band_pct of spot).

    Uses open interest when available; falls back to volume.  In both cases a
    lower ratio means more calls than puts, signalling bullish market positioning.

    Interpretation:
        < 0.50  — strongly bullish (calls heavily outnumber puts)
        < 0.70  — moderately bullish
        0.70–1.20 — neutral / balanced
        > 1.20  — bearish / heavy hedging demand

    Args:
        chain:        ChainData snapshot.
        atm_band_pct: Fraction of spot to define "ATM".  Default ±5%.

    Returns:
        Ratio ≥ 0.  Returns 1.0 (neutral) when call activity is zero.
    """
    lo = chain.spot * (1.0 - atm_band_pct)
    hi = chain.spot * (1.0 + atm_band_pct)

    call_activity = sum(
        c.activity for c in chain.calls if lo <= c.strike <= hi
    )
    put_activity = sum(
        c.activity for c in chain.puts if lo <= c.strike <= hi
    )

    if call_activity <= 0:
        return 1.0  # neutral — no call data

    return round(put_activity / call_activity, 4)


# ── 2. Unusual call activity ───────────────────────────────────────────────────

def unusual_call_activity(
    chain: ChainData,
    avg_volume_threshold: float = 2.0,
) -> tuple[bool, float]:
    """
    Detect call volume running above N times the per-strike average.

    A single strike showing 2× or more the average call volume across all
    strikes may indicate targeted institutional accumulation of calls.

    Args:
        chain:                  ChainData snapshot.
        avg_volume_threshold:   Multiple above average to qualify (default 2×).

    Returns:
        (is_unusual, ratio) where ratio = max_call_volume / avg_call_volume.
        ratio < 1 when fewer than 2 call strikes have volume data.
    """
    call_vols = [
        c.volume for c in chain.calls
        if c.volume is not None and c.volume > 0
    ]

    if len(call_vols) < 2:
        return False, 0.0

    avg_vol = sum(call_vols) / len(call_vols)
    max_vol = max(call_vols)

    if avg_vol <= 0:
        return False, 0.0

    ratio = round(max_vol / avg_vol, 4)
    return ratio >= avg_volume_threshold, ratio


# ── 3. IV skew score ───────────────────────────────────────────────────────────

def iv_skew_score(
    chain: ChainData,
    otm_pct: float = 0.05,
) -> float:
    """
    Score the relative expensiveness of OTM calls vs OTM puts.

    In normal markets, OTM puts trade at a premium (volatility smirk) because
    institutions buy downside protection.  When OTM calls become relatively
    MORE expensive, it indicates directional bullish positioning — smart money
    paying up for upside exposure.

    Algorithm:
        avg_call_iv = mean IV of calls with strike > spot × (1 + otm_pct)
        avg_put_iv  = mean IV of puts  with strike < spot × (1 − otm_pct)
        skew_ratio  = avg_call_iv / avg_put_iv
        score       = 0.5 + 0.5 × tanh((skew_ratio − 1) × 2)

    Returns 0.5 when data is absent (neutral) or at parity.

    Args:
        chain:   ChainData snapshot.
        otm_pct: Minimum distance from spot to qualify as "OTM" (default 5%).

    Returns:
        Float in [0.0, 1.0].  > 0.5 = calls expensive relative to puts = bullish.
    """
    otm_call_ivs = [
        c.implied_vol
        for c in chain.calls
        if c.strike > chain.spot * (1.0 + otm_pct)
        and c.implied_vol is not None and c.implied_vol > 0
    ]
    otm_put_ivs = [
        c.implied_vol
        for c in chain.puts
        if c.strike < chain.spot * (1.0 - otm_pct)
        and c.implied_vol is not None and c.implied_vol > 0
    ]

    if not otm_call_ivs or not otm_put_ivs:
        return 0.5  # neutral when data is absent

    avg_call_iv = sum(otm_call_ivs) / len(otm_call_ivs)
    avg_put_iv  = sum(otm_put_ivs)  / len(otm_put_ivs)

    if avg_put_iv <= 0:
        return 0.5

    skew_ratio = avg_call_iv / avg_put_iv
    # tanh maps (−∞,+∞) → (−1,+1); shift to [0,1]
    score = 0.5 + 0.5 * math.tanh((skew_ratio - 1.0) * 2.0)
    return round(max(0.0, min(1.0, score)), 4)


# ── 4. Composite options score ─────────────────────────────────────────────────

def compute_options_score(chain: ChainData) -> float:
    """
    Combine PCR, unusual call activity, and IV skew into a 0.0–1.0 score.

    Scoring weights:
        PCR < 0.50 (strongly bullish)  : +0.60
        PCR < 0.70 (moderately bullish): +0.40
        Unusual call activity           : +0.25
        IV skew > 0.60 (calls pricey)  : +0.15

    Maximum raw score = 1.00; clamped to 1.0.

    The score feeds into score_conviction() in execution/__init__.py:
        ≥ 0.60 → +0.10 conviction;  ≥ 0.80 → +0.15 conviction.

    Args:
        chain: Pre-fetched ChainData.

    Returns:
        Float in [0.0, 1.0].
    """
    score = 0.0

    pcr = put_call_ratio(chain)
    if pcr < 0.50:
        score += 0.60
    elif pcr < 0.70:
        score += 0.40

    is_unusual, _ = unusual_call_activity(chain)
    if is_unusual:
        score += 0.25

    skew = iv_skew_score(chain)
    if skew > 0.60:
        score += 0.15

    return round(min(1.0, score), 4)


# ── 5. IBKR fetch + master function ───────────────────────────────────────────

def fetch_chain_data(
    ib,
    symbol: str,
    spot: float,
    expiry: str | None = None,
    n_strikes: int = 5,
) -> ChainData | None:
    """
    Fetch a near-term option chain from IBKR and return a ChainData object.

    Requires an active IB() connection.  Uses delayed market data (type 3)
    so no live subscription is needed.  Returns None on any error so the
    caller can fall back gracefully.

    Args:
        ib:        Active IB() instance.
        symbol:    Ticker symbol (e.g. "AAPL").
        spot:      Current spot price (use bars[-1].close to avoid extra fetch).
        expiry:    Specific expiry (YYYYMMDD).  Defaults to the 3rd nearest.
        n_strikes: Number of strikes nearest ATM to include per side.

    Returns:
        ChainData or None on fetch failure.
    """
    try:
        from ib_insync import Stock, Option

        stock = Stock(symbol, "SMART", "USD")
        qualified = ib.qualifyContracts(stock)
        if not qualified:
            return None
        stock = qualified[0]

        chains = ib.reqSecDefOptParams(
            underlyingSymbol=symbol,
            futFopExchange="",
            underlyingSecType=stock.secType,
            underlyingConId=stock.conId,
        )
        if not chains:
            return None

        chain_def = next((c for c in chains if c.exchange == "SMART"), chains[0])
        today = datetime.utcnow().strftime("%Y%m%d")
        valid_expiries = sorted(e for e in chain_def.expirations if e >= today)
        if not valid_expiries:
            return None

        exp = expiry or valid_expiries[min(2, len(valid_expiries) - 1)]

        strikes = sorted(
            float(s) for s in chain_def.strikes
            if 0.6 * spot <= float(s) <= 1.4 * spot
        )
        nearest = sorted(strikes, key=lambda x: fabs(x - spot))[:n_strikes]
        nearest = sorted(nearest)

        # Qualify contracts
        raw_contracts: list[tuple] = []
        for strike in nearest:
            for right in ("C", "P"):
                opt = Option(
                    symbol, exp, strike, right, "SMART",
                    tradingClass=chain_def.tradingClass,
                )
                qualified_opt = ib.qualifyContracts(opt)
                if qualified_opt:
                    raw_contracts.append((qualified_opt[0], strike, right))

        if not raw_contracts:
            return None

        ib.reqMarketDataType(3)
        ibkr_contracts = [c[0] for c in raw_contracts]
        tickers = ib.reqTickers(*ibkr_contracts)

        option_contracts: list[OptionContract] = []
        for (ibkr_c, strike, right), ticker in zip(raw_contracts, tickers):
            def _clean(v) -> float | None:
                return float(v) if v is not None and v == v and v > 0 else None

            bid = _clean(ticker.bid)
            ask = _clean(ticker.ask)
            mid = (bid + ask) / 2.0 if bid and ask else None

            greeks = ticker.modelGreeks
            iv    = _clean(greeks.impliedVol) if greeks else None
            delta = (float(greeks.delta) if greeks and greeks.delta == greeks.delta
                     else None) if greeks else None

            volume = _clean(getattr(ticker, "optVolume", None))
            oi     = _clean(getattr(ticker, "openInterest", None))

            option_contracts.append(OptionContract(
                strike=strike, right=right, expiry=exp,
                bid=bid, ask=ask,
                last=_clean(ticker.last),
                mid=mid,
                volume=volume,
                open_interest=oi,
                implied_vol=iv,
                delta=delta,
            ))

        return ChainData(
            symbol=symbol, spot=spot, expiry=exp,
            contracts=option_contracts,
        )

    except Exception:
        return None


def options_conviction_score(
    symbol: str,
    bars: Sequence,
    ib_connection,
) -> float:
    """
    Deprecated — MassiveOptionsProvider (Polygon S3) is the sole options data
    source for scanning.  TWS is reserved for news and execution only.

    Returns 0.5 (neutral) unconditionally so callers that still reference this
    function get a no-op score rather than an IBKR TWS call.  Use
    MassiveOptionsProvider.compute_options_score() for live options scoring.
    """
    return 0.5
