"""
Layer 7: Execution infrastructure — market scanner and conviction scoring.

The scanner is the missing piece identified in the architecture review.
It loops over a symbol universe, runs signal checks on each, layers
confirmation signals (news, regime), scores conviction, and returns
a ranked list of setups ready for the risk gate.

ZeroMQ pub/sub bus is stubbed here for Phase 5+ — the scanner publishes
signals, the execution subscriber receives and validates before order submission.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Sequence

from quantlab.providers.base import Bar, MarketDataProvider
from quantlab.signals import (
    SignalResult,
    breakout_signal,
    sma_signal,
    regime_is_bullish,
    atr_stop_price,
)

logger = logging.getLogger(__name__)


# ── Low-edge symbol list ───────────────────────────────────────────────────────
# Symbols that showed negative full-period Sharpe on real IBKR data
# (2024-01-02 → 2025-12-31, breakout lookback=5).  Scanner warns when these
# appear in results so the user can apply extra scrutiny before acting.
LOW_EDGE_SYMBOLS: frozenset[str] = frozenset({
    "BAC",   # explicitly flagged; also negative OOS across windows
    "PG",    # Sharpe -4.27, avg OOS Sharpe -8.99 — worst in universe
    "AMGN",  # Sharpe -2.85, avg OOS Sharpe -2.70
    "NEE",   # Sharpe -2.70, avg OOS Sharpe -3.93
    "AMZN",  # Sharpe -2.39, avg OOS Sharpe -2.49
    "PEP",   # Sharpe -2.16, avg OOS Sharpe -2.44
    "CVX",   # Sharpe -1.84, avg OOS Sharpe -4.34
    "META",  # Sharpe -1.60, avg OOS Sharpe -1.61
    "KO",    # avg OOS Sharpe -7.28 despite mild full-period Sharpe
    "MCD",   # avg OOS Sharpe -3.87
    "MA",    # avg OOS Sharpe -3.35
})


# ── News category weights ──────────────────────────────────────────────────────
# Derived from DuckDB trade-level analysis (2024–2025 live IBKR run):
#   earnings avg_ret=+0.32%  management=+0.55%  upgrade=+0.09%
#   analyst_action=+0.04%   downgrade=-0.17%   other=-0.59%
NEWS_CATEGORY_WEIGHTS: dict[str, float] = {
    "earnings":       +0.20,   # strong positive catalyst
    "management":     +0.20,   # strong positive catalyst
    "upgrade":        +0.08,   # weak positive
    "analyst_action": +0.05,   # marginal — don't over-weight generic notes
    "downgrade":      -0.15,   # veto signal: reduces conviction
    "other":          +0.00,   # noise: ignore
    "none":           +0.00,
}


# ── Conviction scoring ─────────────────────────────────────────────────────────

@dataclass
class ScanResult:
    """
    Output of the scanner for a single symbol.
    Carries everything needed for the risk gate decision.
    """

    symbol: str
    scan_date: str
    signal_type: str
    signal: bool
    entry_close: float
    indicator_value: float | None
    lookback: int

    # Conviction layers — market / news
    regime_bullish: bool = True
    news_count: int = 0
    news_category: str = "none"
    news_k_score: float | None = None
    news_c_score: float | None = None
    rel_volume: float | None = None
    atr_stop: float | None = None

    # Wyckoff structural layers (computed from bar history at scan time)
    base_quality: float = 0.0     # base_quality_score()    ≥ 0.6 → +0.15 (disabled)
    absorption: float = 0.0       # absorption_score()      ≥ 0.6 → +0.05 (reduced: 100% fire rate on daily bars)
    volume_character: float = 0.0 # volume_character_score() ≥ 0.6 → +0.10
    wyckoff_spring: bool = False   # is_wyckoff_spring()         True → +0.10

    # Earnings acceleration layer (detected from price/volume anomalies)
    earnings_acceleration: float = 0.0  # earnings_acceleration_score() ≥ 0.5 → +0.10

    # Institutional volume signature layers
    accumulation_ratio: float = 0.0    # accumulation_days_ratio()  ≥ 0.6 → +0.08
    volume_trend: float = 0.0          # volume_trend_score()       (informational)
    climactic_volume: float = 0.0      # climactic_volume_score()   ≥ 0.7 → +0.07

    # Options flow conviction (IBKR chain; enriched post-scan)
    options_conviction: float = 0.0  # IBKR source: ≥ 0.6 → +0.10; ≥ 0.8 → +0.15

    # Polygon/Massive options score (preferred over IBKR options_conviction when > 0)
    options_score: float = 0.0       # MassiveOptionsProvider.compute_options_score()

    # Multi-lookback confirmation (set post-scan when signal fires at ≥2 lookbacks)
    multi_lookback_confirmed: bool = False  # True → +0.05 structural confirmation bonus

    # Relative strength vs market benchmark (SPY)
    rs_score: float = 0.0   # rs_score() ≥ 0.6 → +0.08; ≥ 0.8 → +0.12 (replaces lower)

    # Sector metadata (from SECTOR_MAP; used by sector_filter)
    sector: str = ""               # GICS sector (e.g. "Health Care", "Technology")
    sector_cluster: bool = False   # True when ≥3 same-sector signals on same day

    # Breadth regime adjustment (populated from latest DuckDB breadth_history)
    breadth_regime_adj: float = 0.0  # -0.12 to 0.0 depending on tape
    breadth_override: bool = False   # True → hard veto (McClellan<-100 or bear)

    # Weinstein/Minervini stage classification (set by scan_symbol from bar history)
    # 1=Basing  2=Advancing (long-entry candidates only)  3=Topping  4=Declining  0=Unknown
    stage: int = 0

    # Breakout volume quality — Weinstein's 2× rule (set by scan_symbol)
    breakout_volume_score: float = 0.0   # volume_on_breakout_score() ≥ 0.7 → +0.08

    # PEG ratio score — Boucher's filter (populated by run_universe_scan via EDGAR cache)
    peg_score: float = 0.0   # peg_ratio_score() ≥ 0.7 → +0.06

    # EDGAR fundamentals (populated by run_universe_scan; None = unavailable)
    edgar_acceleration: float | None = None  # real score; falls back to earnings_acceleration

    # Unusual options activity (mid-cap; populated via --with-options flat-file path)
    unusual_options_score: float = 0.0  # score_unusual_activity(); 0.0 = not computed
    market_cap_tier: str = ""           # "mega_cap"|"large_cap"|"mid_cap" — set at scan time

    # Macro regime (populated from FRED + CBOE by run_universe_scan)
    macro_regime: str = "risk_on"   # "risk_on" | "risk_off" | "stress"
    vix_regime: str = "low"         # "low" | "elevated" | "high" | "extreme"

    # Earnings calendar proximity (set by run_universe_scan via EDGAR)
    # "pre_earnings"        — next earnings within 5 trading days  (risk: gap risk)
    # "post_earnings_beat"  — last earnings within 5 trading days, beat  (momentum)
    # "post_earnings_miss"  — last earnings within 5 trading days, miss  (headwind)
    # "neutral"             — no near-term earnings event
    earnings_proximity: str = "neutral"

    # Computed conviction score (0.0 – 1.0)
    conviction_score: float = 0.0

    def is_actionable(self, min_conviction: float = 0.4) -> bool:
        return self.signal and self.conviction_score >= min_conviction


def score_conviction(result: ScanResult) -> float:
    """
    Score 0.0–1.0 based on all active confirmation layers.

    Layer weights (max 1.0, clamped):
        Signal fired               : 0.30  (mandatory — returns 0 if no signal)
        Regime bullish             : 0.20
        News (category-weighted)   : see NEWS_CATEGORY_WEIGHTS
                                       earnings/management → +0.20
                                       upgrade             → +0.08
                                       analyst_action      → +0.05
                                       downgrade           → −0.15 (veto)
                                       other               →  0.00
        Rel volume ≥ 1.5×          : 0.10
        Strong news c_score ≥ 0.7  : 0.10
        Wyckoff absorption ≥ 0.6        : 0.05  (reduced from 0.10; fires on 100%
                                                  of daily-bar signals → too permissive;
                                                  needs intraday data to discriminate)
        Wyckoff vol character ≥ 0.6     : 0.10
        Wyckoff spring detected         : 0.10
        Earnings acceleration ≥ 0.5     : 0.10  (EDGAR score preferred; OHLCV fallback)
        Accumulation days ratio ≥ 0.6   : 0.08
        Climactic volume ≥ 0.7          : 0.07
        Multi-lookback confirmed        : 0.05  (signal fires at ≥2 lookback values)
        Options (tier-aware):
          mid_cap unusual ≥ 0.7       : 0.15  (institutional call spike at 5×+ avg vol)
          mid_cap unusual ≥ 0.5       : 0.08
          mega/large_cap PCR/IV ≥ 0.8 : 0.15  (Polygon options_score; IBKR fallback)
          mega/large_cap PCR/IV ≥ 0.6 : 0.10
          small_cap                   : 0.00  (options too illiquid)
        RS score ≥ 0.6 (outperforming) : 0.08
        RS score ≥ 0.8 (leader)        : 0.12  (replaces the 0.08)
        Breadth regime adjustment       : 0.00 to -0.12 (from 10-day ratio)
        Breadth override                : returns 0.0 immediately (bear market veto)
        Macro regime (FRED+CBOE)        : 0.00 (risk_on), -0.05 (risk_off), -0.10 (stress)
        Stage 2 confirmation            : +0.05  (Weinstein advancing stage only)
        Breakout volume ≥ 2× avg        : +0.08  (Weinstein valid breakout — score ≥ 0.7)
        PEG ratio < 1.0                 : +0.06  (Boucher fairly-valued relative to growth)

    Note: base_quality_score() is intentionally excluded from this scorer.
    Live AAPL analysis (82 signals, 2023–2025) showed base quality is
    anti-predictive for mega-cap large-caps: BQ≥0.6 win rate 26.9% vs
    62.5% for plain signals. Use base_quality_score() as a standalone
    diagnostic tool via analyse_wyckoff_filter.py, not as a scorer input,
    until validated on mid-cap growth names.

    Downgrade news reduces conviction; result is clamped to [0.0, 1.0].
    """
    if not result.signal:
        return 0.0

    # Hard bear-market veto (McClellan < -100 or up_25pct_quarter < 200)
    if result.breadth_override:
        return 0.0

    score = 0.30  # base: signal fired

    if result.regime_bullish:
        score += 0.20

    if result.news_count > 0:
        score += NEWS_CATEGORY_WEIGHTS.get(result.news_category, 0.0)

    if result.rel_volume is not None and result.rel_volume >= 1.5:
        score += 0.10

    if result.news_c_score is not None and result.news_c_score >= 0.7:
        score += 0.10

    # Wyckoff structural confirmation
    if result.absorption >= 0.6:
        score += 0.05   # reduced: daily-bar absorption fires on ~100% of signals
    if result.volume_character >= 0.6:
        score += 0.10
    if result.wyckoff_spring:
        score += 0.10

    # Earnings acceleration — EDGAR-based when available, else OHLCV inferred
    _accel = (
        result.edgar_acceleration
        if result.edgar_acceleration is not None
        else result.earnings_acceleration
    )
    if _accel >= 0.5:
        score += 0.10

    # Institutional volume signature
    if result.accumulation_ratio >= 0.6:
        score += 0.08
    if result.climactic_volume >= 0.7:
        score += 0.07

    # Multi-lookback structural confirmation
    if result.multi_lookback_confirmed:
        score += 0.05

    # Options — tier-aware routing
    _tier = result.market_cap_tier or market_cap_tier(result.symbol)
    if _tier == "mid_cap" and result.unusual_options_score > 0:
        # Mid-cap: unusual volume spike is the primary options signal
        if result.unusual_options_score >= 0.7:
            score += 0.15
        elif result.unusual_options_score >= 0.5:
            score += 0.08
    elif _tier != "small_cap":
        # mega_cap / large_cap (or mid_cap without unusual data): PCR + IV skew
        _opt = result.options_score if result.options_score > 0 else result.options_conviction
        if _opt >= 0.8:
            score += 0.15
        elif _opt >= 0.6:
            score += 0.10
    # small_cap: 0 options contribution (options market too thin)

    # Relative strength vs market benchmark (SPY)
    if result.rs_score >= 0.8:
        score += 0.12
    elif result.rs_score >= 0.6:
        score += 0.08

    # Breadth regime adjustment (0.0 in bull, negative in weak/bear tape)
    score += result.breadth_regime_adj

    # Macro regime adjustment (FRED yield spreads + CBOE VIX)
    if result.macro_regime == "stress":
        score -= 0.10
    elif result.macro_regime == "risk_off":
        score -= 0.05

    # Earnings calendar proximity adjustment
    if result.earnings_proximity == "pre_earnings":
        score -= 0.10   # gap risk — avoid entering before earnings
    elif result.earnings_proximity == "post_earnings_beat":
        score += 0.10   # momentum continuation after beat
    elif result.earnings_proximity == "post_earnings_miss":
        score -= 0.05   # headwind after miss

    # Weinstein stage confirmation — Stage 2 only for long entries
    if result.stage == 2:
        score += 0.05

    # Breakout volume quality — Weinstein's 2× average minimum for valid breakout
    if result.signal_type == "breakout" and result.breakout_volume_score >= 0.7:
        score += 0.08

    # PEG ratio — Boucher's fairly-valued-relative-to-growth filter
    if result.peg_score >= 0.7:
        score += 0.06

    return max(0.0, min(score, 1.0))


def historical_edge_score(
    symbol: str,
    db_path: str | None = None,
    clip_lo: float = -10.0,
    clip_hi: float = 10.0,
) -> float:
    """
    Return a 0.0–1.0 score reflecting a symbol's proven out-of-sample edge.

    Queries the walk_forward_windows DuckDB table for the symbol's average OOS
    Sharpe ratio, clips to [clip_lo, clip_hi] to prevent mock-data outliers from
    distorting the result, then normalises linearly:

        score = (clipped_avg_oos - clip_lo) / (clip_hi - clip_lo)

    Examples with default clip [-10, 10]:
        avg OOS Sharpe  +4.0  →  score 0.70  (strong proven edge)
        avg OOS Sharpe   0.0  →  score 0.50  (neutral / breakeven)
        avg OOS Sharpe  -4.0  →  score 0.30  (penalised)
        avg OOS Sharpe -72.0  →  score 0.00  (clipped to floor)

    Returns 0.5 (neutral) when the DB is unavailable or the symbol has no data.

    Args:
        symbol:  Ticker symbol to look up.
        db_path: Override DB path (defaults to quantlab.duckdb project location).
        clip_lo: Lower clip bound for OOS Sharpe (default −10).
        clip_hi: Upper clip bound for OOS Sharpe (default +10).
    """
    try:
        import duckdb
        from quantlab.storage import DB_PATH
        path = db_path or str(DB_PATH)
        con = duckdb.connect(path)
        row = con.execute(
            "SELECT AVG(oos_sharpe) FROM walk_forward_windows "
            "WHERE symbol = ? AND oos_sharpe IS NOT NULL",
            [symbol],
        ).fetchone()
        con.close()

        if row is None or row[0] is None:
            return 0.5

        avg_oos = float(row[0])
        clipped = max(clip_lo, min(clip_hi, avg_oos))
        return round((clipped - clip_lo) / (clip_hi - clip_lo), 4)

    except Exception:
        return 0.5  # neutral when DB is absent or query fails


# ── Universe management ────────────────────────────────────────────────────────

# Default watchlists — will grow as the system matures
SP500_SAMPLE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK B",
    "UNH", "LLY", "JPM", "V", "XOM", "MA", "AVGO", "PG", "HD", "CVX",
    "MRK", "COST", "ABBV", "KO", "PEP", "BAC", "ADBE", "TMO", "WMT",
    "ACN", "MCD", "CSCO", "ABT", "CRM", "NFLX", "DHR", "ORCL", "NKE",
    "LIN", "TXN", "NEE", "RTX", "BMY", "AMGN", "UPS", "HON", "PM",
    "INTC", "QCOM", "IBM", "CAT", "GS",
]

WATCHLIST_SMALL = ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "GOOGL", "META"]

# ── GICS sector map ────────────────────────────────────────────────────────────
# Maps every SP500_SAMPLE symbol to its GICS sector.
# Used by sector_filter() to detect and penalise same-day sector clustering.

SECTOR_MAP: dict[str, str] = {
    # Information Technology (13)
    "AAPL": "Technology",  "MSFT": "Technology",  "NVDA": "Technology",
    "AVGO": "Technology",  "ADBE": "Technology",  "ACN":  "Technology",
    "CRM":  "Technology",  "CSCO": "Technology",  "ORCL": "Technology",
    "TXN":  "Technology",  "INTC": "Technology",  "QCOM": "Technology",
    "IBM":  "Technology",
    # Consumer Discretionary (6)
    "AMZN": "Consumer Discretionary",  "TSLA": "Consumer Discretionary",
    "HD":   "Consumer Discretionary",  "MCD":  "Consumer Discretionary",
    "NKE":  "Consumer Discretionary",  "NFLX": "Consumer Discretionary",
    # Communication Services (2)
    "GOOGL": "Communication Services",  "META": "Communication Services",
    # Health Care (9)
    "UNH":  "Health Care",  "LLY":  "Health Care",  "MRK":  "Health Care",
    "ABBV": "Health Care",  "TMO":  "Health Care",  "DHR":  "Health Care",
    "ABT":  "Health Care",  "AMGN": "Health Care",  "BMY":  "Health Care",
    # Financials (6)
    "BRK B": "Financials",  "JPM": "Financials",  "V":   "Financials",
    "MA":    "Financials",  "BAC": "Financials",  "GS":  "Financials",
    # Energy (2)
    "XOM": "Energy",  "CVX": "Energy",
    # Consumer Staples (6)
    "PG":   "Consumer Staples",  "KO":   "Consumer Staples",
    "PEP":  "Consumer Staples",  "COST": "Consumer Staples",
    "WMT":  "Consumer Staples",  "PM":   "Consumer Staples",
    # Industrials (4)
    "HON": "Industrials",  "RTX": "Industrials",
    "UPS": "Industrials",  "CAT": "Industrials",
    # Materials (1)
    "LIN": "Materials",
    # Utilities (1)
    "NEE": "Utilities",
}

# Short display labels for scan output (≤ 7 chars)
_SECTOR_ABBREV: dict[str, str] = {
    "Technology":             "Tech",
    "Consumer Discretionary": "CnDisc",
    "Communication Services": "CommSvc",
    "Health Care":            "HlthCr",
    "Financials":             "Fin",
    "Energy":                 "Energy",
    "Consumer Staples":       "CnStap",
    "Industrials":            "Indust",
    "Materials":              "Matls",
    "Utilities":              "Util",
}


# ── Stock profile classification ───────────────────────────────────────────────

MEGA_CAP_LIQUID: frozenset[str] = frozenset({
    # >$500B market cap as of 2024–2025. Continuous analyst coverage, intraday
    # liquidity so high that daily-bar Wyckoff patterns are less reliable.
    # base_quality_score is ANTI-predictive here (see docs/STRATEGY.md).
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META",
})


def stock_profile(symbol: str) -> str:
    """
    Classify a symbol into a conviction-scoring tier.

    Three tiers:

        "mega_cap_liquid"  — >$500B market cap.  Constant analyst coverage
                             makes price action noisy on daily bars.  The
                             base_quality Wyckoff filter is anti-predictive
                             here; use absorption + news scoring only.

        "large_cap_growth" — $50B–$500B.  Full Wyckoff suite applicable once
                             validated across more symbols.  Currently covers
                             all SP500_SAMPLE names outside MEGA_CAP_LIQUID.

        "mid_cap_growth"   — <$50B.  Highest potential conviction lift from
                             the full Wyckoff suite; not yet in SP500_SAMPLE.

    Polygon.io will provide real-time market cap data in Phase 5.  Until then,
    classification is symbol-name based using the SP500_SAMPLE universe.

    Args:
        symbol: Ticker symbol, e.g. "AAPL" or "CAT".

    Returns:
        One of: "mega_cap_liquid", "large_cap_growth", "mid_cap_growth".
    """
    if symbol in MEGA_CAP_LIQUID:
        return "mega_cap_liquid"
    if symbol in SP500_SAMPLE or symbol in WATCHLIST_SMALL:
        return "large_cap_growth"
    return "mid_cap_growth"


def market_cap_tier(symbol: str) -> str:
    """
    Map a symbol to a market-cap tier for options signal routing.

    Tiers:
        "mega_cap"  — >$200B (MEGA_CAP_LIQUID set): use PCR + IV skew only.
        "large_cap" — $10B–$200B (rest of SP500_SAMPLE): unusual volume ≥ 3×.
        "mid_cap"   — $1B–$10B (not in SP500_SAMPLE): unusual volume ≥ 5×,
                       highest signal quality for flat-file detector.
        "small_cap" — <$1B: options market too thin — no score contribution.

    Currently uses symbol-name heuristics (SP500_SAMPLE / MEGA_CAP_LIQUID).
    The "small_cap" bucket is indistinguishable from "mid_cap" without real
    market-cap data; that separation arrives with Polygon reference integration.
    """
    profile = stock_profile(symbol)
    if profile == "mega_cap_liquid":
        return "mega_cap"
    if profile == "large_cap_growth":
        return "large_cap"
    return "mid_cap"   # mid_cap_growth (includes unclassifiable small-caps)


def load_universe(name: str = "small") -> list[str]:
    """
    Return a symbol list by name.

    Supported names:
        "small"                — 7 curated names for fast testing
        "sp500_sample"         — 50-symbol SP500 sample (default for CI/tests)
        "tradeable"            — full filtered optionable US equity universe
                                 loaded from today's parquet cache;
                                 build first with UniverseManager.build_tradeable_universe()
        "tradeable_no_options" — same but skips the IBKR options-check filter
        comma-separated list   — e.g. "AAPL,MSFT,NVDA"

    Args:
        name: Universe name or comma-separated symbol list.

    Returns:
        List of ticker symbols, uppercase.
    """
    if name == "small":
        return WATCHLIST_SMALL
    if name == "sp500_sample":
        return SP500_SAMPLE
    if name in ("tradeable", "tradeable_no_options"):
        from datetime import date as _date
        from quantlab.universe import load_universe_cache
        cached = load_universe_cache(_date.today())
        if cached:
            syms, _ = cached
            logger.info("Loaded tradeable universe from cache: %d symbols", len(syms))
            return syms
        logger.warning(
            "Tradeable universe not cached for today. "
            "Run: python -c \"from quantlab.universe import UniverseManager; "
            "UniverseManager().build_tradeable_universe(date.today(), polygon_provider)\""
        )
        return SP500_SAMPLE   # graceful fallback
    # custom comma-separated
    return [s.strip().upper() for s in name.split(",") if s.strip()]


# ── Scanner ───────────────────────────────────────────────────────────────────

def scan_symbol(
    symbol: str,
    bars: Sequence[Bar],
    signal_type: str = "breakout",
    lookback: int = 20,
    regime_bars: Sequence[Bar] | None = None,
    news_features=None,
    min_rel_volume: float | None = 1.5,
) -> ScanResult | None:
    """
    Run all signal and confirmation checks for a single symbol.

    Args:
        symbol:         Ticker symbol.
        bars:           Recent bar history (at least lookback + 1 bars).
        signal_type:    "breakout" or "sma".
        lookback:       Signal lookback period.
        regime_bars:    SPY bars for regime filter (optional).
        news_features:  NewsFeatures dataclass from quantlab.news (optional).
        min_rel_volume: Minimum relative volume threshold.

    Returns:
        ScanResult with conviction_score, or None if not enough bars.
    """
    from quantlab.signals import (
        relative_volume as _rel_vol,
        stage_classification as _stage_cls,
        volume_on_breakout_score as _vol_breakout,
    )
    from quantlab.signals.wyckoff import (
        absorption_score as _absorption,
        base_quality_score as _base_quality,
        volume_character_score as _vol_char,
        is_wyckoff_spring as _spring,
    )
    from quantlab.signals.earnings import (
        compute_earnings_profile as _earn_profile,
        earnings_acceleration_score as _earn_score,
    )
    from quantlab.signals.volume_profile import (
        accumulation_days_ratio as _accum_ratio,
        volume_trend_score as _vol_trend,
        climactic_volume_score as _climax,
    )
    from quantlab.signals.relative_strength import rs_score as _rs_score

    if len(bars) <= lookback:
        logger.debug(f"{symbol}: not enough bars ({len(bars)} <= {lookback})")
        return None

    # Run primary signal
    if signal_type == "breakout":
        signal_result = breakout_signal(bars, symbol, lookback, min_rel_volume=None)
    elif signal_type == "sma":
        signal_result = sma_signal(bars, symbol, lookback)
    else:
        raise ValueError(f"Unknown signal_type: {signal_type}")

    if signal_result is None:
        return None

    today = bars[-1].as_of.isoformat()

    # Regime filter
    bullish = True
    if regime_bars and len(regime_bars) > 200:
        bullish = regime_is_bullish(regime_bars, sma_period=200)

    # Relative volume
    rv = _rel_vol(bars, period=20)

    # ATR stop
    stop = atr_stop_price(bars, signal_result.entry_close)

    # Wyckoff structural scores
    bq     = _base_quality(bars)
    ab     = _absorption(bars)
    vc     = _vol_char(bars)
    spring = _spring(bars)

    # Earnings acceleration (pure bar-based, no fundamental data required)
    earn_profile = _earn_profile(symbol, bars)
    ea           = _earn_score(earn_profile)

    # Relative strength vs benchmark (SPY via regime_bars; 0.0 when not available)
    rs = _rs_score(bars, regime_bars) if regime_bars and len(regime_bars) > 126 else 0.0

    # Institutional volume signature
    accum_ratio = _accum_ratio(bars)
    vol_trend   = _vol_trend(bars)
    climax      = _climax(bars)

    # Stage classification (Weinstein/Minervini) and breakout volume quality
    stage     = _stage_cls(bars)
    bvs       = _vol_breakout(bars)

    # News features
    n_count = 0
    n_cat = "none"
    k_score = None
    c_score = None
    if news_features is not None:
        n_count = news_features.total_count
        n_cat = news_features.dominant_category
        k_score = news_features.avg_k_score
        c_score = news_features.avg_c_score

    result = ScanResult(
        symbol=symbol,
        scan_date=today,
        signal_type=signal_type,
        signal=signal_result.signal,
        entry_close=signal_result.entry_close,
        indicator_value=signal_result.indicator_value,
        lookback=lookback,
        regime_bullish=bullish,
        news_count=n_count,
        news_category=n_cat,
        news_k_score=k_score,
        news_c_score=c_score,
        rel_volume=rv,
        atr_stop=stop,
        base_quality=bq,
        absorption=ab,
        volume_character=vc,
        wyckoff_spring=spring,
        earnings_acceleration=ea,
        accumulation_ratio=accum_ratio,
        volume_trend=vol_trend,
        climactic_volume=climax,
        sector=SECTOR_MAP.get(symbol, ""),
        rs_score=rs,
        market_cap_tier=market_cap_tier(symbol),
        stage=stage,
        breakout_volume_score=bvs,
    )

    result.conviction_score = score_conviction(result)

    if symbol in LOW_EDGE_SYMBOLS and result.signal:
        logger.warning(
            "%s: signal fired but symbol is in LOW_EDGE_SYMBOLS "
            "(negative OOS edge on historical IBKR data) — apply extra scrutiny",
            symbol,
        )

    return result


def sector_filter(
    results: list[ScanResult],
    cluster_threshold: int = 3,
    penalty: float = 0.05,
) -> list[ScanResult]:
    """
    Penalise same-sector signal clusters on the same scan day.

    When three or more symbols from the same GICS sector appear in the
    results list simultaneously, each symbol in that cluster receives a
    −0.05 conviction penalty and is flagged with sector_cluster=True.

    Rationale: a broad macro move (e.g. rising oil prices) can cause all
    Energy names to break out together.  Letting every correlated signal
    land in the watchlist at full conviction inflates exposure to a single
    factor.  The penalty reduces their scores relative to single-sector
    breakouts that represent more idiosyncratic edge.

    Args:
        results:           List of ScanResult objects (any status — filter
                           happens before the actionable threshold is applied).
        cluster_threshold: Minimum signals in a sector to trigger penalty (default 3).
        penalty:           Conviction reduction per clustered symbol (default 0.05).

    Returns:
        The same list, modified in-place and re-sorted by conviction descending.
    """
    from collections import Counter

    sector_counts = Counter(
        r.sector for r in results if r.sector and r.signal
    )

    clustered: list[str] = []
    for r in results:
        if r.sector and sector_counts[r.sector] >= cluster_threshold:
            r.conviction_score = max(0.0, round(r.conviction_score - penalty, 4))
            r.sector_cluster   = True
            if r.sector not in clustered:
                clustered.append(r.sector)
                logger.info(
                    "sector_filter: %s has %d signals — applying −%.2f penalty",
                    r.sector, sector_counts[r.sector], penalty,
                )

    results.sort(key=lambda r: r.conviction_score, reverse=True)
    return results


def run_universe_scan(
    provider: MarketDataProvider,
    symbols: list[str],
    start_date: date,
    end_date: date,
    signal_type: str = "breakout",
    lookback: int = 20,
    min_conviction: float = 0.4,
    cost_bps: float = 10.0,
    ibkr_connection=None,  # live IB() instance for news; None = skip news
) -> list[ScanResult]:
    """
    Scan a universe of symbols and return ranked actionable setups.

    This is the top-level entry point for the daily pre-market scan.
    Results are sorted by conviction_score descending.

    Args:
        provider:       MarketDataProvider instance.
        symbols:        List of ticker symbols to scan.
        start_date:     Bar history start date.
        end_date:       Bar history end date (today for live scans).
        signal_type:    "breakout" or "sma".
        lookback:       Signal lookback period in bars.
        min_conviction: Minimum score to include in results.
        cost_bps:       Transaction cost (for display/logging).
        ibkr_connection: Live IB() for news fetching (None = price-only scan).

    Returns:
        List of ScanResult sorted by conviction_score descending.
    """
    from contextlib import nullcontext
    from quantlab.news import fetch_news, compute_news_features
    from quantlab.storage import append_trades_to_db
    from datetime import datetime

    results = []
    total = len(symbols)

    # Use the provider as a context manager when it supports one (e.g. IbkrProvider).
    # This opens a single persistent TWS connection for all bar requests in this scan,
    # eliminating the per-symbol connect/disconnect handshake (~2 s each).
    # Providers without __enter__ (MockMarketDataProvider, PolygonProvider) fall
    # through to nullcontext so this change is backward-compatible.
    _ctx = provider if hasattr(provider, "__enter__") else nullcontext()
    with _ctx:
        # Load latest breadth snapshot for regime adjustment and override flag.
        # Non-fatal — scan proceeds with neutral breadth (adj=0, override=False).
        from quantlab.signals.breadth import get_latest_snapshot, breadth_regime_adjustment
        _breadth_snap = get_latest_snapshot()
        _breadth_adj, _breadth_override = breadth_regime_adjustment(_breadth_snap)
        if _breadth_snap:
            logger.info(
                "Breadth: %s  tape=%s  10d-ratio=%s  McClellan=%s",
                _breadth_snap.date, _breadth_snap.tape,
                f"{_breadth_snap.ratio_10d:.2f}" if _breadth_snap.ratio_10d else "--",
                f"{_breadth_snap.mcclellan_oscillator:+.0f}" if _breadth_snap.mcclellan_oscillator else "--",
            )

        # Fetch macro context (CBOE VIX always; FRED if API key configured).
        # Non-fatal — defaults to risk_on / low when unavailable.
        _macro_regime = "risk_on"
        _vix_regime = "low"
        _vix_close: float | None = None
        try:
            from datetime import timedelta as _td
            from quantlab.providers.cboe import fetch_vix_history, classify_vix_regime
            _vix_bars = fetch_vix_history(end_date - _td(days=10), end_date)
            if _vix_bars:
                _vix_close = _vix_bars[-1].close
                _vix_regime, _vix_score = classify_vix_regime(_vix_close)
                logger.info("VIX: %.2f → %s (score %d)", _vix_close, _vix_regime, _vix_score)
        except Exception as _vix_err:
            logger.debug("VIX fetch failed: %s", _vix_err)

        try:
            from quantlab.config import settings as _cfg
            _fred_key = getattr(_cfg, "fred_api_key", "") or ""
            if _fred_key:
                from quantlab.providers.fred import fetch_macro_snapshot, classify_macro_regime as _cmr
                _snap = fetch_macro_snapshot(_fred_key, end_date)
                if _vix_close is not None:
                    _snap.vix_close = _vix_close
                    _snap.macro_regime = _cmr(_snap)
                _macro_regime = _snap.macro_regime
                logger.info(
                    "Macro regime: %s  10y2y=%s  HY=%s  VIX=%s",
                    _macro_regime,
                    f"{_snap.yield_spread_10y2y:+.2f}" if _snap.yield_spread_10y2y is not None else "--",
                    f"{_snap.hy_credit_spread:.2f}" if _snap.hy_credit_spread is not None else "--",
                    f"{_vix_close:.2f}" if _vix_close is not None else "--",
                )
            else:
                logger.debug("FRED_API_KEY not configured — macro regime defaults to risk_on")
        except Exception as _fred_err:
            logger.debug("FRED macro context failed: %s", _fred_err)

        # Import EDGAR helpers once before the per-symbol loop.
        try:
            from quantlab.providers.edgar import (
                get_edgar_acceleration as _get_edgar_accel,
                get_next_earnings_date as _get_next_earnings,
                get_last_earnings_result as _get_last_earnings,
                count_trading_days as _count_trading_days,
                get_edgar_peg_score as _get_edgar_peg_score,
            )
        except Exception:
            _get_edgar_accel = None           # type: ignore[assignment]
            _get_next_earnings = None         # type: ignore[assignment]
            _get_last_earnings = None         # type: ignore[assignment]
            _count_trading_days = None        # type: ignore[assignment]
            _get_edgar_peg_score = None       # type: ignore[assignment]

        # Import real-time earnings press release checker (highest priority proximity source).
        try:
            from quantlab.news.earnings_parser import (
                get_recent_earnings_result as _get_recent_earn_result,
            )
        except Exception:
            _get_recent_earn_result = None  # type: ignore[assignment]

        # Fetch SPY bars once for regime filter and RS calculation.
        # Failures are non-fatal — scan proceeds with regime_bullish=True and rs_score=0.
        spy_bars = None
        try:
            spy_bars = list(provider.get_daily_bars("SPY", start_date, end_date))
            if spy_bars:
                logger.info("SPY bars: %d bars loaded (regime filter + RS reference)", len(spy_bars))
        except Exception as _spy_err:
            logger.debug("SPY bars unavailable (%s) — regime=bullish, RS scores=0", _spy_err)

        for i, symbol in enumerate(symbols, 1):
            try:
                logger.info(f"[{i}/{total}] Scanning {symbol}...")

                bars = provider.get_daily_bars(symbol, start_date, end_date)
                if not bars:
                    logger.warning(f"{symbol}: no bars returned")
                    continue

                # Fetch EDGAR earnings acceleration (DuckDB-cached, 7-day TTL).
                # Non-fatal — OHLCV inference used when EDGAR is unavailable.
                edgar_accel: float | None = None
                if _get_edgar_accel is not None:
                    try:
                        edgar_accel = _get_edgar_accel(symbol)
                    except Exception as _ea_err:
                        logger.debug("%s: EDGAR acceleration unavailable: %s", symbol, _ea_err)

                # PEG score from EDGAR cache (reads eps_diluted + eps_growth columns;
                # no additional network request — 0.0 when cache miss).
                _peg_score: float = 0.0
                if _get_edgar_peg_score is not None:
                    try:
                        _peg_score = _get_edgar_peg_score(symbol, bars[-1].close)
                    except Exception as _pg_err:
                        logger.debug("%s: PEG score unavailable: %s", symbol, _pg_err)

                # Fetch news if IBKR connection is provided
                news_feat = None
                if ibkr_connection is not None:
                    try:
                        from ib_insync import Stock as _Stock
                        contract = _Stock(symbol, "SMART", "USD")
                        qualified = ibkr_connection.qualifyContracts(contract)
                        if qualified:
                            news_items = fetch_news(ibkr_connection, qualified[0], days=30, limit=50)
                            news_feat = compute_news_features(
                                news_items,
                                end_date.isoformat(),
                                lookback_days=7,
                            )
                            # Parse earnings beat/miss from headlines → earnings_results table
                            try:
                                from quantlab.news.earnings_parser import (
                                    make_earnings_result,
                                    store_earnings_result,
                                )
                                for _item in news_items:
                                    _hl = getattr(_item, "headline", "") or ""
                                    _er = make_earnings_result(symbol, _hl)
                                    if _er is not None:
                                        store_earnings_result(_er)
                                        logger.info(
                                            "%s: earnings headline stored — beat_score=%.2f",
                                            symbol, _er.beat_score,
                                        )
                            except Exception as _ep_err:
                                logger.debug(
                                    "%s: earnings parse from news failed: %s", symbol, _ep_err
                                )
                    except Exception as e:
                        logger.debug(f"{symbol} news fetch failed: {e}")

                result = scan_symbol(
                    symbol=symbol,
                    bars=bars,
                    signal_type=signal_type,
                    lookback=lookback,
                    regime_bars=spy_bars,   # enables regime filter + RS scoring
                    news_features=news_feat,
                )

                if result is not None:
                    # Apply breadth, macro, and EDGAR adjustments, then re-score
                    result.breadth_regime_adj  = _breadth_adj
                    result.breadth_override    = _breadth_override
                    result.macro_regime        = _macro_regime
                    result.vix_regime          = _vix_regime
                    result.edgar_acceleration  = edgar_accel
                    result.peg_score           = _peg_score

                    # Earnings calendar proximity
                    # Priority 1: real-time press release beat/miss (most authoritative)
                    # Priority 2: EDGAR filing-based historical estimate (fallback)
                    try:
                        _proximity = "neutral"
                        _used_real_time = False

                        if _get_recent_earn_result is not None:
                            _press = _get_recent_earn_result(symbol, max_days=5)
                            if _press is not None:
                                _used_real_time = True
                                if _press.beat_score >= 0.7:
                                    _proximity = "post_earnings_beat"
                                elif _press.beat_score <= 0.3:
                                    _proximity = "post_earnings_miss"
                                # 0.3 < score < 0.7 → neutral (mixed signals)
                                logger.debug(
                                    "%s: press-release beat_score=%.2f → %s",
                                    symbol, _press.beat_score, _proximity,
                                )

                        if not _used_real_time:
                            # EDGAR fallback: pre-earnings proximity + historical beat
                            if _get_next_earnings is not None:
                                _next = _get_next_earnings(symbol)
                                if _next:
                                    _next_date, _days_until = _next
                                    if 0 <= _days_until <= 5:
                                        _proximity = "pre_earnings"
                            if _proximity == "neutral" and (
                                _get_last_earnings is not None
                                and _count_trading_days is not None
                            ):
                                _last = _get_last_earnings(symbol)
                                if _last:
                                    _last_date, _was_beat = _last
                                    _days_since = _count_trading_days(
                                        _last_date, end_date
                                    )
                                    if 0 <= _days_since <= 5:
                                        _proximity = (
                                            "post_earnings_beat"
                                            if _was_beat else "post_earnings_miss"
                                        )

                        result.earnings_proximity = _proximity
                        if _proximity != "neutral":
                            logger.info(
                                "%s: earnings_proximity=%s", symbol, _proximity
                            )
                    except Exception as _ep_err:
                        logger.debug(
                            "%s: earnings_proximity unavailable: %s", symbol, _ep_err
                        )

                    result.conviction_score    = score_conviction(result)
                    if edgar_accel is not None:
                        logger.debug(
                            "%s: edgar_accel=%.2f  ohlcv_accel=%.2f",
                            symbol, edgar_accel, result.earnings_acceleration,
                        )
                    results.append(result)

            except Exception as e:
                logger.error(f"{symbol}: scan error — {e}")
                continue

    # Sort by conviction score, highest first
    results.sort(key=lambda r: r.conviction_score, reverse=True)

    # Apply sector correlation filter — penalises clusters of ≥3 same-sector signals
    results = sector_filter(results)

    # Filter to actionable
    actionable = [r for r in results if r.is_actionable(min_conviction)]

    logger.info(
        "Scan complete: %d symbols → %d processed → %d actionable "
        "(min_conviction=%s  macro=%s  vix=%s)",
        total, len(results), len(actionable), min_conviction, _macro_regime, _vix_regime,
    )

    return actionable


# ── ZeroMQ stub (Phase 5+) ─────────────────────────────────────────────────────

class SignalPublisher:
    """
    ZeroMQ publisher stub for Phase 5+ live execution.

    The scanner publishes ScanResult objects on a ZMQ PUB socket.
    The execution engine subscribes on a SUB socket and validates
    before submitting orders.

    This is a stub — implement when moving to paper trading.
    """

    def __init__(self, address: str = "tcp://127.0.0.1:5555") -> None:
        self.address = address
        self._socket = None

    def connect(self) -> None:
        try:
            import zmq
            ctx = zmq.Context()
            self._socket = ctx.socket(zmq.PUB)
            self._socket.bind(self.address)
            logger.info(f"Signal publisher bound to {self.address}")
        except ImportError:
            logger.warning("pyzmq not installed — SignalPublisher is a no-op")

    def publish(self, result: ScanResult) -> None:
        if self._socket is None:
            return
        import json
        payload = json.dumps(result.__dict__, default=str)
        self._socket.send_string(f"SIGNAL {payload}")

    def close(self) -> None:
        if self._socket:
            self._socket.close()
