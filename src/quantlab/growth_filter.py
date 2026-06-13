"""
quantlab.growth_filter — growth-stock pre-filter that runs BEFORE stage analysis.

Motivation (2026-06-13)
-----------------------
The candidate funnel was stage-first: in a correction tape the table filled
with whatever held Stage 2 — regional banks, insurers, REITs, staples (KO, MO,
MRK, UNH, ~40 banks).  This is a GROWTH breakout system (O'Neil / Minervini /
Kell): those names can't deliver explosive moves, they dilute every downstream
signal, and our research shows unusual-options signals are only predictive on
$1B–$10B mid-caps (06-12: KO/MO/MRK/UNH all flagged at z=10 — mega-cap noise).

We invert the funnel.  A growth-universe pre-filter runs at universe load, so
ALL downstream signals are computed only on the population they were designed
for.  Two tiers:

  TIER 1 — HARD GATES (the tradeable hunting ground)
    1. Market cap: $1B ≤ cap ≤ $10B core; soft band to $50B only if ADR% ≥ 5%
       (fast large movers without readmitting slow mega-caps).
    2. ADR% (20-day avg of high/low − 1) ≥ 3.5% — removes most staples /
       utilities / REITs / banks naturally (preferred over sector exclusion).
    3. Liquidity: 20-day avg dollar volume ≥ $20M AND price ≥ $10.
    4. Security type: CS-only filter is already applied upstream by the
       universe builder; this runs after it.

  TIER 2 — GROWTH QUALIFICATION (the prey; uses the period-matched EDGAR YoY
  from commit a0c5231 — winsorized, turned_positive flags)
    Qualify on ANY of: EPS YoY ≥ +25% · revenue YoY ≥ +20% · turned_positive ·
    acceleration (latest YoY rate > prior quarter's, latest YoY ≥ +15%).
    Revenue acceleration is weighted at least as heavily as EPS in ranking —
    now-correct GAAP EPS still swings 469–900% off small/recovering bases, so
    revenue YoY is the cleaner acceleration signal.

DATA SEMANTICS (consistent with the EDGAR MISSING ≠ ZERO convention)
    Insufficient fundamental history (foreign filers, recent IPOs, < 4 quarters)
    and quarantined/NULL YoY values are UNAVAILABLE, not failing — they route to
    a counted UNQUALIFIED-DATA bucket, never silently dropped as "failed growth."
    Recent IPOs (≥ 2 quarters) qualify on revenue alone; EPS is often negative
    early and positivity is NOT required.

The build persists per-gate booleans AND raw computed values to the
growth_universe table (not just a symbol list) so the IC monitor and future
re-weighting can study them.  DEFAULT = BYPASS ON: the filter is built but
inactive until the qualified list is reviewed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Config — every threshold lives here (no buried magic numbers)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class GrowthFilterConfig:
    """All growth-filter thresholds.  Override via from_config() or directly."""

    # ── Tier 1: cap band ──────────────────────────────────────────────────────
    cap_min: float = 1_000_000_000.0        # $1B  core lower bound
    cap_core_max: float = 10_000_000_000.0  # $10B core upper bound
    cap_soft_max: float = 50_000_000_000.0  # $50B soft upper bound …
    cap_soft_band_min_adr: float = 0.05     # … admitted only if ADR% ≥ 5%

    # ── Tier 1: ADR% (average daily range) ────────────────────────────────────
    adr_min: float = 0.035                  # ≥ 3.5%
    adr_window: int = 20                    # 20-day average

    # ── Tier 1: liquidity ─────────────────────────────────────────────────────
    min_dollar_vol: float = 20_000_000.0    # ≥ $20M 20-day avg dollar volume
    min_price: float = 10.0                 # ≥ $10
    liquidity_window: int = 20

    # Optional SIC/sector exclusion — default OFF (ADR% gate handles defensives)
    sector_exclude: tuple[str, ...] = ()

    # ── Tier 2: growth qualification ──────────────────────────────────────────
    eps_yoy_min: float = 0.25               # EPS YoY ≥ +25%
    rev_yoy_min: float = 0.20               # revenue YoY ≥ +20%
    accel_min_yoy: float = 0.15             # acceleration qualifies at ≥ +15%
    ipo_min_quarters: int = 2               # IPO revenue-only path needs ≥ 2 q
    min_quarters_for_growth: int = 4        # < this ⇒ IPO/insufficient path

    # ── Ranking weights (revenue ≥ EPS) ───────────────────────────────────────
    rank_rev_weight: float = 1.0
    rank_eps_weight: float = 0.5            # EPS weighted half — noisier signal
    rank_winsorize: float = 3.0            # ±300% cap on YoY rank inputs
    rank_turned_positive_eps: float = 0.25  # inflection stand-in, NOT 900%
    rank_accel_bonus: float = 0.25

    # ── Pipeline ──────────────────────────────────────────────────────────────
    bypass: bool = True                     # DEFAULT BYPASS ON (built, inactive)

    @classmethod
    def from_config(cls) -> "GrowthFilterConfig":
        """Build from scanner config overrides under the ``growth_filter`` key.

        Defaults live in this dataclass; scanner config only supplies overrides,
        so a partial ``growth_filter`` dict need not list every field.
        """
        import dataclasses
        try:
            from quantlab.utils import get_config
            overrides = dict(get_config("scanner").get("growth_filter", {}) or {})
        except Exception:
            overrides = {}
        valid = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in overrides.items() if k in valid}
        if "sector_exclude" in kwargs and kwargs["sector_exclude"] is not None:
            kwargs["sector_exclude"] = tuple(kwargs["sector_exclude"])
        return cls(**kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# Per-symbol facts and result
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GrowthFacts:
    """Computed inputs for one symbol.  None ⇒ unavailable (MISSING ≠ ZERO)."""
    symbol: str
    market_cap: Optional[float] = None
    cap_source: str = ""
    adr_pct: Optional[float] = None
    dollar_vol: Optional[float] = None
    price: Optional[float] = None
    eps_yoy: Optional[float] = None
    rev_yoy: Optional[float] = None
    turned_positive: bool = False
    rev_accel: Optional[bool] = None
    eps_accel: Optional[bool] = None
    n_quarters: Optional[int] = None


# bucket values
QUALIFIED = "qualified"
UNQUALIFIED_DATA = "unqualified_data"
FAILED_GROWTH = "failed_growth"
FAILED_ADR = "failed_adr"
FAILED_CAP = "failed_cap"
FAILED_LIQUIDITY = "failed_liquidity"


@dataclass
class GrowthResult:
    facts: GrowthFacts
    pass_liquidity: bool = False
    pass_cap: bool = False
    pass_adr: bool = False
    bucket: str = FAILED_LIQUIDITY
    growth_reasons: tuple[str, ...] = ()
    rank: Optional[float] = None

    @property
    def symbol(self) -> str:
        return self.facts.symbol

    @property
    def growth_qualified(self) -> bool:
        return self.bucket == QUALIFIED


# ══════════════════════════════════════════════════════════════════════════════
# Pure gate functions (fully testable offline)
# ══════════════════════════════════════════════════════════════════════════════

def passes_liquidity(facts: GrowthFacts, cfg: GrowthFilterConfig) -> bool:
    if facts.dollar_vol is None or facts.price is None:
        return False
    return facts.dollar_vol >= cfg.min_dollar_vol and facts.price >= cfg.min_price


def passes_cap_band(facts: GrowthFacts, cfg: GrowthFilterConfig) -> bool:
    """Core band $1B–$10B; soft band to $50B only when ADR% ≥ soft-band floor."""
    cap = facts.market_cap
    if cap is None:
        return False
    if cfg.cap_min <= cap <= cfg.cap_core_max:
        return True
    if cfg.cap_core_max < cap <= cfg.cap_soft_max:
        return facts.adr_pct is not None and facts.adr_pct >= cfg.cap_soft_band_min_adr
    return False


def passes_adr(facts: GrowthFacts, cfg: GrowthFilterConfig) -> bool:
    if facts.adr_pct is None:
        return False
    return facts.adr_pct >= cfg.adr_min


def qualify_growth(
    facts: GrowthFacts, cfg: GrowthFilterConfig
) -> tuple[str, tuple[str, ...]]:
    """Tier-2 growth status for a Tier-1 survivor.

    Returns ``(bucket, reasons)`` where bucket ∈ {qualified, unqualified_data,
    failed_growth}.  unqualified_data means the growth fundamentals are
    UNAVAILABLE (foreign filer, quarantined/NULL YoY, too-new IPO) — never a
    silent "failed growth."  failed_growth means data was present but no
    qualifier met its threshold.
    """
    reasons: list[str] = []

    # ── IPO / insufficient-history path ───────────────────────────────────────
    if facts.n_quarters is not None and facts.n_quarters < cfg.min_quarters_for_growth:
        if facts.n_quarters < cfg.ipo_min_quarters:
            return UNQUALIFIED_DATA, ()        # too new to judge at all
        # Recent IPO: qualify on revenue alone; EPS positivity NOT required.
        if facts.rev_yoy is not None:
            if facts.rev_yoy >= cfg.rev_yoy_min:
                return QUALIFIED, ("ipo_rev",)
            return FAILED_GROWTH, ()
        return UNQUALIFIED_DATA, ()            # revenue YoY not computable yet

    # ── Standard path ─────────────────────────────────────────────────────────
    have_data = False
    if facts.rev_yoy is not None:
        have_data = True
        if facts.rev_yoy >= cfg.rev_yoy_min:
            reasons.append("rev")
    if facts.eps_yoy is not None:
        have_data = True
        if facts.eps_yoy >= cfg.eps_yoy_min:
            reasons.append("eps")
    if facts.turned_positive:
        have_data = True
        reasons.append("turned_positive")
    # Acceleration: latest YoY rate > prior quarter's, with latest YoY ≥ floor.
    if facts.rev_accel is not None:
        have_data = True
        if facts.rev_accel and facts.rev_yoy is not None and facts.rev_yoy >= cfg.accel_min_yoy:
            reasons.append("rev_accel")
    if facts.eps_accel is not None:
        have_data = True
        if facts.eps_accel and facts.eps_yoy is not None and facts.eps_yoy >= cfg.accel_min_yoy:
            reasons.append("eps_accel")

    if reasons:
        return QUALIFIED, tuple(reasons)
    if not have_data:
        return UNQUALIFIED_DATA, ()
    return FAILED_GROWTH, ()


def _winsorize(v: Optional[float], cap: float) -> float:
    if v is None:
        return 0.0
    return max(-cap, min(cap, v))


def growth_rank(facts: GrowthFacts, cfg: GrowthFilterConfig) -> float:
    """Ranking score for the qualified list — revenue weighted ≥ EPS.

    turned_positive is a real inflection but a tiny absolute base, so it
    contributes a modest fixed stand-in (rank_turned_positive_eps), NOT the
    raw 900% GAAP swing.  Acceleration adds a small bonus.
    """
    rev = _winsorize(facts.rev_yoy, cfg.rank_winsorize)
    if facts.eps_yoy is not None:
        eps = _winsorize(facts.eps_yoy, cfg.rank_winsorize)
    elif facts.turned_positive:
        eps = cfg.rank_turned_positive_eps
    else:
        eps = 0.0
    score = cfg.rank_rev_weight * rev + cfg.rank_eps_weight * eps
    if facts.rev_accel:
        score += cfg.rank_accel_bonus
    return round(score, 6)


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation + funnel
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_symbol(facts: GrowthFacts, cfg: GrowthFilterConfig) -> GrowthResult:
    """Apply gates in funnel order; bucket at the first failure."""
    res = GrowthResult(facts=facts)
    res.pass_liquidity = passes_liquidity(facts, cfg)
    res.pass_cap = passes_cap_band(facts, cfg)
    res.pass_adr = passes_adr(facts, cfg)

    if not res.pass_liquidity:
        res.bucket = FAILED_LIQUIDITY
    elif not res.pass_cap:
        res.bucket = FAILED_CAP
    elif not res.pass_adr:
        res.bucket = FAILED_ADR
    else:
        bucket, reasons = qualify_growth(facts, cfg)
        res.bucket = bucket
        res.growth_reasons = reasons
        if bucket == QUALIFIED:
            res.rank = growth_rank(facts, cfg)
    return res


@dataclass
class FunnelCounts:
    """Sequential per-gate counts in REPORT order (each level passes all prior)."""
    total: int = 0
    after_liquidity: int = 0
    after_cap: int = 0
    after_adr: int = 0          # == Tier-1 survivors
    growth_qualified: int = 0
    unqualified_data: int = 0
    failed_growth: int = 0

    def render(self) -> str:
        """One-line funnel with the binding constraint always visible."""
        line = (
            f"Universe {self.total:,} → liquidity/price {self.after_liquidity:,} "
            f"→ cap band {self.after_cap:,} → ADR% {self.after_adr:,} "
            f"→ growth-qualified {self.growth_qualified:,} "
            f"(+ {self.unqualified_data:,} unqualified-data)"
        )
        return line


def compute_funnel(results: list[GrowthResult]) -> FunnelCounts:
    f = FunnelCounts(total=len(results))
    for r in results:
        if r.pass_liquidity:
            f.after_liquidity += 1
            if r.pass_cap:
                f.after_cap += 1
                if r.pass_adr:
                    f.after_adr += 1
                    if r.bucket == QUALIFIED:
                        f.growth_qualified += 1
                    elif r.bucket == UNQUALIFIED_DATA:
                        f.unqualified_data += 1
                    elif r.bucket == FAILED_GROWTH:
                        f.failed_growth += 1
    return f


def evaluate_universe(
    facts_by_symbol: dict[str, GrowthFacts], cfg: GrowthFilterConfig
) -> tuple[list[GrowthResult], FunnelCounts]:
    """Pure core: evaluate every symbol, return results + funnel counts."""
    results = [evaluate_symbol(facts_by_symbol[s], cfg)
               for s in facts_by_symbol]
    return results, compute_funnel(results)


# ══════════════════════════════════════════════════════════════════════════════
# Data gathering (ADR / dollar-vol from grouped-daily history)
# ══════════════════════════════════════════════════════════════════════════════

def compute_adr_dollar_vol(
    bars: list, cfg: GrowthFilterConfig
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """From a recent daily-bar history (oldest- or newest-first), compute
    (adr_pct, dollar_vol, price).

    adr_pct  = mean over the last ``adr_window`` days of (high/low − 1).
    dollar_vol = mean over the last ``liquidity_window`` days of close×volume.
    price    = most recent close.

    Returns (None, None, None) components individually when a window is empty.
    Each bar must expose ``high``, ``low``, ``close``, ``volume`` and an ``as_of``
    date (PolygonProvider Bar / PriceBar both qualify).
    """
    if not bars:
        return None, None, None
    ordered = sorted(bars, key=lambda b: b.as_of)
    price = ordered[-1].close if ordered else None

    adr_bars = ordered[-cfg.adr_window:]
    ranges = [
        (b.high / b.low - 1.0)
        for b in adr_bars
        if b.low and b.low > 0 and b.high is not None
    ]
    adr_pct = sum(ranges) / len(ranges) if ranges else None

    liq_bars = ordered[-cfg.liquidity_window:]
    dvols = [
        b.close * b.volume
        for b in liq_bars
        if b.close is not None and b.volume is not None
    ]
    dollar_vol = sum(dvols) / len(dvols) if dvols else None
    return adr_pct, dollar_vol, price


def load_grouped_history(n_days: int, as_of: date) -> dict[str, list]:
    """Load up to ``n_days`` recent grouped-daily caches → {symbol: [Bar, …]}.

    Reads the offline breadth Parquet cache written by update_breadth.py — no
    network call.  Bars carry as_of/open/high/low/close/volume.
    """
    from quantlab.providers.polygon import PolygonProvider
    from quantlab.market_calendar import prev_trading_day

    pg = PolygonProvider.__new__(PolygonProvider)  # cache-only; no API key needed
    hist: dict[str, list] = {}
    d = as_of
    loaded = 0
    # Walk back generously to cover holidays/weekends until n_days are gathered.
    for _ in range(n_days * 3):
        cached = None
        try:
            cached = pg._load_breadth_cache(d)
        except Exception:
            cached = None
        if cached:
            for sym, bar in cached.items():
                hist.setdefault(sym, []).append(bar)
            loaded += 1
            if loaded >= n_days:
                break
        d = prev_trading_day(d)
    return hist


def load_growth_fundamentals(
    symbols: list[str], max_age_days: int = 30, db_path: str | None = None
) -> dict[str, dict]:
    """Latest edgar_fundamentals row per symbol → growth-tier inputs.

    Returns {symbol: {eps_yoy, rev_yoy, turned_positive, rev_accel, eps_accel,
    n_quarters}}.  Absent symbols simply don't appear (treated downstream as all
    unavailable).
    """
    if not symbols:
        return {}
    try:
        import duckdb
        from quantlab.storage import DB_PATH
        from datetime import timedelta
    except Exception:
        return {}
    out: dict[str, dict] = {}
    try:
        con = duckdb.connect(db_path or str(DB_PATH))
        cutoff = (date.today() - timedelta(days=max_age_days)).isoformat()
        ph = ",".join("?" * len(symbols))
        rows = con.execute(
            f"""
            SELECT symbol, eps_growth, revenue_growth, eps_turned_positive,
                   rev_yoy_accel, eps_yoy_accel, n_quarters
            FROM edgar_fundamentals
            WHERE symbol IN ({ph}) AND fetch_date >= ?
            ORDER BY fetch_date DESC
            """,
            list(symbols) + [cutoff],
        ).fetchall()
        con.close()
        for sym, eps, rev, tp, ra, ea, nq in rows:
            if sym in out:
                continue  # newest first — keep first seen
            out[sym] = {
                "eps_yoy": eps,
                "rev_yoy": rev,
                "turned_positive": bool(tp) if tp is not None else False,
                "rev_accel": (None if ra is None else bool(ra)),
                "eps_accel": (None if ea is None else bool(ea)),
                "n_quarters": (None if nq is None else int(nq)),
            }
    except Exception as exc:
        logger.warning("load_growth_fundamentals failed: %s", exc)
    return out


def _market_caps_cache_path(as_of: date):
    from quantlab.storage import DATA_PROCESSED, ensure_dirs
    ensure_dirs()
    return DATA_PROCESSED / f"market_caps_{as_of.isoformat()}.parquet"


def fetch_market_caps(
    symbols: list[str], as_of: date, polygon_provider=None,
    max_workers: int = 4,
) -> dict[str, float]:
    """{symbol: market_cap} from Polygon ticker details, cached daily to Parquet.

    Cap source: Polygon /v3/reference/tickers/{symbol}.market_cap — the most
    authoritative single source in the codebase.  Cached so re-runs are instant.
    Symbols whose details lack a market_cap simply don't appear in the result
    (treated as cap-unavailable downstream).
    """
    path = _market_caps_cache_path(as_of)
    cached: dict[str, float] = {}
    if path.exists():
        try:
            import pyarrow.parquet as pq
            tbl = pq.read_table(path).to_pydict()
            cached = {s: c for s, c in zip(tbl.get("symbol", []), tbl.get("market_cap", []))}
        except Exception:
            cached = {}

    missing = [s for s in symbols if s not in cached]
    if missing:
        if polygon_provider is None:
            from quantlab.providers.polygon import PolygonProvider
            polygon_provider = PolygonProvider()

        from concurrent.futures import ThreadPoolExecutor

        def _one(sym: str):
            try:
                d = polygon_provider.get_ticker_details(sym)
                cap = d.get("market_cap")
                return sym, (float(cap) if cap else None)
            except Exception:
                return sym, None

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for sym, cap in ex.map(_one, missing):
                if cap is not None:
                    cached[sym] = cap

        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            syms = list(cached.keys())
            pq.write_table(
                pa.table({"symbol": syms, "market_cap": [cached[s] for s in syms]}),
                path,
            )
        except Exception as exc:
            logger.debug("market_caps cache write failed: %s", exc)

    return {s: cached[s] for s in symbols if s in cached}


# ══════════════════════════════════════════════════════════════════════════════
# Orchestration + persistence
# ══════════════════════════════════════════════════════════════════════════════

def assemble_facts(
    symbols: list[str],
    as_of: date,
    cfg: GrowthFilterConfig,
    polygon_provider=None,
    db_path: str | None = None,
    skip_market_cap: bool = False,
) -> dict[str, GrowthFacts]:
    """Gather cap / ADR / liquidity / growth inputs for every symbol."""
    hist = load_grouped_history(max(cfg.adr_window, cfg.liquidity_window), as_of)
    fundamentals = load_growth_fundamentals(symbols, db_path=db_path)
    caps = ({} if skip_market_cap
            else fetch_market_caps(symbols, as_of, polygon_provider))
    cap_src = "polygon_ticker_details"

    facts: dict[str, GrowthFacts] = {}
    for sym in symbols:
        adr_pct, dollar_vol, price = compute_adr_dollar_vol(hist.get(sym, []), cfg)
        fnd = fundamentals.get(sym, {})
        facts[sym] = GrowthFacts(
            symbol=sym,
            market_cap=caps.get(sym),
            cap_source=cap_src if sym in caps else "",
            adr_pct=adr_pct,
            dollar_vol=dollar_vol,
            price=price,
            eps_yoy=fnd.get("eps_yoy"),
            rev_yoy=fnd.get("rev_yoy"),
            turned_positive=fnd.get("turned_positive", False),
            rev_accel=fnd.get("rev_accel"),
            eps_accel=fnd.get("eps_accel"),
            n_quarters=fnd.get("n_quarters"),
        )
    return facts


def save_growth_universe(
    as_of: date, results: list[GrowthResult], db_path: str | None = None
) -> None:
    """Persist per-symbol gate booleans + raw values to growth_universe. Non-fatal."""
    try:
        import duckdb
        from quantlab.storage import DB_PATH, _ensure_schema
        con = duckdb.connect(db_path or str(DB_PATH))
        _ensure_schema(con)
        con.execute("DELETE FROM growth_universe WHERE as_of_date = ?", [as_of.isoformat()])
        for r in results:
            f = r.facts
            con.execute(
                """
                INSERT INTO growth_universe
                    (as_of_date, symbol, market_cap, adr_pct, dollar_vol, price,
                     eps_yoy, rev_yoy, turned_positive, rev_accel, eps_accel,
                     n_quarters, cap_source, pass_liquidity, pass_cap, pass_adr,
                     growth_qualified, bucket)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    as_of.isoformat(), f.symbol, f.market_cap, f.adr_pct,
                    f.dollar_vol, f.price, f.eps_yoy, f.rev_yoy, f.turned_positive,
                    f.rev_accel, f.eps_accel, f.n_quarters, f.cap_source,
                    r.pass_liquidity, r.pass_cap, r.pass_adr, r.growth_qualified,
                    r.bucket,
                ],
            )
        con.close()
    except Exception as exc:
        logger.warning("save_growth_universe failed: %s", exc)


def build_growth_universe(
    symbols: list[str],
    as_of: date | None = None,
    cfg: GrowthFilterConfig | None = None,
    polygon_provider=None,
    db_path: str | None = None,
    persist: bool = True,
    skip_market_cap: bool = False,
    facts_by_symbol: dict[str, GrowthFacts] | None = None,
) -> tuple[list[str], list[GrowthResult], FunnelCounts]:
    """Run the growth pre-filter over ``symbols``.

    Returns (qualified_symbols, results, funnel).  Persists to growth_universe
    when ``persist`` is True.  Honours cfg.bypass: when bypassing, the funnel and
    per-gate values are still COMPUTED and persisted (so the report shows what
    the filter WOULD do during A/B), but the returned symbol list is the full
    input universe rather than the qualified subset.

    ``facts_by_symbol`` may be supplied pre-assembled (tests / callers that
    already gathered the inputs), bypassing the network/DB data gathering.
    """
    as_of = as_of or date.today()
    cfg = cfg or GrowthFilterConfig.from_config()

    facts = facts_by_symbol if facts_by_symbol is not None else assemble_facts(
        symbols, as_of, cfg, polygon_provider, db_path, skip_market_cap
    )
    results, funnel = evaluate_universe(facts, cfg)
    if persist:
        save_growth_universe(as_of, results, db_path)

    qualified = [r.symbol for r in sorted(
        (x for x in results if x.growth_qualified),
        key=lambda x: (x.rank if x.rank is not None else float("-inf")),
        reverse=True,
    )]

    if cfg.bypass:
        logger.info(
            "Growth filter BYPASS ON — returning full universe (%d). %s",
            len(symbols), funnel.render(),
        )
        return list(symbols), results, funnel

    return qualified, results, funnel


def load_excluded_defensive(
    as_of: date, limit: int = 10, db_path: str | None = None
) -> list[dict]:
    """Top names the filter excluded at a Tier-1 hard gate, by dollar volume.

    These are the large/low-ADR defensives (banks, insurers, REITs, staples,
    mega-caps) the growth filter evicts — surfaced ONLY as tape-character market
    context (the optional, config-off report panel), never as candidates.
    Returns dicts with symbol, market_cap, adr_pct, dollar_vol, bucket.
    """
    try:
        import duckdb
        from quantlab.storage import DB_PATH, _ensure_schema
        con = duckdb.connect(db_path or str(DB_PATH))
        _ensure_schema(con)
        rows = con.execute(
            """
            SELECT symbol, market_cap, adr_pct, dollar_vol, bucket
            FROM growth_universe
            WHERE as_of_date = ? AND bucket IN (?, ?)
            ORDER BY dollar_vol DESC NULLS LAST
            LIMIT ?
            """,
            [as_of.isoformat(), FAILED_CAP, FAILED_ADR, limit],
        ).fetchall()
        con.close()
    except Exception:
        return []
    return [
        {"symbol": s, "market_cap": mc, "adr_pct": adr,
         "dollar_vol": dv, "bucket": b}
        for s, mc, adr, dv, b in rows
    ]


def load_growth_funnel(as_of: date, db_path: str | None = None) -> Optional[FunnelCounts]:
    """Reconstruct funnel counts from persisted growth_universe rows."""
    try:
        import duckdb
        from quantlab.storage import DB_PATH, _ensure_schema
        con = duckdb.connect(db_path or str(DB_PATH))
        _ensure_schema(con)
        rows = con.execute(
            """
            SELECT pass_liquidity, pass_cap, pass_adr, bucket
            FROM growth_universe WHERE as_of_date = ?
            """,
            [as_of.isoformat()],
        ).fetchall()
        con.close()
    except Exception:
        return None
    if not rows:
        return None
    f = FunnelCounts(total=len(rows))
    for pl, pc, pa, bucket in rows:
        if pl:
            f.after_liquidity += 1
            if pc:
                f.after_cap += 1
                if pa:
                    f.after_adr += 1
                    if bucket == QUALIFIED:
                        f.growth_qualified += 1
                    elif bucket == UNQUALIFIED_DATA:
                        f.unqualified_data += 1
                    elif bucket == FAILED_GROWTH:
                        f.failed_growth += 1
    return f
