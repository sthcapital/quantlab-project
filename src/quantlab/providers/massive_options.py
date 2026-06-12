"""
Polygon.io REST API + Massive S3 flat-file options provider.

REST API  (api.polygon.io)
    /v3/snapshot/options/{symbol}  — full chain with Greeks, IV, OI, bid/ask,
                                     volume; paginated.

S3 Flat Files (files.massive.com)
    us_options_opra/day_aggs_v1/   — daily OHLCV for every options contract.
    Schema: ticker, volume, open, close, high, low, window_start, transactions.

API key  : POLYGON_API_KEY env var (same key used by PolygonProvider).
S3 creds : MASSIVE_ACCESS_KEY / MASSIVE_SECRET_KEY, or boto3 env defaults.
S3 endpoint: MASSIVE_S3_ENDPOINT (default https://files.massive.com).
S3 bucket  : MASSIVE_S3_BUCKET   (default files.massive.com).
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_REST_BASE = "https://api.polygon.io"
_DEFAULT_S3_ENDPOINT = "https://files.massive.com"
_DEFAULT_S3_BUCKET = "files.massive.com"
_S3_OPTIONS_PREFIX = "us_options_opra/day_aggs_v1"
_HEADERS = {"User-Agent": "QuantLab Research quantlab@sthcapital.com"}


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OptionContract:
    """A single option contract with Greeks and market data from Polygon."""

    ticker: str           # OCC ticker, e.g. O:AAPL260620C00310000
    expiry: date
    strike: float
    option_type: str      # "C" (call) or "P" (put)
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume: Optional[float] = None
    open_interest: Optional[float] = None
    iv: Optional[float] = None      # implied volatility (decimal, e.g. 0.28)
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None

    @property
    def activity(self) -> float:
        """Best available measure of market activity (OI preferred, then volume)."""
        return self.open_interest or self.volume or 0.0


# ── Helpers ────────────────────────────────────────────────────────────────────

def _clean(v) -> Optional[float]:
    """Return float or None; filters NaN and non-numeric values."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if f != f else f  # NaN check
    except (ValueError, TypeError):
        return None


# ── Provider ───────────────────────────────────────────────────────────────────

class MassiveOptionsProvider:
    """
    Options data from Polygon REST (chain snapshots) and Massive S3 (history).

    In-memory chain cache: each symbol's chain is fetched at most once per
    instance lifetime, so calling get_put_call_ratio / get_iv_skew /
    get_unusual_call_activity on the same symbol costs one REST fetch.

    DuckDB cache: today's computed options_score is persisted to the
    options_snapshots table so re-runs within the same calendar day skip
    the REST fetch entirely.
    """

    def __init__(
        self,
        api_key: str | None = None,
        s3_endpoint: str | None = None,
        s3_bucket: str | None = None,
        s3_access_key: str | None = None,
        s3_secret_key: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("POLYGON_API_KEY", "")
        self.s3_endpoint = (
            s3_endpoint
            or os.getenv("MASSIVE_S3_ENDPOINT", _DEFAULT_S3_ENDPOINT)
        )
        self.s3_bucket = (
            s3_bucket
            or os.getenv("MASSIVE_S3_BUCKET", _DEFAULT_S3_BUCKET)
        )
        self.s3_access_key = s3_access_key or os.getenv("MASSIVE_ACCESS_KEY", "")
        self.s3_secret_key = s3_secret_key or os.getenv("MASSIVE_SECRET_KEY", "")
        self._chain_cache: dict[str, list[OptionContract]] = {}
        self._session = requests.Session()
        if not self.api_key:
            logger.warning("POLYGON_API_KEY not set — options REST calls will fail with 403")

    # ── Parsing ────────────────────────────────────────────────────────────────

    @staticmethod
    def parse_option_ticker(ticker: str) -> tuple[str, date, str, float]:
        """
        Parse an OCC-format option ticker into (symbol, expiry, type, strike).

        Format: O:AAPL260620C00310000
            O:        — options prefix (stripped)
            AAPL      — underlying symbol (variable length)
            260620    — expiry YYMMDD
            C         — option type: C=call, P=put
            00310000  — strike × 1000 (8 digits, last 3 are decimal)

        Example:
            parse_option_ticker("O:AAPL260620C00310000")
            → ("AAPL", date(2026, 6, 20), "C", 310.0)

        Raises:
            ValueError: On malformed ticker.
        """
        raw = ticker.removeprefix("O:")
        # Suffix is always 15 chars: YYMMDD(6) + type(1) + strike(8)
        if len(raw) < 16:
            raise ValueError(f"Option ticker too short: {ticker!r}")
        symbol = raw[:-15]
        expiry = datetime.strptime(raw[-15:-9], "%y%m%d").date()
        option_type = raw[-9]
        strike = int(raw[-8:]) / 1000.0
        if option_type not in ("C", "P"):
            raise ValueError(f"Unexpected option type {option_type!r} in {ticker!r}")
        return symbol, expiry, option_type, strike

    # ── REST API ────────────────────────────────────────────────────────────────

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = f"{_REST_BASE}{path}"
        p = dict(params or {})
        p["apiKey"] = self.api_key
        resp = self._session.get(url, params=p, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def get_options_chain(self, symbol: str) -> list[OptionContract]:
        """
        Fetch the full options chain from /v3/snapshot/options/{symbol}.

        Paginates automatically until all contracts are retrieved.
        Greeks (delta, gamma, theta, vega) are confirmed working on liquid
        contracts at the paid tier.

        Returns:
            List of OptionContract sorted by (expiry, strike, option_type).
        """
        contracts: list[OptionContract] = []
        cursor: str | None = None

        while True:
            params: dict = {"limit": 250}
            if cursor:
                params["cursor"] = cursor

            data = self._get(f"/v3/snapshot/options/{symbol}", params)

            for result in data.get("results", []):
                details = result.get("details") or {}
                greeks = result.get("greeks") or {}
                quote = result.get("last_quote") or {}
                day = result.get("day") or {}

                raw_ticker = details.get("ticker", "")
                try:
                    _, expiry, option_type, strike = self.parse_option_ticker(raw_ticker)
                except (ValueError, IndexError):
                    continue

                contracts.append(OptionContract(
                    ticker=raw_ticker,
                    expiry=expiry,
                    strike=strike,
                    option_type=option_type,
                    bid=_clean(quote.get("bid")),
                    ask=_clean(quote.get("ask")),
                    volume=_clean(day.get("volume")),
                    open_interest=_clean(result.get("open_interest")),
                    iv=_clean(result.get("implied_volatility")),
                    delta=_clean(greeks.get("delta")),
                    gamma=_clean(greeks.get("gamma")),
                    theta=_clean(greeks.get("theta")),
                    vega=_clean(greeks.get("vega")),
                ))

            cursor = data.get("next_cursor")
            if not cursor:
                break

        contracts.sort(key=lambda c: (c.expiry, c.strike, c.option_type))
        return contracts

    def _get_chain_cached(self, symbol: str) -> list[OptionContract]:
        """Return chain from in-memory cache or fetch from REST (once per instance)."""
        if symbol not in self._chain_cache:
            self._chain_cache[symbol] = self.get_options_chain(symbol)
        return self._chain_cache[symbol]

    # ── Analysis ────────────────────────────────────────────────────────────────

    def get_put_call_ratio(
        self,
        symbol: str,
        spot_price: float = 0.0,
        atm_band_pct: float = 0.05,
    ) -> float:
        """
        Total put OI / total call OI for ATM strikes within ±atm_band_pct of spot.

        When spot_price is 0 or not provided, uses the full chain (all strikes).
        Prefers open_interest; falls back to volume.

        Returns 1.0 (neutral) when call activity is zero.
        """
        chain = self._get_chain_cached(symbol)
        if not chain:
            return 1.0

        if spot_price > 0:
            lo, hi = spot_price * (1 - atm_band_pct), spot_price * (1 + atm_band_pct)
            calls = [c for c in chain if c.option_type == "C" and lo <= c.strike <= hi]
            puts  = [c for c in chain if c.option_type == "P" and lo <= c.strike <= hi]
        else:
            calls = [c for c in chain if c.option_type == "C"]
            puts  = [c for c in chain if c.option_type == "P"]

        call_act = sum(c.activity for c in calls)
        put_act  = sum(c.activity for c in puts)

        if call_act <= 0:
            return 1.0
        return round(put_act / call_act, 4)

    def get_iv_skew(self, symbol: str, spot_price: float) -> float:
        """
        OTM call IV vs OTM put IV skew score, returns 0.0–1.0.

        Positive skew (> 0.5): calls more expensive than puts → bullish positioning.
        Uses strikes > spot×1.05 for calls, < spot×0.95 for puts.
        Returns 0.5 (neutral) when IV data is insufficient.
        """
        chain = self._get_chain_cached(symbol)

        otm_call_ivs = [
            c.iv for c in chain
            if c.option_type == "C"
            and c.strike > spot_price * 1.05
            and c.iv is not None and c.iv > 0
        ]
        otm_put_ivs = [
            c.iv for c in chain
            if c.option_type == "P"
            and c.strike < spot_price * 0.95
            and c.iv is not None and c.iv > 0
        ]

        if not otm_call_ivs or not otm_put_ivs:
            return 0.5

        avg_call_iv = sum(otm_call_ivs) / len(otm_call_ivs)
        avg_put_iv  = sum(otm_put_ivs)  / len(otm_put_ivs)

        if avg_put_iv <= 0:
            return 0.5

        skew_ratio = avg_call_iv / avg_put_iv
        score = 0.5 + 0.5 * math.tanh((skew_ratio - 1.0) * 2.0)
        return round(max(0.0, min(1.0, score)), 4)

    def get_unusual_call_activity(self, symbol: str) -> list[OptionContract]:
        """
        Return call contracts with volume > 2× average call volume across the chain.

        Unusual call activity may indicate targeted institutional accumulation.
        Returns empty list when fewer than 2 call strikes have volume data.
        """
        chain = self._get_chain_cached(symbol)
        calls = [c for c in chain if c.option_type == "C" and c.volume and c.volume > 0]

        if len(calls) < 2:
            return []

        avg_vol = sum(c.volume for c in calls) / len(calls)  # type: ignore[arg-type]
        if avg_vol <= 0:
            return []

        return [c for c in calls if c.volume is not None and c.volume > 2.0 * avg_vol]

    def compute_options_score(self, symbol: str, spot_price: float) -> float:
        """
        Composite PCR + IV skew + unusual calls into a 0.0–1.0 score.

        Scoring weights:
            PCR < 0.50 (strongly bullish)  : +0.60
            PCR < 0.70 (moderately bullish): +0.40
            Unusual call activity (2× avg) : +0.25
            IV skew > 0.60 (calls pricey)  : +0.15

        Checks DuckDB cache first; re-fetches only when today's row is absent.
        Caches computed score on success.

        Returns 0.0 on fetch failure (non-fatal).
        """
        cached = self._load_cache(symbol)
        if cached is not None:
            logger.debug("%s: options_score from cache: %.4f", symbol, cached)
            return cached

        try:
            self._get_chain_cached(symbol)
        except Exception as exc:
            logger.warning("%s: options chain fetch failed: %s", symbol, exc)
            return 0.0

        pcr   = self.get_put_call_ratio(symbol, spot_price)
        skew  = self.get_iv_skew(symbol, spot_price)
        unusual = self.get_unusual_call_activity(symbol)

        score = 0.0
        if pcr < 0.50:
            score += 0.60
        elif pcr < 0.70:
            score += 0.40
        if unusual:
            score += 0.25
        if skew > 0.60:
            score += 0.15

        result_score = round(min(1.0, score), 4)

        chain = self._chain_cache.get(symbol, [])
        self._save_cache(
            symbol=symbol,
            spot_price=spot_price,
            pcr=pcr,
            iv_skew=skew,
            unusual_calls=bool(unusual),
            options_score=result_score,
            call_count=sum(1 for c in chain if c.option_type == "C"),
            put_count=sum(1 for c in chain if c.option_type == "P"),
        )

        logger.info(
            "%s: options_score=%.2f  pcr=%.2f  iv_skew=%.2f  unusual=%s",
            symbol, result_score, pcr, skew, bool(unusual),
        )
        return result_score

    def compute_relative_options_score(
        self,
        symbol: str,
        spot_price: float,
        baseline_call_volumes: list[float],
    ) -> Optional[dict]:
        """
        Relative (per-symbol baseline) options score — the recalibrated path.

        Today's total call volume is z-scored against the symbol's OWN
        trailing baseline (build it with
        FlatFileProvider.get_call_volume_history) and blended with continuous
        PCR / IV-skew tilts — see quantlab.signals.options_relative.

        The cross-sectional gate (top decile of the day's scores) runs AFTER
        every symbol is scored: unusual_flag is left NULL here and set by
        mark_unusual_flags() in the monitor's second pass.

        DuckDB day-cache: when today's row already carries a rel_score the
        stored values are returned without a REST fetch — same once-per-day
        economy as the legacy path.  Intraday chain volumes are partial,
        which depresses z-scores roughly uniformly across symbols; the gate
        is rank-based, so relative ordering survives.

        Legacy fields (pcr, iv_skew, unusual_calls, options_score) are
        computed from the same chain and stored alongside so the old and new
        distributions stay comparable in options_snapshots.

        Returns:
            dict with keys call_volume, put_volume, vol_zscore, rel_score,
            pcr, iv_skew — vol_zscore/rel_score are None when the baseline is
            too short (MISSING ≠ ZERO).  None on chain fetch failure.
        """
        from quantlab.signals.options_relative import (
            relative_options_score,
            volume_zscore,
        )

        cached = self._load_relative_cache(symbol)
        if cached is not None:
            logger.debug("%s: relative options score from cache", symbol)
            return cached

        try:
            chain = self._get_chain_cached(symbol)
        except Exception as exc:
            logger.warning("%s: options chain fetch failed: %s", symbol, exc)
            return None

        call_volume = sum(c.volume or 0.0 for c in chain if c.option_type == "C")
        put_volume  = sum(c.volume or 0.0 for c in chain if c.option_type == "P")

        pcr  = self.get_put_call_ratio(symbol, spot_price)
        skew = self.get_iv_skew(symbol, spot_price)
        unusual_legacy = bool(self.get_unusual_call_activity(symbol))

        legacy_score = 0.0
        if pcr < 0.50:
            legacy_score += 0.60
        elif pcr < 0.70:
            legacy_score += 0.40
        if unusual_legacy:
            legacy_score += 0.25
        if skew > 0.60:
            legacy_score += 0.15
        legacy_score = round(min(1.0, legacy_score), 4)

        vol_z = volume_zscore(call_volume, baseline_call_volumes)
        rel   = relative_options_score(vol_z, pcr=pcr, iv_skew=skew)

        self._save_relative_cache(
            symbol=symbol,
            spot_price=spot_price,
            pcr=pcr,
            iv_skew=skew,
            unusual_calls=unusual_legacy,
            options_score=legacy_score,
            call_count=sum(1 for c in chain if c.option_type == "C"),
            put_count=sum(1 for c in chain if c.option_type == "P"),
            call_volume=call_volume,
            put_volume=put_volume,
            vol_zscore=vol_z,
            rel_score=rel,
        )

        logger.info(
            "%s: rel_score=%s  vol_z=%s  call_vol=%.0f  pcr=%.2f  iv_skew=%.2f",
            symbol,
            f"{rel:.4f}" if rel is not None else "None",
            f"{vol_z:.2f}" if vol_z is not None else "None",
            call_volume, pcr, skew,
        )
        return {
            "call_volume": call_volume,
            "put_volume":  put_volume,
            "vol_zscore":  vol_z,
            "rel_score":   rel,
            "pcr":         pcr,
            "iv_skew":     skew,
        }

    def mark_unusual_flags(
        self,
        flagged: set[str],
        snap_date: Optional[date] = None,
        put_dominated: Optional[set[str]] = None,
    ) -> None:
        """
        Persist the day's cross-sectional gate result to options_snapshots.

        Sets unusual_flag TRUE for ``flagged`` symbols and FALSE for every
        other row scored that day (rel_score NOT NULL).  Rows that were never
        scored keep unusual_flag NULL — MISSING ≠ ZERO.

        ``put_dominated`` symbols cleared the volume/liquidity gates but were
        blocked by the PCR ceiling — tagged TRUE (other scored rows FALSE,
        unscored NULL) as future short-side signal data.
        """
        try:
            import duckdb
            from quantlab.storage import DB_PATH

            con = duckdb.connect(str(DB_PATH))
            self._ensure_table(con)
            day = snap_date or date.today()
            con.execute(
                """
                UPDATE options_snapshots SET unusual_flag = FALSE
                WHERE snap_date = ? AND rel_score IS NOT NULL
                """,
                [day],
            )
            if flagged:
                placeholders = ", ".join("?" for _ in flagged)
                con.execute(
                    f"""
                    UPDATE options_snapshots SET unusual_flag = TRUE
                    WHERE snap_date = ? AND symbol IN ({placeholders})
                    """,
                    [day, *flagged],
                )
            if put_dominated is not None:
                con.execute(
                    """
                    UPDATE options_snapshots SET put_dominated = FALSE
                    WHERE snap_date = ? AND rel_score IS NOT NULL
                    """,
                    [day],
                )
                if put_dominated:
                    placeholders = ", ".join("?" for _ in put_dominated)
                    con.execute(
                        f"""
                        UPDATE options_snapshots SET put_dominated = TRUE
                        WHERE snap_date = ? AND symbol IN ({placeholders})
                        """,
                        [day, *put_dominated],
                    )
            self._update_flag_freshness(con, day, flagged)
            con.close()
        except Exception as exc:
            logger.warning("mark_unusual_flags failed: %s", exc)

    @staticmethod
    def _update_flag_freshness(con, day: date, flagged: set[str]) -> None:
        """
        Persist first_flagged_date / flag_streak for every row gated on
        ``day`` (rel_score NOT NULL).  Flagged rows get an episode start date
        and a streak ≥ 1; gated-unflagged rows get (NULL, 0); ungated rows
        stay NULL.  Gate-refused / degenerate-universe dates are neutral —
        same skip_dates convention as watchlist.remove_stale.
        """
        from datetime import timedelta

        from quantlab.signals.options_relative import flag_freshness
        from quantlab.utils import get_config
        from quantlab.watchlist import _degenerate_build_dates

        lapse = int(get_config("scanner").get(
            "options_flag_episode_lapse_sessions", 3))
        skip = _degenerate_build_dates(con)

        def _as_date(v):
            if v is None or isinstance(v, date):
                return v
            if hasattr(v, "date"):
                return v.date()
            return date.fromisoformat(str(v)[:10])

        # Prior-session flag history for every symbol (60 calendar days is
        # ample for a 3-session lapse + any realistic streak)
        hist_rows = con.execute(
            """
            SELECT symbol, snap_date, unusual_flag, first_flagged_date
            FROM options_snapshots
            WHERE snap_date < ? AND snap_date >= ?
            ORDER BY snap_date
            """,
            [day, day - timedelta(days=60)],
        ).fetchall()
        history: dict[str, list] = {}
        for sym, d, fl, ff in hist_rows:
            history.setdefault(sym, []).append(
                (_as_date(d), fl, _as_date(ff))
            )

        gated_today = [r[0] for r in con.execute(
            "SELECT symbol FROM options_snapshots "
            "WHERE snap_date = ? AND rel_score IS NOT NULL",
            [day],
        ).fetchall()]

        for sym in gated_today:
            first, streak = flag_freshness(
                sym in flagged, day, history.get(sym, []),
                lapse_sessions=lapse, skip_dates=skip,
            )
            con.execute(
                """
                UPDATE options_snapshots
                SET first_flagged_date = ?, flag_streak = ?
                WHERE symbol = ? AND snap_date = ?
                """,
                [first, streak, sym, day],
            )

    def _load_relative_cache(self, symbol: str) -> Optional[dict]:
        """Return today's stored relative-score row, or None when not yet scored."""
        try:
            import duckdb
            from quantlab.storage import DB_PATH

            con = duckdb.connect(str(DB_PATH))
            self._ensure_table(con)
            row = con.execute(
                """
                SELECT call_volume, put_volume, vol_zscore, rel_score, pcr, iv_skew
                FROM options_snapshots
                WHERE symbol = ? AND snap_date = CURRENT_DATE
                  AND rel_score IS NOT NULL
                """,
                [symbol],
            ).fetchone()
            con.close()
            if row is None:
                return None
            keys = ("call_volume", "put_volume", "vol_zscore",
                    "rel_score", "pcr", "iv_skew")
            return dict(zip(keys, row))
        except Exception as exc:
            logger.debug("relative cache read failed for %s: %s", symbol, exc)
            return None

    def _save_relative_cache(self, symbol: str, spot_price: float, pcr: float,
                             iv_skew: float, unusual_calls: bool,
                             options_score: float, call_count: int,
                             put_count: int, call_volume: float,
                             put_volume: float, vol_zscore: Optional[float],
                             rel_score: Optional[float]) -> None:
        """Persist today's full snapshot row (unusual_flag stays NULL). Non-fatal."""
        try:
            import duckdb
            from quantlab.storage import DB_PATH

            con = duckdb.connect(str(DB_PATH))
            self._ensure_table(con)
            con.execute(
                """
                INSERT OR REPLACE INTO options_snapshots
                    (symbol, snap_date, spot_price, pcr, iv_skew,
                     unusual_calls, options_score, call_count, put_count,
                     call_volume, put_volume, vol_zscore, rel_score, unusual_flag)
                VALUES (?, CURRENT_DATE, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                [
                    symbol, spot_price, pcr, iv_skew,
                    unusual_calls, options_score, call_count, put_count,
                    call_volume, put_volume, vol_zscore, rel_score,
                ],
            )
            con.close()
        except Exception as exc:
            logger.warning("options_snapshots relative write failed for %s: %s", symbol, exc)

    # ── S3 / Historical OHLCV ──────────────────────────────────────────────────

    def get_historical_ohlcv(
        self,
        option_ticker: str,
        start_date: date,
        end_date: date,
    ) -> list[dict]:
        """
        Download daily OHLCV for an option contract.

        Tries Massive S3 flat files first (boto3 + pyarrow required).
        Falls back to Polygon REST /v2/aggs/ticker/ on any S3 failure.

        S3 path pattern:
            {bucket}/{prefix}/{YYYY}/{MM}/{DD}/*.parquet

        Schema: ticker, volume, open, close, high, low, window_start, transactions.

        Returns:
            List of dicts with keys: date, open, high, low, close, volume, transactions.
        """
        try:
            return self._s3_historical(option_ticker, start_date, end_date)
        except Exception as exc:
            logger.debug(
                "S3 historical options failed for %s: %s — falling back to REST",
                option_ticker, exc,
            )
        return self._rest_historical(option_ticker, start_date, end_date)

    def _s3_historical(
        self,
        option_ticker: str,
        start_date: date,
        end_date: date,
    ) -> list[dict]:
        """Download from Massive S3 flat files via boto3 + pyarrow."""
        import boto3
        import io
        import pyarrow.parquet as pq

        s3 = boto3.client(
            "s3",
            endpoint_url=self.s3_endpoint,
            aws_access_key_id=self.s3_access_key or None,
            aws_secret_access_key=self.s3_secret_key or None,
        )

        results: list[dict] = []
        current = start_date
        while current <= end_date:
            prefix = (
                f"{_S3_OPTIONS_PREFIX}"
                f"/{current.year}/{current.month:02d}/{current.day:02d}/"
            )
            try:
                paginator = s3.get_paginator("list_objects_v2")
                for page in paginator.paginate(Bucket=self.s3_bucket, Prefix=prefix):
                    for obj in page.get("Contents", []):
                        if not obj["Key"].endswith(".parquet"):
                            continue
                        body = s3.get_object(
                            Bucket=self.s3_bucket, Key=obj["Key"]
                        )["Body"].read()
                        df = pq.read_table(io.BytesIO(body)).to_pydict()
                        tickers = df.get("ticker", [])
                        for i, t in enumerate(tickers):
                            if t == option_ticker:
                                results.append({
                                    "date": current,
                                    "open":  df["open"][i],
                                    "high":  df["high"][i],
                                    "low":   df["low"][i],
                                    "close": df["close"][i],
                                    "volume": df["volume"][i],
                                    "transactions": df.get(
                                        "transactions",
                                        [None] * len(tickers)
                                    )[i],
                                })
            except Exception:
                pass  # missing partition (weekend / holiday) — skip silently
            current += timedelta(days=1)

        return results

    def _rest_historical(
        self,
        option_ticker: str,
        start_date: date,
        end_date: date,
    ) -> list[dict]:
        """Fetch from Polygon REST /v2/aggs/ticker/{ticker}/range/1/day."""
        encoded = option_ticker.replace(":", "%3A")
        path = (
            f"/v2/aggs/ticker/{encoded}/range/1/day"
            f"/{start_date.isoformat()}/{end_date.isoformat()}"
        )
        data = self._get(path, {"adjusted": "true", "sort": "asc", "limit": 50000})

        results: list[dict] = []
        for item in data.get("results", []):
            t_ms = item.get("t", 0)
            from datetime import timezone as _tz
            bar_date = datetime.fromtimestamp(t_ms / 1000, tz=_tz.utc).date() if t_ms else start_date
            if start_date <= bar_date <= end_date:
                results.append({
                    "date":   bar_date,
                    "open":   item.get("o"),
                    "high":   item.get("h"),
                    "low":    item.get("l"),
                    "close":  item.get("c"),
                    "volume": item.get("v"),
                    "transactions": item.get("n"),
                })
        return results

    # ── DuckDB cache ───────────────────────────────────────────────────────────

    @staticmethod
    def _ensure_table(con) -> None:
        con.execute("""
            CREATE TABLE IF NOT EXISTS options_snapshots (
                symbol        VARCHAR,
                snap_date     DATE,
                spot_price    DOUBLE,
                pcr           DOUBLE,
                iv_skew       DOUBLE,
                unusual_calls BOOLEAN,
                options_score DOUBLE,
                call_count    INTEGER,
                put_count     INTEGER,
                PRIMARY KEY (symbol, snap_date)
            )
        """)
        # Relative-scoring columns (2026-06 recalibration).  Nullable with no
        # defaults — MISSING ≠ ZERO: a NULL rel_score means "not scored", a
        # NULL unusual_flag means "not gated", never 0/false.
        for col, col_type in (
            ("call_volume", "DOUBLE"),
            ("put_volume",  "DOUBLE"),
            ("vol_zscore",  "DOUBLE"),
            ("rel_score",   "DOUBLE"),
            ("unusual_flag", "BOOLEAN"),
            # Cleared the volume/liquidity gates but PCR > ceiling — a
            # put-dominated anomaly kept as future short-side signal data
            ("put_dominated", "BOOLEAN"),
            # Flag freshness: episode start and consecutive flagged sessions
            # (0 on gated-unflagged rows, NULL on ungated — MISSING ≠ ZERO)
            ("first_flagged_date", "DATE"),
            ("flag_streak", "INTEGER"),
        ):
            con.execute(
                f"ALTER TABLE options_snapshots ADD COLUMN IF NOT EXISTS {col} {col_type}"
            )

    def _load_cache(self, symbol: str) -> Optional[float]:
        """Return today's cached options_score or None."""
        try:
            import duckdb
            from quantlab.storage import DB_PATH

            con = duckdb.connect(str(DB_PATH))
            self._ensure_table(con)
            row = con.execute(
                """
                SELECT options_score FROM options_snapshots
                WHERE symbol = ? AND snap_date = CURRENT_DATE
                """,
                [symbol],
            ).fetchone()
            con.close()
            return float(row[0]) if row is not None else None
        except Exception as exc:
            logger.debug("options_snapshots cache read failed for %s: %s", symbol, exc)
            return None

    def _save_cache(
        self,
        symbol: str,
        spot_price: float,
        pcr: float,
        iv_skew: float,
        unusual_calls: bool,
        options_score: float,
        call_count: int,
        put_count: int,
    ) -> None:
        """Persist options metrics to DuckDB. Non-fatal."""
        try:
            import duckdb
            from quantlab.storage import DB_PATH

            con = duckdb.connect(str(DB_PATH))
            self._ensure_table(con)
            con.execute(
                """
                INSERT OR REPLACE INTO options_snapshots
                    (symbol, snap_date, spot_price, pcr, iv_skew,
                     unusual_calls, options_score, call_count, put_count)
                VALUES (?, CURRENT_DATE, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    symbol, spot_price, pcr, iv_skew,
                    unusual_calls, options_score, call_count, put_count,
                ],
            )
            con.close()
        except Exception as exc:
            logger.warning("options_snapshots cache write failed for %s: %s", symbol, exc)
