"""
IC Monitor — rolling information coefficient per signal.

Correlates conviction scorer signal values (captured in scan_results at scan
time) with realized forward returns (captured in watchlist post-close).

Designed to run after track_forward_returns.py updates the watchlist.
Meaningful IC estimates accumulate over ~60 trading days of full-universe scans.
"""

from __future__ import annotations

import argparse
import logging
import math
from datetime import date, timedelta
from typing import Optional

from quantlab.storage import get_db

logger = logging.getLogger(__name__)

# Maps IC monitor signal names → scan_results column expressions
SIGNAL_COLUMNS: dict[str, str] = {
    "ar":                    "accumulation_ratio",
    "vt":                    "volume_trend",
    "cv":                    "climactic_volume",
    "rs":                    "rs_score",
    "edgar_accel":           "edgar_acceleration",
    "ohlcv_accel":           "earnings_acceleration",
    "breakout_volume_score": "breakout_volume_score",
    "peg_score":             "peg_score",
    "stage":                 "stage",
    "regime_score":          "CAST(regime_bullish AS INTEGER)",
}

# Maps horizon (trading days) → watchlist return column
HORIZON_COLUMNS: dict[int, str] = {
    1:  "realized_ret_1d",
    5:  "realized_ret_5d",
    10: "realized_ret_10d",
}

MIN_OBSERVATIONS = 30


class ICMonitor:
    """Rolling information coefficient monitor for conviction scorer signals."""

    def compute_ic(
        self,
        signal_name: str,
        horizon: int,
        lookback_days: int = 60,
    ) -> Optional[float]:
        """
        Compute Pearson IC between signal_name and forward returns at horizon.

        Queries scan_results joined with watchlist over the lookback window.
        Returns None if fewer than MIN_OBSERVATIONS (30) pairs are available.
        Stores result in signal_ic_history when IC is computed.
        """
        if signal_name not in SIGNAL_COLUMNS:
            logger.warning("Unknown signal: %s", signal_name)
            return None
        if horizon not in HORIZON_COLUMNS:
            logger.warning("Unknown horizon: %d  (supported: %s)", horizon, sorted(HORIZON_COLUMNS))
            return None

        ret_col = HORIZON_COLUMNS[horizon]
        # regime_score uses a SQL expression, not a bare column name
        is_expr = signal_name == "regime_score"
        sig_expr = SIGNAL_COLUMNS[signal_name]
        sig_ref = sig_expr if is_expr else f"sr.{sig_expr}"

        cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
        today_str = date.today().isoformat()

        try:
            con = get_db()
            rows = con.execute(f"""
                SELECT {sig_ref} AS signal_val,
                       w.{ret_col}  AS ret_val
                FROM scan_results sr
                JOIN watchlist w
                  ON sr.symbol    = w.symbol
                 AND sr.scan_date = w.date_added
                WHERE sr.scan_date >= ?
                  AND {sig_ref} IS NOT NULL
                  AND w.{ret_col} IS NOT NULL
            """, [cutoff]).fetchall()
            con.close()
        except Exception as e:
            logger.error("IC query failed for %s H=%d: %s", signal_name, horizon, e)
            return None

        n = len(rows)
        if n < MIN_OBSERVATIONS:
            logger.info(
                "Insufficient data for %s H=%d: %d observations (need %d)",
                signal_name, horizon, n, MIN_OBSERVATIONS,
            )
            return None

        xs = [float(r[0]) for r in rows]
        ys = [float(r[1]) for r in rows]

        ic = _pearson(xs, ys)
        if ic is None:
            return None

        ic_ir = self._compute_ic_ir(signal_name, horizon, ic, lookback_days)
        flagged = self._is_flagged(signal_name, horizon, ic, threshold=0.02)
        mean_sig = sum(xs) / n
        mean_ret = sum(ys) / n

        try:
            con = get_db()
            con.execute("""
                INSERT OR REPLACE INTO signal_ic_history
                    (computed_date, signal_name, horizon, ic, ic_ir,
                     n_observations, mean_signal, mean_return, flagged)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [today_str, signal_name, horizon, ic, ic_ir,
                  n, mean_sig, mean_ret, flagged])
            con.close()
        except Exception as e:
            logger.error("IC history insert failed: %s", e)

        logger.debug(
            "IC %s H=%d: %.4f  n=%d  flagged=%s",
            signal_name, horizon, ic, n, flagged,
        )
        return ic

    def compute_all(
        self,
        horizons: list[int] | None = None,
        lookback_days: int = 60,
    ) -> dict[str, dict[int, Optional[float]]]:
        """
        Run compute_ic for every tracked signal at each horizon.

        Returns nested dict: {signal_name: {horizon: ic_or_None}}.
        Logs a summary table of results.
        """
        if horizons is None:
            horizons = [1, 5, 10]

        results: dict[str, dict[int, Optional[float]]] = {}
        for signal_name in SIGNAL_COLUMNS:
            results[signal_name] = {}
            for h in horizons:
                results[signal_name][h] = self.compute_ic(signal_name, h, lookback_days)

        header = f"{'Signal':<25} {'H=1':>8} {'H=5':>8} {'H=10':>8}"
        logger.info(header)
        logger.info("-" * len(header))
        for sig, h_map in results.items():
            def _fmtic(v: Optional[float]) -> str:
                return f"{v:8.4f}" if v is not None else "     N/A"
            logger.info(
                "%-25s %s %s %s",
                sig,
                _fmtic(h_map.get(1)),
                _fmtic(h_map.get(5)),
                _fmtic(h_map.get(10)),
            )

        return results

    def ic_decay_curve(
        self,
        signal_name: str,
        lookback_days: int = 60,
    ) -> list[tuple[int, Optional[float]]]:
        """Return [(horizon, ic), ...] for H=1, 5, 10."""
        return [
            (h, self.compute_ic(signal_name, h, lookback_days))
            for h in [1, 5, 10]
        ]

    def flag_weak_signals(self, threshold: float = 0.02) -> list[str]:
        """
        Return signal names where 60-day rolling mean IC < threshold.

        Signals returned here should be reviewed or reweighted in score_conviction.
        """
        cutoff = (date.today() - timedelta(days=60)).isoformat()
        weak: list[str] = []
        try:
            con = get_db()
            rows = con.execute("""
                SELECT signal_name, AVG(ic) AS mean_ic
                FROM signal_ic_history
                WHERE computed_date >= ?
                GROUP BY signal_name
                HAVING AVG(ic) < ?
            """, [cutoff, threshold]).fetchall()
            con.close()
            weak = [r[0] for r in rows]
        except Exception as e:
            logger.error("flag_weak_signals query failed: %s", e)
        return weak

    def summary_report(self) -> None:
        """
        Print formatted IC summary table.

        Status categories:
            STRONG      IC > 0.05
            ADEQUATE    0.02 ≤ IC ≤ 0.05
            WEAK        IC < 0.02
            INSUFFICIENT  < 30 observations (no IC computed)
        """
        col_w = 25 + 10 + 10 + 10 + 8 + 14
        header = (
            f"{'Signal':<25} {'H=1 IC':>10} {'H=5 IC':>10} "
            f"{'H=10 IC':>10} {'IC IR':>8} {'Status':>14}"
        )
        print(header)
        print("-" * len(header))

        today_str = date.today().isoformat()

        for signal_name in SIGNAL_COLUMNS:
            h1 = h5 = h10 = ic_ir = None
            try:
                con = get_db()
                rows = con.execute("""
                    SELECT horizon, ic, ic_ir
                    FROM signal_ic_history
                    WHERE signal_name = ? AND computed_date = ?
                """, [signal_name, today_str]).fetchall()
                con.close()
                for row in rows:
                    if row[0] == 1:
                        h1 = row[1]
                        ic_ir = row[2]
                    elif row[0] == 5:
                        h5 = row[1]
                    elif row[0] == 10:
                        h10 = row[1]
            except Exception:
                pass

            if h1 is None and h5 is None and h10 is None:
                status = "INSUFFICIENT"
            elif h1 is not None and h1 > 0.05:
                status = "STRONG"
            elif h1 is not None and h1 >= 0.02:
                status = "ADEQUATE"
            else:
                status = "WEAK"

            def _fmt_ic(v: Optional[float]) -> str:
                return f"{v:10.4f}" if v is not None else f"{'N/A':>10}"

            def _fmt_ir(v: Optional[float]) -> str:
                return f"{v:8.3f}" if v is not None else f"{'N/A':>8}"

            print(
                f"{signal_name:<25} {_fmt_ic(h1)} {_fmt_ic(h5)} {_fmt_ic(h10)} "
                f"{_fmt_ir(ic_ir)} {status:>14}"
            )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _compute_ic_ir(
        self,
        signal_name: str,
        horizon: int,
        current_ic: float,
        lookback_days: int,
    ) -> Optional[float]:
        """IC divided by rolling standard deviation of IC (information ratio)."""
        cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
        try:
            con = get_db()
            rows = con.execute("""
                SELECT ic FROM signal_ic_history
                WHERE signal_name = ? AND horizon = ? AND computed_date >= ?
            """, [signal_name, horizon, cutoff]).fetchall()
            con.close()
        except Exception:
            return None

        ics = [r[0] for r in rows if r[0] is not None] + [current_ic]
        if len(ics) < 2:
            return None

        mean_ic = sum(ics) / len(ics)
        std_ic = math.sqrt(
            sum((x - mean_ic) ** 2 for x in ics) / (len(ics) - 1)
        )
        if std_ic == 0:
            return None
        return mean_ic / std_ic

    def _is_flagged(
        self,
        signal_name: str,
        horizon: int,
        current_ic: float,
        threshold: float = 0.02,
    ) -> bool:
        """True if 60-day rolling mean IC (including current) < threshold."""
        cutoff = (date.today() - timedelta(days=60)).isoformat()
        try:
            con = get_db()
            rows = con.execute("""
                SELECT ic FROM signal_ic_history
                WHERE signal_name = ? AND horizon = ? AND computed_date >= ?
            """, [signal_name, horizon, cutoff]).fetchall()
            con.close()
        except Exception:
            return False

        ics = [r[0] for r in rows if r[0] is not None] + [current_ic]
        if not ics:
            return False
        return (sum(ics) / len(ics)) < threshold


# ── Pearson correlation ───────────────────────────────────────────────────────

def _pearson(xs: list[float], ys: list[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    var_x = sum((x - mx) ** 2 for x in xs)
    var_y = sum((y - my) ** 2 for y in ys)
    denom = math.sqrt(var_x * var_y)
    if denom == 0:
        return None
    return cov / denom


# ── CLI entry point ───────────────────────────────────────────────────────────

def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    _setup_logging()

    parser = argparse.ArgumentParser(
        description="IC Monitor — rolling information coefficient per signal"
    )
    parser.add_argument(
        "--horizon", type=int, nargs="+", default=[1, 5, 10],
        help="Forward return horizons in trading days (default: 1 5 10)",
    )
    parser.add_argument(
        "--lookback", type=int, default=60,
        help="Lookback window in calendar days (default: 60)",
    )
    parser.add_argument(
        "--signal", type=str, default=None,
        help="Single signal name to compute IC for (default: all signals)",
    )
    args = parser.parse_args()

    monitor = ICMonitor()

    if args.signal:
        for h in args.horizon:
            ic = monitor.compute_ic(args.signal, h, args.lookback)
            logger.info(
                "IC %s H=%d: %s",
                args.signal, h,
                f"{ic:.4f}" if ic is not None else "N/A (insufficient data)",
            )
    else:
        monitor.compute_all(horizons=args.horizon, lookback_days=args.lookback)

    monitor.summary_report()

    weak = monitor.flag_weak_signals()
    if weak:
        logger.warning("Weak signals (60d mean IC < 0.02): %s", ", ".join(weak))
    else:
        logger.info("No weak signals detected.")


if __name__ == "__main__":
    main()
