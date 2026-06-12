"""
Layer 5: Storage — DuckDB, Parquet, CSV.

Design:
    data/raw/          — untouched source files (never modify)
    data/processed/    — cleaned Parquet bar files per symbol
    output/            — backtest results, trade CSVs, charts
    quantlab.duckdb    — local research database (gitignored)

DuckDB is used lightly at first:
    - Each backtest run appends to the trades table
    - Bar data stored as Parquet, queried directly by DuckDB
    - Cross-run analysis and joins happen via SQL
"""

from __future__ import annotations

import csv
from dataclasses import asdict, fields
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from quantlab.research import TradeRecord, PerformanceMetrics
    from quantlab.providers.base import Bar

# ── Project paths ──────────────────────────────────────────────────────────────

def _project_root() -> Path:
    """Locate the project root by walking up from this file."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parents[3]  # fallback


PROJECT_ROOT = _project_root()
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"
DB_PATH = PROJECT_ROOT / "quantlab.duckdb"


def ensure_dirs() -> None:
    for d in [DATA_RAW, DATA_PROCESSED, OUTPUT_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ── Parquet bar storage ────────────────────────────────────────────────────────

def save_bars_parquet(symbol: str, bars: Sequence) -> Path:
    """
    Save bars to data/processed/<symbol>_bars.parquet via PyArrow.
    Falls back to CSV if PyArrow is not installed.
    """
    ensure_dirs()
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table(
            {
                "date": [b.as_of.isoformat() for b in bars],
                "open": [b.open for b in bars],
                "high": [b.high for b in bars],
                "low": [b.low for b in bars],
                "close": [b.close for b in bars],
                "volume": [b.volume for b in bars],
                "symbol": [symbol] * len(bars),
            }
        )
        path = DATA_PROCESSED / f"{symbol}_bars.parquet"
        pq.write_table(table, path)
        return path

    except ImportError:
        # Graceful fallback if PyArrow not yet installed
        return save_bars_csv(symbol, bars)


def save_bars_csv(symbol: str, bars: Sequence) -> Path:
    """Fallback: save bars as CSV to data/processed/<symbol>_bars.csv."""
    ensure_dirs()
    path = DATA_PROCESSED / f"{symbol}_bars.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["date", "open", "high", "low", "close", "volume", "symbol"]
        )
        writer.writeheader()
        for b in bars:
            writer.writerow(
                {
                    "date": b.as_of.isoformat(),
                    "open": b.open,
                    "high": b.high,
                    "low": b.low,
                    "close": b.close,
                    "volume": b.volume,
                    "symbol": symbol,
                }
            )
    return path


# ── Trade CSV export ───────────────────────────────────────────────────────────

TRADE_FIELDS = [
    "symbol", "signal_date", "entry_date", "entry_price",
    "exit_date", "exit_price", "trade_return",
    "ret_1d", "ret_3d", "ret_5d", "mfe_5d", "mae_5d",
    "atr_stop", "news_category", "news_count",
    "news_k_score", "news_c_score", "cost_bps",
]


def export_trades_csv(
    symbol: str,
    signal_type: str,
    trades: list,
    run_tag: str = "",
) -> Path:
    """Write trade records to output/<symbol>_<signal_type>_trades.csv."""
    ensure_dirs()
    suffix = f"_{run_tag}" if run_tag else ""
    path = OUTPUT_DIR / f"{symbol}_{signal_type}{suffix}_trades.csv"

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TRADE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for trade in trades:
            row = asdict(trade) if hasattr(trade, "__dataclass_fields__") else dict(trade)
            writer.writerow(row)

    return path


# ── DuckDB storage ─────────────────────────────────────────────────────────────

def get_db():
    """
    Return a DuckDB connection to the local research database.

    Creates the database and schema on first call.
    Usage::

        with get_db() as con:
            con.execute("SELECT * FROM trades LIMIT 10").fetchdf()
    """
    try:
        import duckdb

        con = duckdb.connect(str(DB_PATH))
        _ensure_schema(con)
        return con

    except ImportError:
        raise RuntimeError(
            "DuckDB is not installed. Run: pip install duckdb"
        )


def _ensure_schema(con) -> None:
    """Create tables if they do not exist."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            run_id          VARCHAR,
            symbol          VARCHAR,
            signal_type     VARCHAR,
            lookback        INTEGER,
            signal_date     DATE,
            entry_date      DATE,
            entry_price     DOUBLE,
            exit_date       DATE,
            exit_price      DOUBLE,
            trade_return    DOUBLE,
            ret_1d          DOUBLE,
            ret_3d          DOUBLE,
            ret_5d          DOUBLE,
            mfe_5d          DOUBLE,
            mae_5d          DOUBLE,
            atr_stop        DOUBLE,
            news_category   VARCHAR,
            news_count      INTEGER,
            news_k_score    DOUBLE,
            news_c_score    DOUBLE,
            cost_bps        DOUBLE,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS backtest_runs (
            run_id          VARCHAR PRIMARY KEY,
            symbol          VARCHAR,
            signal_type     VARCHAR,
            lookback        INTEGER,
            start_date      DATE,
            end_date        DATE,
            bar_count       INTEGER,
            trade_count     INTEGER,
            total_return    DOUBLE,
            max_drawdown    DOUBLE,
            sharpe_ratio    DOUBLE,
            sortino_ratio   DOUBLE,
            calmar_ratio    DOUBLE,
            profit_factor   DOUBLE,
            win_rate        DOUBLE,
            expectancy      DOUBLE,
            sufficient_sample BOOLEAN,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # scan_results: migrate to enriched schema if scan_id column is absent
    try:
        existing = {
            row[1]
            for row in con.execute("PRAGMA table_info(scan_results)").fetchall()
        }
        if "scan_id" not in existing:
            con.execute("DROP TABLE IF EXISTS scan_results")
    except Exception:
        pass

    con.execute("""
        CREATE TABLE IF NOT EXISTS scan_results (
            scan_id              VARCHAR,
            scan_date            DATE,
            symbol               VARCHAR,
            signal_type          VARCHAR,
            lookback             INTEGER,
            entry_close          DOUBLE,
            indicator_value      DOUBLE,
            conviction_score     DOUBLE,
            regime_bullish       BOOLEAN,
            -- news
            news_category        VARCHAR,
            news_count           INTEGER,
            news_c_score         DOUBLE,
            -- market
            rel_volume           DOUBLE,
            atr_stop             DOUBLE,
            -- wyckoff
            base_quality         DOUBLE,
            absorption           DOUBLE,
            volume_character     DOUBLE,
            wyckoff_spring       BOOLEAN,
            -- earnings acceleration
            earnings_acceleration DOUBLE,
            -- volume profile
            accumulation_ratio   DOUBLE,
            volume_trend         DOUBLE,
            climactic_volume     DOUBLE,
            sector               VARCHAR,
            sector_cluster       BOOLEAN DEFAULT FALSE,
            -- IC monitor signals
            rs_score             DOUBLE DEFAULT 0.0,
            edgar_acceleration   DOUBLE,
            breakout_volume_score DOUBLE DEFAULT 0.0,
            peg_score            DOUBLE DEFAULT 0.0,
            stage                INTEGER DEFAULT 0,
            created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Migration: add columns to scan_results if absent (handles pre-existing DBs)
    try:
        _sr_cols = {r[1] for r in con.execute("PRAGMA table_info(scan_results)").fetchall()}
        for _col, _dtype in [
            ("sector",                "VARCHAR"),
            ("sector_cluster",        "BOOLEAN"),
            ("rs_score",              "DOUBLE DEFAULT 0.0"),
            ("edgar_acceleration",    "DOUBLE"),
            ("breakout_volume_score", "DOUBLE DEFAULT 0.0"),
            ("peg_score",             "DOUBLE DEFAULT 0.0"),
            ("stage",                 "INTEGER DEFAULT 0"),
        ]:
            if _col not in _sr_cols:
                con.execute(f"ALTER TABLE scan_results ADD COLUMN {_col} {_dtype}")
    except Exception:
        pass

    con.execute("""
        CREATE TABLE IF NOT EXISTS walk_forward_windows (
            run_id              VARCHAR,
            symbol              VARCHAR,
            signal_type         VARCHAR,
            lookback            INTEGER,
            window_index        INTEGER,
            is_start_bar        INTEGER,
            is_end_bar          INTEGER,
            oos_start_bar       INTEGER,
            oos_end_bar         INTEGER,
            is_sharpe           DOUBLE,
            is_total_return     DOUBLE,
            is_trade_count      INTEGER,
            is_sufficient       BOOLEAN,
            oos_sharpe          DOUBLE,
            oos_total_return    DOUBLE,
            oos_trade_count     INTEGER,
            oos_sufficient      BOOLEAN,
            created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            watch_id            VARCHAR PRIMARY KEY,  -- symbol_YYYY-MM-DD
            symbol              VARCHAR,
            date_added          DATE,
            entry_price         DOUBLE,
            atr_stop            DOUBLE,
            conviction_score    DOUBLE,
            signal_layers       VARCHAR,   -- comma-separated labels of layers that fired
            lookback            INTEGER,
            signal_type         VARCHAR,
            -- Forward return tracking (filled by track_forward_returns.py)
            price_1d            DOUBLE,
            price_3d            DOUBLE,
            price_5d            DOUBLE,
            realized_ret_1d     DOUBLE,
            realized_ret_3d     DOUBLE,
            realized_ret_5d     DOUBLE,
            -- Live tracking
            current_price       DOUBLE,
            unrealized_ret      DOUBLE,
            days_on_watch       INTEGER DEFAULT 0,
            -- 2R price target (entry + 2*(entry - atr_stop)); NULL until computed
            target_price        DOUBLE,
            -- Status lifecycle: watching → triggered/stopped_out/expired/target_hit
            status              VARCHAR DEFAULT 'watching',
            date_updated        DATE,
            -- Audit / override notes
            breadth_override_note VARCHAR DEFAULT '',
            created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Migration: add columns to watchlist if absent (handles pre-existing DBs)
    try:
        _wl_cols = {r[1] for r in con.execute("PRAGMA table_info(watchlist)").fetchall()}
        for _col, _dtype in [
            ("breadth_override_note", "VARCHAR DEFAULT ''"),
            ("price_10d",             "DOUBLE"),
            ("realized_ret_10d",      "DOUBLE"),
        ]:
            if _col not in _wl_cols:
                con.execute(f"ALTER TABLE watchlist ADD COLUMN {_col} {_dtype}")
        if "target_price" not in _wl_cols:
            con.execute("ALTER TABLE watchlist ADD COLUMN target_price DOUBLE")
            con.execute("""
                UPDATE watchlist
                SET target_price = entry_price + 2 * (entry_price - atr_stop)
                WHERE atr_stop IS NOT NULL
                  AND entry_price IS NOT NULL
                  AND atr_stop < entry_price
            """)
    except Exception:
        pass


    con.execute("""
        CREATE TABLE IF NOT EXISTS breadth_history (
            date                    DATE PRIMARY KEY,
            advances                INTEGER,
            declines                INTEGER,
            unchanged               INTEGER,
            total_stocks            INTEGER,
            up_4pct_count           INTEGER,
            down_4pct_count         INTEGER,
            up_25pct_quarter        INTEGER,
            down_25pct_quarter      INTEGER,
            new_highs_52w           INTEGER,
            new_lows_52w            INTEGER,
            pct_above_10sma         DOUBLE  DEFAULT 0.0,
            pct_above_20sma         DOUBLE,
            pct_above_50sma         DOUBLE,
            pct_above_200sma        DOUBLE,
            advance_decline_ratio   DOUBLE,
            new_high_low_ratio      DOUBLE,
            ratio_10d               DOUBLE,
            mcclellan_oscillator    DOUBLE,
            mcclellan_summation     DOUBLE,
            ad_line                 INTEGER,
            tape                    VARCHAR DEFAULT 'NEUTRAL',
            spy_above_200sma        BOOLEAN DEFAULT TRUE,
            spy_200sma_slope        DOUBLE  DEFAULT 0.0,
            up_25pct_month          INTEGER DEFAULT 0,
            dn_25pct_month          INTEGER DEFAULT 0,
            up_50pct_month          INTEGER DEFAULT 0,
            dn_50pct_month          INTEGER DEFAULT 0,
            up_13pct_34d            INTEGER DEFAULT 0,
            dn_13pct_34d            INTEGER DEFAULT 0,
            uvol                    DOUBLE  DEFAULT 0.0,
            dvol                    DOUBLE  DEFAULT 0.0,
            uvol_dvol_ratio         DOUBLE  DEFAULT 0.0,
            equity_pcr              DOUBLE  DEFAULT 0.0,
            index_pcr               DOUBLE  DEFAULT 0.0,
            total_pcr               DOUBLE  DEFAULT 0.0,
            pcr_regime              VARCHAR DEFAULT 'neutral',
            spy_21ema               DOUBLE  DEFAULT 0.0,
            spy_50sma               DOUBLE  DEFAULT 0.0,
            spy_pct_above_21ema     DOUBLE  DEFAULT 0.0,
            spy_pct_above_50sma     DOUBLE  DEFAULT 0.0,
            spy_pct_above_200sma    DOUBLE  DEFAULT 0.0,
            created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Migration: add any absent breadth_history columns (handles pre-existing DBs)
    try:
        _bh_cols = {r[0] for r in con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'breadth_history'"
        ).fetchall()}
        for _col, _def in [
            ("spy_above_200sma",  "BOOLEAN DEFAULT TRUE"),
            ("spy_200sma_slope",   "DOUBLE DEFAULT 0.0"),
            ("pct_above_10sma",    "DOUBLE DEFAULT 0.0"),
            ("up_25pct_month",     "INTEGER DEFAULT 0"),
            ("dn_25pct_month",     "INTEGER DEFAULT 0"),
            ("up_50pct_month",     "INTEGER DEFAULT 0"),
            ("dn_50pct_month",     "INTEGER DEFAULT 0"),
            ("up_13pct_34d",       "INTEGER DEFAULT 0"),
            ("dn_13pct_34d",       "INTEGER DEFAULT 0"),
            ("uvol",               "DOUBLE DEFAULT 0.0"),
            ("dvol",               "DOUBLE DEFAULT 0.0"),
            ("uvol_dvol_ratio",    "DOUBLE DEFAULT 0.0"),
            ("equity_pcr",             "DOUBLE DEFAULT 0.0"),
            ("index_pcr",              "DOUBLE DEFAULT 0.0"),
            ("total_pcr",              "DOUBLE DEFAULT 0.0"),
            ("pcr_regime",             "VARCHAR DEFAULT 'neutral'"),
            ("spy_21ema",              "DOUBLE DEFAULT 0.0"),
            ("spy_50sma",              "DOUBLE DEFAULT 0.0"),
            ("spy_pct_above_21ema",    "DOUBLE DEFAULT 0.0"),
            ("spy_pct_above_50sma",    "DOUBLE DEFAULT 0.0"),
            ("spy_pct_above_200sma",   "DOUBLE DEFAULT 0.0"),
        ]:
            if _col not in _bh_cols:
                con.execute(f"ALTER TABLE breadth_history ADD COLUMN {_col} {_def}")
    except Exception:
        pass

    con.execute("""
        CREATE TABLE IF NOT EXISTS universe_history (
            date                DATE PRIMARY KEY,
            total_raw           INTEGER,
            after_price         INTEGER,
            after_volume        INTEGER,
            after_dollar_vol    INTEGER,
            after_symbol_filter INTEGER,
            optionable_count    INTEGER,
            final_count         INTEGER,
            min_price           DOUBLE,
            min_volume          DOUBLE,
            min_dollar_volume   DOUBLE,
            optionable_only     BOOLEAN,
            created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS institutional_watchlist (
            symbol                TEXT PRIMARY KEY,
            first_seen            DATE,
            last_seen             DATE,
            consecutive_days      INTEGER DEFAULT 1,
            stage                 INTEGER DEFAULT 0,
            conviction_score      FLOAT,
            entry_price           FLOAT,
            options_signal        BOOLEAN DEFAULT FALSE,
            volume_dry_up         BOOLEAN DEFAULT FALSE,
            earnings_score        FLOAT,
            peg_score             FLOAT,
            breakout_volume_score FLOAT,
            tape                  TEXT DEFAULT '',
            notes                 TEXT DEFAULT '',
            updated_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            explosion_score       FLOAT DEFAULT 0.0
        )
    """)

    # Migration: add explosion_score to institutional_watchlist for pre-existing DBs
    try:
        _iwl_cols = {r[1] for r in con.execute("PRAGMA table_info(institutional_watchlist)").fetchall()}
        if "explosion_score" not in _iwl_cols:
            con.execute("ALTER TABLE institutional_watchlist ADD COLUMN explosion_score FLOAT DEFAULT 0.0")
    except Exception:
        pass

    con.execute("""
        CREATE TABLE IF NOT EXISTS basing_watchlist (
            symbol                TEXT PRIMARY KEY,
            first_seen            DATE,
            last_seen             DATE,
            consecutive_days      INTEGER DEFAULT 1,
            stage                 INTEGER DEFAULT 0,
            conviction_score      FLOAT,
            entry_price           FLOAT,
            options_signal        BOOLEAN DEFAULT FALSE,
            volume_dry_up         BOOLEAN DEFAULT FALSE,
            earnings_score        FLOAT,
            peg_score             FLOAT,
            breakout_volume_score FLOAT,
            tape                  TEXT DEFAULT '',
            notes                 TEXT DEFAULT '',
            updated_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            explosion_score       FLOAT DEFAULT 0.0
        )
    """)

    # Migration: add explosion_score to basing_watchlist for pre-existing DBs
    try:
        _bw_cols = {r[1] for r in con.execute("PRAGMA table_info(basing_watchlist)").fetchall()}
        if "explosion_score" not in _bw_cols:
            con.execute("ALTER TABLE basing_watchlist ADD COLUMN explosion_score FLOAT DEFAULT 0.0")
    except Exception:
        pass

    con.execute("""
        CREATE TABLE IF NOT EXISTS daily_reports (
            date             DATE PRIMARY KEY,
            tape             TEXT,
            mcclellan        FLOAT,
            candidates       INTEGER,
            multi_day        INTEGER,
            top_symbols      TEXT,
            generated_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS signal_ic_history (
            computed_date    DATE,
            signal_name      TEXT,
            horizon          INTEGER,
            ic               DOUBLE,
            ic_ir            DOUBLE,
            n_observations   INTEGER,
            mean_signal      DOUBLE,
            mean_return      DOUBLE,
            flagged          BOOLEAN,
            PRIMARY KEY (computed_date, signal_name, horizon)
        )
    """)


def save_equity_curve_chart(
    equity_curve: list[float],
    bars: Sequence,
    symbol: str,
    signal_type: str,
    run_tag: str = "",
) -> Path:
    """
    Save an equity curve + drawdown chart to output/ as a PNG.

    Plots strategy equity alongside a buy-and-hold baseline, with a drawdown
    panel below. Requires matplotlib (already a project dependency).

    Args:
        equity_curve: Portfolio value at each bar, length == len(bars).
        bars:         OHLCV bar sequence used in the backtest.
        symbol:       Ticker symbol (used in title and filename).
        signal_type:  Signal name (used in title and filename).
        run_tag:      Optional suffix for the filename.

    Returns:
        Path to the saved PNG file.
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive — safe in scripts and tests
    import matplotlib.pyplot as plt

    ensure_dirs()
    suffix = f"_{run_tag}" if run_tag else ""
    path = OUTPUT_DIR / f"{symbol}_{signal_type}{suffix}_equity.png"

    dates = [b.as_of for b in bars]
    closes = [b.close for b in bars]
    bah = [c / closes[0] * equity_curve[0] for c in closes]

    peak = equity_curve[0]
    drawdowns = []
    for e in equity_curve:
        peak = max(peak, e)
        drawdowns.append((e / peak) - 1.0)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]}
    )

    ax1.plot(dates, equity_curve, color="steelblue", linewidth=1.5, label="Strategy")
    ax1.plot(dates, bah, color="gray", linewidth=1.0, alpha=0.55, label="Buy & Hold")
    ax1.set_title(f"{symbol} — {signal_type} equity curve")
    ax1.set_ylabel("Portfolio value ($)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(dates, drawdowns, 0, color="crimson", alpha=0.45, label="Drawdown")
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)

    return path


def append_backtest_run(
    run_id: str,
    symbol: str,
    signal_type: str,
    lookback: int,
    start_date,
    end_date,
    metrics,
) -> None:
    """
    Insert a PerformanceMetrics summary row into the backtest_runs DuckDB table.

    Storage errors are caught and printed — they never crash the research workflow.

    Args:
        run_id:      Unique identifier for this run (e.g. from make_run_id).
        symbol:      Ticker symbol.
        signal_type: "breakout" or "sma".
        lookback:    Signal lookback period.
        start_date:  First bar date.
        end_date:    Last bar date.
        metrics:     PerformanceMetrics instance from compute_metrics / run_backtest.
    """
    try:
        con = get_db()
        con.execute("""
            INSERT OR REPLACE INTO backtest_runs (
                run_id, symbol, signal_type, lookback,
                start_date, end_date, bar_count, trade_count,
                total_return, max_drawdown, sharpe_ratio, sortino_ratio,
                calmar_ratio, profit_factor, win_rate, expectancy,
                sufficient_sample
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            run_id, symbol, signal_type, lookback,
            str(start_date), str(end_date),
            metrics.bar_count, metrics.trade_count,
            metrics.total_return, metrics.max_drawdown,
            metrics.sharpe_ratio, metrics.sortino_ratio,
            metrics.calmar_ratio, metrics.profit_factor,
            metrics.win_rate, metrics.expectancy,
            metrics.sufficient_sample,
        ])
        con.close()
    except Exception as e:
        print(f"[storage] DuckDB backtest_runs insert failed: {e}")


def append_walk_forward_windows(
    run_id: str,
    symbol: str,
    signal_type: str,
    lookback: int,
    windows: list,
) -> None:
    """
    Insert all IS/OOS window rows for a single symbol into walk_forward_windows.

    One row per WalkForwardWindow. OOS columns are NULL when the OOS slice was
    too short to backtest (out_of_sample is None).

    Storage errors are caught and printed — they never crash the research loop.
    """
    try:
        con = get_db()
        for w in windows:
            oos = w.out_of_sample
            con.execute("""
                INSERT INTO walk_forward_windows (
                    run_id, symbol, signal_type, lookback, window_index,
                    is_start_bar, is_end_bar, oos_start_bar, oos_end_bar,
                    is_sharpe, is_total_return, is_trade_count, is_sufficient,
                    oos_sharpe, oos_total_return, oos_trade_count, oos_sufficient
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                run_id, symbol, signal_type, lookback, w.window_index,
                w.is_start_bar, w.is_end_bar, w.oos_start_bar, w.oos_end_bar,
                w.in_sample.sharpe_ratio, w.in_sample.total_return,
                w.in_sample.trade_count, w.in_sample.sufficient_sample,
                oos.sharpe_ratio if oos else None,
                oos.total_return if oos else None,
                oos.trade_count if oos else None,
                oos.sufficient_sample if oos else None,
            ])
        con.close()
    except Exception as e:
        print(f"[storage] DuckDB walk_forward_windows insert failed: {e}")


def query_oos_ranking(run_id: str, top_n: int = 10) -> list[dict]:
    """
    Return symbols ranked by average OOS Sharpe for a given universe run.

    Only windows with a non-NULL oos_sharpe are included in the average.
    Symbols with no valid OOS windows are excluded from the ranking.

    Returns a list of dicts with keys: symbol, avg_oos_sharpe, avg_oos_return,
    avg_is_sharpe, oos_windows.
    """
    try:
        con = get_db()
        rows = con.execute("""
            SELECT
                symbol,
                AVG(oos_sharpe)         AS avg_oos_sharpe,
                AVG(oos_total_return)   AS avg_oos_return,
                AVG(is_sharpe)          AS avg_is_sharpe,
                COUNT(oos_sharpe)       AS oos_windows
            FROM walk_forward_windows
            WHERE run_id = ?
              AND oos_sharpe IS NOT NULL
            GROUP BY symbol
            ORDER BY avg_oos_sharpe DESC
            LIMIT ?
        """, [run_id, top_n]).fetchall()
        con.close()
        return [
            {
                "symbol": r[0],
                "avg_oos_sharpe": r[1],
                "avg_oos_return": r[2],
                "avg_is_sharpe": r[3],
                "oos_windows": int(r[4]),
            }
            for r in rows
        ]
    except Exception as e:
        print(f"[storage] DuckDB query failed: {e}")
        return []


def append_trades_to_db(
    run_id: str,
    signal_type: str,
    lookback: int,
    trades: list,
) -> None:
    """Append trade records from a backtest run into the DuckDB trades table."""
    try:
        con = get_db()
        for trade in trades:
            row = asdict(trade) if hasattr(trade, "__dataclass_fields__") else dict(trade)
            row["run_id"] = run_id
            row["signal_type"] = signal_type
            row["lookback"] = lookback

            con.execute("""
                INSERT INTO trades (
                    run_id, symbol, signal_type, lookback,
                    signal_date, entry_date, entry_price,
                    exit_date, exit_price, trade_return,
                    ret_1d, ret_3d, ret_5d, mfe_5d, mae_5d,
                    atr_stop, news_category, news_count,
                    news_k_score, news_c_score, cost_bps
                ) VALUES (
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?
                )
            """, [
                row.get("run_id"), row.get("symbol"), signal_type, lookback,
                row.get("signal_date"), row.get("entry_date"), row.get("entry_price"),
                row.get("exit_date"), row.get("exit_price"), row.get("trade_return"),
                row.get("ret_1d"), row.get("ret_3d"), row.get("ret_5d"),
                row.get("mfe_5d"), row.get("mae_5d"),
                row.get("atr_stop"), row.get("news_category"), row.get("news_count"),
                row.get("news_k_score"), row.get("news_c_score"), row.get("cost_bps", 0.0),
            ])
        con.close()
    except Exception as e:
        # Storage errors never crash research — log and continue
        print(f"[storage] DuckDB append failed: {e}")


def append_scan_results(scan_id: str, results: list) -> None:
    """
    Persist a list of ScanResult objects to the scan_results DuckDB table.

    Each row captures every conviction layer so historical scans can be
    replayed and analysed: which layers were firing, at what scores, and
    how conviction evolved across daily runs.

    Storage errors are caught and printed — they never crash the scan.

    Args:
        scan_id: Unique run identifier (e.g. from make_run_id()).
        results: List of ScanResult dataclass instances.
    """
    try:
        con = get_db()
        for r in results:
            con.execute("""
                INSERT INTO scan_results (
                    scan_id, scan_date, symbol, signal_type, lookback,
                    entry_close, indicator_value, conviction_score, regime_bullish,
                    news_category, news_count, news_c_score,
                    rel_volume, atr_stop,
                    base_quality, absorption, volume_character, wyckoff_spring,
                    earnings_acceleration,
                    accumulation_ratio, volume_trend, climactic_volume,
                    sector, sector_cluster,
                    rs_score, edgar_acceleration,
                    breakout_volume_score, peg_score, stage
                ) VALUES (
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?,
                    ?, ?, ?, ?,
                    ?,
                    ?, ?, ?,
                    ?, ?,
                    ?, ?,
                    ?, ?, ?
                )
            """, [
                scan_id, r.scan_date, r.symbol, r.signal_type, r.lookback,
                r.entry_close, r.indicator_value, r.conviction_score, r.regime_bullish,
                r.news_category, r.news_count, r.news_c_score,
                r.rel_volume, r.atr_stop,
                r.base_quality, r.absorption, r.volume_character, r.wyckoff_spring,
                r.earnings_acceleration,
                r.accumulation_ratio, r.volume_trend, r.climactic_volume,
                getattr(r, "sector", ""), getattr(r, "sector_cluster", False),
                getattr(r, "rs_score", 0.0), getattr(r, "edgar_acceleration", None),
                getattr(r, "breakout_volume_score", 0.0),
                getattr(r, "peg_score", None), getattr(r, "stage", 0),
            ])
        con.close()
    except Exception as e:
        print(f"[storage] scan_results insert failed: {e}")
