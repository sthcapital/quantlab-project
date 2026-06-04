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

    con.execute("""
        CREATE TABLE IF NOT EXISTS scan_results (
            scan_date       DATE,
            symbol          VARCHAR,
            signal_type     VARCHAR,
            entry_close     DOUBLE,
            indicator_value DOUBLE,
            news_category   VARCHAR,
            news_count      INTEGER,
            conviction_score DOUBLE,
            regime_bullish  BOOLEAN,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)


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
