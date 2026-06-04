"""
QuantLab — institutional-grade quant research and trading system.

Package layout mirrors the 7-layer architecture:

    quantlab.providers   — Layer 1: market data (IBKR, HTTP, mock)
    quantlab.news        — Layer 2: headline fetch, clean, classify, score
    quantlab.options     — Layer 3: option chain, quotes, Greeks, IV features
    quantlab.signals     — Layer 4: signal generation (SMA, breakout, regime)
    quantlab.research    — Layer 4: backtesting, forward returns, MFE/MAE
    quantlab.storage     — Layer 5: DuckDB, Parquet, CSV I/O
    quantlab.risk        — Layer 6: metrics (Sharpe, Sortino, Calmar), stops
    quantlab.execution   — Layer 7: scanner, ZMQ bus, order management
    quantlab.utils       — shared helpers (dates, logging, config)
"""

__version__ = "0.2.0"
