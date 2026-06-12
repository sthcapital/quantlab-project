#!/usr/bin/env python
"""One-off diagnostic: re-score the current universe with the fixed
ExplosionScore semantics and report the score distribution plus the rank
correlation between explosion_score and component coverage.

Calls run_universe_scan() directly (flatfile bars, no news, min_conviction=0)
so no watchlist tables are mutated — this is read-only analysis.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta

import pandas as pd


def main() -> None:
    from quantlab.execution import run_universe_scan
    from quantlab.providers import create_market_data_provider
    from quantlab.universe import load_universe_cache

    # Most recent cached tradeable universe (same source as evening_scan.sh)
    symbols = None
    for d in range(7):
        cached = load_universe_cache(date.today() - timedelta(days=d))
        if cached:
            symbols, _stats = cached
            print(f"Universe cache: {len(symbols)} symbols "
                  f"({(date.today() - timedelta(days=d)).isoformat()})")
            break
    if not symbols:
        print("No universe cache found — aborting")
        sys.exit(1)

    end = date.today()
    start = end - timedelta(days=420)  # ≥252 trading bars for RS percentile

    results = run_universe_scan(
        provider=create_market_data_provider("flatfile"),
        symbols=symbols,
        start_date=start,
        end_date=end,
        signal_type="breakout",
        lookback=5,
        min_conviction=0.0,   # keep every processed symbol for the distribution
        ibkr_connection=None,
    )

    rows = [
        {
            "symbol": r.symbol,
            "explosion_score": r.explosion_score,
            "components": r.explosion_components,
            "conviction": r.conviction_score,
            "stage": r.stage,
        }
        for r in results
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        print("\nNo symbols processed — check flat-file data availability")
        sys.exit(1)
    df.to_csv("output/rescore_universe_analysis.csv", index=False)

    n = len(df)
    n_none = int(df.explosion_score.isna().sum())
    n_zero = int((df.explosion_score == 0.0).sum())
    n_pop = int((df.explosion_score > 0.0).sum())
    print(f"\nProcessed symbols           : {n}")
    print(f"explosion_score None  ('—') : {n_none}  ({n_none/n:.1%})")
    print(f"explosion_score 0.00 (real) : {n_zero}  ({n_zero/n:.1%})")
    print(f"explosion_score populated   : {n_pop}  ({n_pop/n:.1%})")

    print("\nComponent-count distribution:")
    print(df.components.value_counts().sort_index().to_string())

    scored = df[df.explosion_score.notna()]
    if len(scored) > 2 and scored.components.nunique() > 1:
        rho = scored.explosion_score.corr(scored.components, method="spearman")
        print(f"\nSpearman rank corr (explosion_score vs component_count, scored only): {rho:+.3f}")
        if abs(rho) >= 0.4:
            print("WARNING: strong coverage-score correlation — gate may be too loose")
    else:
        print("\nSpearman: not computable (insufficient variation in component_count)")

    print("\nTop 15 by explosion_score:")
    top = scored.sort_values("explosion_score", ascending=False).head(15)
    print(top.to_string(index=False))
    lowcov_top = (top.components < 6).mean()
    print(f"\nShare of top 15 with <6 components: {lowcov_top:.0%}")


if __name__ == "__main__":
    main()
