"""
scripts/generate_report.py — Daily institutional pre-breakout watchlist report.

Outputs:
  A) data/reports/YYYY-MM-DD_watchlist.html
  B) DuckDB daily_reports table row

Usage:
    python scripts/generate_report.py
    python scripts/generate_report.py --date 2026-06-08
    python scripts/generate_report.py --reports-dir /tmp/reports
"""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── Row-color logic ────────────────────────────────────────────────────────────

def _row_class(entry: dict) -> str:
    """Return a CSS class string for a watchlist row."""
    stage = entry.get("stage", 0)
    days  = entry.get("consecutive_days", 1)
    opts  = entry.get("options_signal", False)
    if stage == 2 and opts and days >= 2:
        return "row-green"
    if stage == 2:
        return "row-yellow"
    if stage == 1:
        return "row-grey"
    return ""


def _pct(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v) * 100:+.1f}%"
    except (TypeError, ValueError):
        return "—"


def _fmt(v, decimals: int = 2) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def _tick(v: bool | None) -> str:
    return "✓" if v else "–"


# ── HTML generation ────────────────────────────────────────────────────────────

_CSS = """
body { font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
       background:#0f0f0f; color:#e0e0e0; margin:0; padding:20px; }
h1   { color:#f0f0f0; font-size:1.3rem; margin-bottom:4px; }
.subtitle { color:#888; font-size:.85rem; margin-bottom:20px; }
.header-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:12px;
               margin-bottom:24px; }
.kpi  { background:#1a1a2e; border-radius:8px; padding:14px; }
.kpi .label { font-size:.75rem; color:#888; text-transform:uppercase;
              letter-spacing:.05em; }
.kpi .value { font-size:1.5rem; font-weight:700; margin-top:4px; }
.tape-BULL     { color:#00d68f; }
.tape-RECOVERY { color:#ffaa00; }
.tape-CAUTION  { color:#ffcc00; }
.tape-BEAR     { color:#ff4444; }
table { width:100%; border-collapse:collapse; font-size:.82rem;
        background:#111; border-radius:8px; overflow:hidden; }
th { background:#1e1e3e; color:#aaa; font-weight:600; padding:10px 8px;
     text-align:left; white-space:nowrap; }
td { padding:8px; border-bottom:1px solid #222; vertical-align:middle; }
tr:last-child td { border-bottom:none; }
.row-green  td { background:#0d2b1a; }
.row-yellow td { background:#2b2200; }
.row-grey   td { background:#1a1a1a; }
.sym { font-weight:700; font-size:.9rem; }
.stage2 { color:#00d68f; }
.stage1 { color:#aaa; }
.days-badge { background:#1e1e3e; border-radius:4px; padding:2px 7px;
              font-weight:700; font-size:.8rem; }
.priority-section { margin-top:28px; }
.priority-section h2 { color:#ffaa00; font-size:1rem; margin-bottom:14px; }
.card { background:#1a1a2e; border-radius:8px; padding:16px; margin-bottom:12px;
        border-left:4px solid #3a3aff; }
.card.opts-fired { border-left-color:#00d68f; }
.card h3 { margin:0 0 8px; font-size:1rem; color:#e0e0e0; }
.card .detail { display:flex; flex-wrap:wrap; gap:10px; font-size:.8rem; color:#aaa; }
.card .detail span { background:#111; border-radius:4px; padding:3px 8px; }
"""


def build_html(
    report_date: date,
    candidates: list[dict],
    breadth_snap,
    backtest_map: dict,
) -> str:
    tape   = breadth_snap.tape if breadth_snap else "N/A"
    mcl    = f"{breadth_snap.mcclellan_oscillator:+.0f}" if breadth_snap and breadth_snap.mcclellan_oscillator is not None else "—"
    ratio  = f"{breadth_snap.ratio_10d:.2f}" if breadth_snap and breadth_snap.ratio_10d is not None else "—"
    n_cand = len(candidates)
    n_multi = sum(1 for c in candidates if c.get("consecutive_days", 1) >= 2)
    tape_cls = f"tape-{tape}"

    # Header KPIs
    header = f"""
    <div class="header-grid">
      <div class="kpi">
        <div class="label">Market Tape</div>
        <div class="value {tape_cls}">{tape}</div>
      </div>
      <div class="kpi">
        <div class="label">McClellan</div>
        <div class="value">{mcl}</div>
      </div>
      <div class="kpi">
        <div class="label">10d Ratio</div>
        <div class="value">{ratio}</div>
      </div>
      <div class="kpi">
        <div class="label">Candidates / Multi-Day</div>
        <div class="value">{n_cand} <span style="color:#888;font-size:1rem">/ {n_multi}</span></div>
      </div>
    </div>
    """

    # Main table
    rows_html = ""
    for e in candidates:
        stage     = e.get("stage", 0)
        sym_cls   = "stage2" if stage == 2 else ("stage1" if stage == 1 else "")
        stage_lbl = {1: "Base", 2: "Adv.", 3: "Top", 4: "Dec."}.get(stage, "?")
        bkt = backtest_map.get(e["symbol"], {})
        wr  = f"{bkt.get('win_rate', 0)*100:.0f}%" if bkt.get("win_rate") is not None else "—"
        rows_html += f"""
        <tr class="{_row_class(e)}">
          <td class="sym {sym_cls}">{e['symbol']}</td>
          <td>{stage_lbl}</td>
          <td><span class="days-badge">{e.get('consecutive_days',1)}d</span></td>
          <td>{_fmt(e.get('conviction_score'))}</td>
          <td>{_tick(e.get('options_signal'))}</td>
          <td>{_tick(e.get('volume_dry_up'))}</td>
          <td>{_pct(e.get('earnings_score'))}</td>
          <td>{_fmt(e.get('peg_score'))}</td>
          <td>{_fmt(e.get('breakout_volume_score'))}</td>
          <td>{e.get('tape','')}</td>
          <td>{wr}</td>
        </tr>
        """

    table = f"""
    <table>
      <thead>
        <tr>
          <th>Symbol</th><th>Stage</th><th>Days</th><th>Conviction</th>
          <th>Options</th><th>Vol Dry-Up</th><th>EPS YoY</th><th>PEG</th>
          <th>Brkout Vol</th><th>Tape</th><th>WinRate</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    """

    # Pre-breakout alert cards (top 5 multi-day)
    top5 = [c for c in candidates if c.get("consecutive_days", 1) >= 2][:5]
    cards_html = ""
    for e in top5:
        bkt  = backtest_map.get(e["symbol"], {})
        wr   = f"Win rate: {bkt.get('win_rate',0)*100:.0f}%  avg ret: {bkt.get('avg_return',0)*100:+.1f}%" if bkt else "Backtest: pending"
        card_cls = "card opts-fired" if e.get("options_signal") else "card"
        cards_html += f"""
        <div class="{card_cls}">
          <h3>⭐ {e['symbol']} — Day {e.get('consecutive_days',1)}</h3>
          <div class="detail">
            <span>Conviction: {_fmt(e.get('conviction_score'))}</span>
            <span>Stage: {e.get('stage',0)}</span>
            <span>EPS YoY: {_pct(e.get('earnings_score'))}</span>
            <span>PEG: {_fmt(e.get('peg_score'))}</span>
            <span>Brkout Vol: {_fmt(e.get('breakout_volume_score'))}</span>
            <span>Options: {_tick(e.get('options_signal'))}</span>
            <span>Vol Dry-Up: {_tick(e.get('volume_dry_up'))}</span>
            <span>First Seen: {e.get('first_seen','')}</span>
            <span>{wr}</span>
          </div>
        </div>
        """

    priority_section = f"""
    <div class="priority-section">
      <h2>Pre-Breakout Alerts — Top {len(top5)} Multi-Day Candidates</h2>
      {cards_html if cards_html else '<p style="color:#888">No multi-day candidates yet.</p>'}
    </div>
    """ if top5 else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>QuantLab Institutional Watchlist — {report_date}</title>
  <style>{_CSS}</style>
</head>
<body>
  <h1>QuantLab Institutional Pre-Breakout Watchlist</h1>
  <div class="subtitle">{report_date} · generated {datetime.now().strftime('%H:%M:%S ET')}</div>
  {header}
  {table}
  {priority_section}
</body>
</html>
"""


# ── DuckDB persistence ─────────────────────────────────────────────────────────

def _save_report_row(
    report_date: date,
    tape: str,
    mcclellan: float | None,
    n_candidates: int,
    n_multi: int,
    top_symbols: list[str],
    db_path: str | None = None,
) -> None:
    try:
        import duckdb
        from quantlab.storage import DB_PATH, _ensure_schema
        path = db_path or str(DB_PATH)
        con = duckdb.connect(path)
        _ensure_schema(con)
        con.execute(
            """
            INSERT OR REPLACE INTO daily_reports
                (date, tape, mcclellan, candidates, multi_day, top_symbols, generated_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            [
                report_date.isoformat(),
                tape,
                mcclellan,
                n_candidates,
                n_multi,
                json.dumps(top_symbols),
            ],
        )
        con.close()
    except Exception as exc:
        print(f"[generate_report] daily_reports insert failed: {exc}")


# ── Backtest lookup ────────────────────────────────────────────────────────────

def _load_backtest_map(symbols: list[str], db_path: str | None = None) -> dict:
    """Return {symbol: {win_rate, avg_return}} from DuckDB backtest_runs."""
    if not symbols:
        return {}
    try:
        import duckdb
        from quantlab.storage import DB_PATH
        path = db_path or str(DB_PATH)
        con = duckdb.connect(path)
        placeholders = ",".join("?" * len(symbols))
        rows = con.execute(
            f"""
            SELECT symbol,
                   AVG(win_rate)     AS win_rate,
                   AVG(total_return) AS avg_return
            FROM backtest_runs
            WHERE symbol IN ({placeholders})
            GROUP BY symbol
            """,
            symbols,
        ).fetchall()
        con.close()
        return {r[0]: {"win_rate": r[1], "avg_return": r[2]} for r in rows}
    except Exception:
        return {}


# ── Main ───────────────────────────────────────────────────────────────────────

def generate(
    report_date: date | None = None,
    reports_dir: Path | None = None,
    db_path: str | None = None,
) -> Path:
    """
    Generate the HTML report and persist the DuckDB summary row.

    Returns the Path to the written HTML file.
    """
    from quantlab.watchlist import InstitutionalWatchlist
    from quantlab.signals.breadth import get_latest_snapshot

    report_date  = report_date or date.today()
    reports_dir  = reports_dir or (Path(__file__).parent.parent / "data" / "reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    iwl        = InstitutionalWatchlist(db_path=db_path)
    candidates = iwl.get_candidates()
    breadth    = get_latest_snapshot()

    symbols     = [c["symbol"] for c in candidates]
    backtest_map = _load_backtest_map(symbols, db_path)

    html = build_html(report_date, candidates, breadth, backtest_map)

    out_path = reports_dir / f"{report_date.isoformat()}_watchlist.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"  report → {out_path}  ({len(candidates)} candidates)")

    # Persist summary row
    n_multi    = sum(1 for c in candidates if c.get("consecutive_days", 1) >= 2)
    top5       = [c["symbol"] for c in candidates[:5]]
    tape       = breadth.tape if breadth else "N/A"
    mcclellan  = breadth.mcclellan_oscillator if breadth else None
    _save_report_row(report_date, tape, mcclellan, len(candidates), n_multi, top5, db_path)

    return out_path


def main() -> None:
    parser = ArgumentParser(description="Generate daily institutional watchlist report.")
    parser.add_argument("--date", default=None, help="Report date YYYY-MM-DD (default: today)")
    parser.add_argument("--reports-dir", default=None,
                        help="Override output directory for HTML files")
    args = parser.parse_args()

    report_date = date.fromisoformat(args.date) if args.date else date.today()
    reports_dir = Path(args.reports_dir) if args.reports_dir else None

    generate(report_date=report_date, reports_dir=reports_dir)


if __name__ == "__main__":
    main()
