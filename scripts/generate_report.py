"""
scripts/generate_report.py — STH Capital daily briefing PDF report.

Output: /mnt/c/Users/hadda/Desktop/Daily Report/YYYY-MM-DD_watchlist.pdf
        (override with --reports-dir for testing)

DuckDB: daily_reports table row updated after each run.

Usage:
    python scripts/generate_report.py
    python scripts/generate_report.py --date 2026-06-08
    python scripts/generate_report.py --reports-dir /tmp/reports
"""

from __future__ import annotations

import json
import os
import sys
from argparse import ArgumentParser
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reportlab.lib.colors import HexColor, white, black, lightgrey
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ── Brand colors ──────────────────────────────────────────────────────────────
C_NAVY   = HexColor("#1a1a2e")
C_ACCENT = HexColor("#e94560")
C_GREEN  = HexColor("#00b894")
C_RED    = HexColor("#d63031")
C_YELLOW = HexColor("#fdcb6e")
C_ORANGE = HexColor("#e17055")
C_LGRAY  = HexColor("#f0f2f5")
C_MGRAY  = HexColor("#cccccc")
C_DGRAY  = HexColor("#444444")
C_TGRAY  = HexColor("#888888")

# Row shading for candidate table
_ROW_GREEN  = HexColor("#d4f5e9")   # Stage 2 + options + 2+ days
_ROW_YELLOW = HexColor("#fff9e0")   # Stage 2, no options or single day

_TAPE_COLORS = {
    "BULL":     C_GREEN,
    "RECOVERY": C_YELLOW,
    "CAUTION":  C_ORANGE,
    "BEAR":     C_RED,
}

PAGE_W, PAGE_H = letter
MARGIN = 0.75 * inch
CONTENT_W = PAGE_W - 2 * MARGIN


# ── Paragraph styles ──────────────────────────────────────────────────────────

def _styles() -> dict:
    S = {}
    S["title"] = ParagraphStyle(
        "RLTitle", fontName="Helvetica-Bold", fontSize=20,
        textColor=C_NAVY, spaceAfter=2,
    )
    S["date_line"] = ParagraphStyle(
        "RLDateLine", fontName="Helvetica", fontSize=10,
        textColor=C_TGRAY, spaceAfter=14,
    )
    S["section"] = ParagraphStyle(
        "RLSection", fontName="Helvetica-Bold", fontSize=11,
        textColor=C_NAVY, spaceBefore=12, spaceAfter=5,
    )
    S["body"] = ParagraphStyle(
        "RLBody", fontName="Helvetica", fontSize=9,
        textColor=C_DGRAY, leading=14,
    )
    S["small"] = ParagraphStyle(
        "RLSmall", fontName="Helvetica", fontSize=7.5,
        textColor=C_TGRAY, leading=11,
    )
    S["cell"] = ParagraphStyle(
        "RLCell", fontName="Helvetica", fontSize=8, textColor=black, leading=11,
    )
    S["cell_bold"] = ParagraphStyle(
        "RLCellBold", fontName="Helvetica-Bold", fontSize=8, textColor=black, leading=11,
    )
    S["tape_label"] = ParagraphStyle(
        "RLTape", fontName="Helvetica-Bold", fontSize=22,
        textColor=C_NAVY, spaceAfter=0,
    )
    S["alert_header"] = ParagraphStyle(
        "RLAlertHeader", fontName="Helvetica-Bold", fontSize=10,
        textColor=white, leading=14,
    )
    S["alert_body"] = ParagraphStyle(
        "RLAlertBody", fontName="Helvetica", fontSize=8,
        textColor=C_DGRAY, leading=12,
    )
    S["alert_footnote"] = ParagraphStyle(
        "RLAlertFN", fontName="Helvetica-Oblique", fontSize=7.5,
        textColor=C_TGRAY, leading=11,
    )
    return S


# ── Helper formatters ─────────────────────────────────────────────────────────

def _pct(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v) * 100:+.1f}%"
    except (TypeError, ValueError):
        return "—"


def _fmt(v, d: int = 2) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.{d}f}"
    except (TypeError, ValueError):
        return "—"


def _tick(v) -> str:
    return "✓" if v else "–"


def _stage_lbl(n: int) -> str:
    return {1: "1·Base", 2: "2·Adv", 3: "3·Top", 4: "4·Dec"}.get(n, "?")


# ── Footer callback ────────────────────────────────────────────────────────────

def _make_footer(ts: str):
    def _cb(canvas, doc):
        canvas.saveState()
        y = 0.38 * inch
        canvas.setStrokeColor(C_MGRAY)
        canvas.setLineWidth(0.4)
        canvas.line(MARGIN, y + 13, PAGE_W - MARGIN, y + 13)
        canvas.setFont("Helvetica", 6.5)
        canvas.setFillColor(C_TGRAY)
        canvas.drawString(MARGIN, y, "STH Capital  |  QuantLab  |  Confidential")
        canvas.drawCentredString(PAGE_W / 2, y, f"Generated {ts}")
        canvas.drawRightString(PAGE_W - MARGIN, y, f"Page {doc.page}")
        canvas.restoreState()
    return _cb


# ── Page 1 — Market Summary ────────────────────────────────────────────────────

def _page1(
    S: dict,
    report_date: date,
    breadth_snap,
    n_cand: int,
    n_multi: int,
    abt_entry: dict | None,
) -> list:
    tape  = (breadth_snap.tape if breadth_snap else "N/A")
    mcl   = (f"{breadth_snap.mcclellan_oscillator:+.0f}"
             if breadth_snap and breadth_snap.mcclellan_oscillator is not None else "—")
    ratio = (f"{breadth_snap.ratio_10d:.2f}"
             if breadth_snap and breadth_snap.ratio_10d is not None else "—")
    ad_ln = (f"{breadth_snap.ad_line:,}"
             if breadth_snap and breadth_snap.ad_line is not None else "—")
    summ  = (f"{breadth_snap.mcclellan_summation:,.0f}"
             if breadth_snap and breadth_snap.mcclellan_summation is not None else "—")
    tape_c = _TAPE_COLORS.get(tape, C_DGRAY)

    e: list = []

    # ── Title block ───────────────────────────────────────────────────────────
    e.append(Paragraph("STH Capital — QuantLab Daily Briefing", S["title"]))
    e.append(Paragraph(report_date.strftime("%A, %B %d, %Y"), S["date_line"]))

    # ── Tape badge (colored box via single-cell Table) ────────────────────────
    tape_inner = Paragraph(
        f"<font color='white'><b>{tape}</b></font>",
        ParagraphStyle("TBadge", fontName="Helvetica-Bold", fontSize=18,
                       textColor=white, alignment=TA_CENTER),
    )
    tape_tbl = Table([[tape_inner]], colWidths=[1.6 * inch], rowHeights=[0.42 * inch])
    tape_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), tape_c),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
    ]))
    e.append(Table([[tape_tbl, Spacer(0.1 * inch, 0.1 * inch)]],
                   colWidths=[1.8 * inch, CONTENT_W - 1.8 * inch]))
    e.append(Spacer(1, 0.15 * inch))

    # ── Market breadth metrics ─────────────────────────────────────────────────
    e.append(Paragraph("Market Breadth", S["section"]))
    met_data = [
        ["McClellan Oscillator", mcl, "10-Day Breadth Ratio", ratio],
        ["AD Line",              ad_ln, "McClellan Summation",  summ],
    ]
    col_w = [1.8 * inch, 1.0 * inch, 1.8 * inch, 1.2 * inch]
    met_tbl = Table(met_data, colWidths=col_w)
    met_tbl.setStyle(TableStyle([
        ("FONTNAME",  (0, 0), (-1, -1), "Helvetica"),
        ("FONTNAME",  (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",  (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE",  (0, 0), (-1, -1), 8.5),
        ("ALIGN",     (1, 0), (1, -1), "RIGHT"),
        ("ALIGN",     (3, 0), (3, -1), "RIGHT"),
        ("BACKGROUND",(0, 0), (-1, 0), C_LGRAY),
        ("GRID",      (0, 0), (-1, -1), 0.4, C_MGRAY),
        ("PADDING",   (0, 0), (-1, -1), 6),
    ]))
    e.append(met_tbl)
    e.append(Spacer(1, 0.15 * inch))

    # ── Scan summary ───────────────────────────────────────────────────────────
    e.append(Paragraph("Scan Summary", S["section"]))
    sum_data = [
        ["Candidates Identified",              str(n_cand)],
        ["Multi-Day Candidates (2+ days)",     str(n_multi)],
    ]
    sum_tbl = Table(sum_data, colWidths=[3.0 * inch, 0.8 * inch])
    sum_tbl.setStyle(TableStyle([
        ("FONTNAME",  (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",  (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE",  (0, 0), (-1, -1), 8.5),
        ("ALIGN",     (1, 0), (1, -1), "CENTER"),
        ("BACKGROUND",(0, 0), (-1, 0), C_LGRAY),
        ("GRID",      (0, 0), (-1, -1), 0.4, C_MGRAY),
        ("PADDING",   (0, 0), (-1, -1), 6),
    ]))
    e.append(sum_tbl)

    # ── Open position P&L (if available) ──────────────────────────────────────
    if abt_entry:
        e.append(Spacer(1, 0.15 * inch))
        e.append(Paragraph("Open Position Monitor", S["section"]))
        sym    = abt_entry.get("symbol", "—")
        ep     = abt_entry.get("entry_price") or 0.0
        cp     = abt_entry.get("current_price")
        unrl   = abt_entry.get("unrealized_ret")
        days_w = abt_entry.get("days_on_watch", 0)
        status = (abt_entry.get("status") or "watching").upper()
        cp_str   = f"${cp:.2f}"    if cp   is not None else "—"
        unrl_str = f"{unrl*100:+.2f}%" if unrl is not None else "—"
        pnl_c    = C_GREEN if (unrl or 0) >= 0 else C_RED

        pos_data = [
            ["Symbol", "Entry",      "Current",  "Unrealized P&L", "Days", "Status"],
            [sym,      f"${ep:.2f}", cp_str,     unrl_str,          str(days_w), status],
        ]
        pos_tbl = Table(
            pos_data,
            colWidths=[0.85*inch, 0.85*inch, 0.85*inch, 1.15*inch, 0.7*inch, 0.9*inch],
        )
        pos_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), C_NAVY),
            ("TEXTCOLOR",  (0, 0), (-1, 0), white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 8),
            ("FONTNAME",   (0, 1), (0,  1), "Helvetica-Bold"),
            ("TEXTCOLOR",  (3, 1), (3,  1), pnl_c),
            ("FONTNAME",   (3, 1), (3,  1), "Helvetica-Bold"),
            ("ALIGN",      (1, 0), (-1, -1), "CENTER"),
            ("GRID",       (0, 0), (-1, -1), 0.4, C_MGRAY),
            ("PADDING",    (0, 0), (-1, -1), 6),
        ]))
        e.append(pos_tbl)

    return e


# ── Page 2+ — Candidate Table ─────────────────────────────────────────────────

def _candidate_table(
    S: dict,
    candidates: list[dict],
    backtest_map: dict,
) -> list:
    e: list = []
    e.append(PageBreak())
    e.append(Paragraph("Pre-Breakout Candidate Table", S["section"]))
    e.append(Paragraph(
        "Sorted by consecutive days ↓ then conviction ↓.  "
        "Green = Stage 2 + options + 2+ days.  Yellow = Stage 2.",
        S["small"],
    ))
    e.append(Spacer(1, 0.08 * inch))

    headers = ["#", "Symbol", "Stage", "Days", "Conv", "Opts", "VDU",
               "EPS %", "PEG", "Notes"]
    cw = [0.28, 0.72, 0.60, 0.40, 0.50, 0.38, 0.38, 0.62, 0.45, 1.95]
    col_widths = [w * inch for w in cw]

    rows = [headers]
    row_cmds: list = []

    for idx, en in enumerate(candidates, 1):
        stage = en.get("stage", 0)
        days  = en.get("consecutive_days", 1)
        opts  = en.get("options_signal", False)
        bkt   = backtest_map.get(en["symbol"], {})
        wr    = (f"WR:{bkt['win_rate']*100:.0f}%"
                 if bkt.get("win_rate") is not None else "")

        rows.append([
            str(idx),
            en["symbol"],
            _stage_lbl(stage),
            f"{days}d",
            _fmt(en.get("conviction_score")),
            _tick(opts),
            _tick(en.get("volume_dry_up")),
            _pct(en.get("earnings_score")),
            _fmt(en.get("peg_score")),
            wr,
        ])

        # Row shading — applied after ROWBACKGROUNDS, so these take priority
        ri = idx  # table row index (header = 0)
        if stage == 2 and opts and days >= 2:
            row_cmds.append(("BACKGROUND", (0, ri), (-1, ri), _ROW_GREEN))
        elif stage == 2:
            row_cmds.append(("BACKGROUND", (0, ri), (-1, ri), _ROW_YELLOW))

    style_cmds = [
        ("BACKGROUND",    (0, 0), (-1, 0),  C_NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 7.5),
        ("FONTNAME",      (1, 1), (1, -1),  "Helvetica-Bold"),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("ALIGN",         (1, 0), (1, -1),  "LEFT"),
        ("ALIGN",         (9, 0), (9, -1),  "LEFT"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [white, C_LGRAY]),
        ("GRID",          (0, 0), (-1, -1), 0.3, C_MGRAY),
        ("LINEBELOW",     (0, 0), (-1, 0),  1.0, C_NAVY),
        ("PADDING",       (0, 0), (-1, -1), 4),
        ("TOPPADDING",    (0, 0), (-1, 0),  6),
        ("BOTTOMPADDING", (0, 0), (-1, 0),  6),
    ] + row_cmds

    tbl = Table(rows, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle(style_cmds))
    e.append(tbl)
    return e


# ── Pre-Breakout Alert Blocks ─────────────────────────────────────────────────

def _alert_section(
    S: dict,
    candidates: list[dict],
    backtest_map: dict,
) -> list:
    top5 = [c for c in candidates if c.get("consecutive_days", 1) >= 2][:5]
    if not top5:
        return []

    e: list = []
    e.append(Spacer(1, 0.2 * inch))
    e.append(HRFlowable(width=CONTENT_W, thickness=0.5, color=C_MGRAY))
    e.append(Spacer(1, 0.08 * inch))
    e.append(Paragraph(
        "Pre-Breakout Candidates — Institutional Positioning Detected",
        S["section"],
    ))

    for en in top5:
        sym   = en["symbol"]
        days  = en.get("consecutive_days", 1)
        stage = en.get("stage", 0)
        bkt   = backtest_map.get(sym, {})
        wr_str = (
            f"Win rate: {bkt['win_rate']*100:.0f}%  |  Avg return: {bkt['avg_return']*100:+.1f}%"
            if bkt.get("win_rate") is not None
            else "Backtest: pending"
        )
        signal_line = (
            f"Stage: {_stage_lbl(stage)}   |   Conviction: {_fmt(en.get('conviction_score'))}"
            f"   |   EPS YoY: {_pct(en.get('earnings_score'))}"
            f"   |   PEG: {_fmt(en.get('peg_score'))}"
        )
        detail_line = (
            f"{'✓ Options signal' if en.get('options_signal') else '– No options signal'}"
            f"   |   {'✓ Volume dry-up' if en.get('volume_dry_up') else '– Vol normal'}"
            f"   |   Brkout Vol: {_fmt(en.get('breakout_volume_score'))}"
            f"   |   First seen: {en.get('first_seen', '—')}"
        )
        header_c = C_GREEN if en.get("options_signal") else C_NAVY
        card_bg   = _ROW_GREEN if en.get("options_signal") else _ROW_YELLOW

        card_data = [
            [Paragraph(f"★  {sym}  —  Day {days}", S["alert_header"])],
            [Paragraph(signal_line, S["alert_body"])],
            [Paragraph(detail_line, S["alert_body"])],
            [Paragraph(wr_str,      S["alert_footnote"])],
        ]
        card = Table(card_data, colWidths=[CONTENT_W])
        card.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  header_c),
            ("BACKGROUND",    (0, 1), (-1, -1), card_bg),
            ("GRID",          (0, 0), (-1, -1), 0.3, C_MGRAY),
            ("PADDING",       (0, 0), (-1, -1), 7),
            ("TOPPADDING",    (0, 0), (-1, 0),  8),
            ("BOTTOMPADDING", (0, 0), (-1, 0),  8),
        ]))
        e.append(card)
        e.append(Spacer(1, 0.1 * inch))

    return e


# ── Data helpers ───────────────────────────────────────────────────────────────

def _load_abt_entry(db_path: str | None = None) -> dict | None:
    """Return the top active watchlist entry that has a current price set."""
    try:
        import duckdb
        from quantlab.storage import DB_PATH
        con = duckdb.connect(db_path or str(DB_PATH))
        row = con.execute(
            """
            SELECT symbol, entry_price, current_price, unrealized_ret,
                   days_on_watch, status
            FROM watchlist
            WHERE status = 'watching' AND current_price IS NOT NULL
            ORDER BY conviction_score DESC LIMIT 1
            """
        ).fetchone()
        con.close()
        if row:
            return dict(zip(
                ["symbol", "entry_price", "current_price", "unrealized_ret",
                 "days_on_watch", "status"], row,
            ))
    except Exception:
        pass
    return None


def _load_backtest_map(symbols: list[str], db_path: str | None = None) -> dict:
    if not symbols:
        return {}
    try:
        import duckdb
        from quantlab.storage import DB_PATH
        con = duckdb.connect(db_path or str(DB_PATH))
        ph  = ",".join("?" * len(symbols))
        rows = con.execute(
            f"SELECT symbol, AVG(win_rate), AVG(total_return) "
            f"FROM backtest_runs WHERE symbol IN ({ph}) GROUP BY symbol",
            symbols,
        ).fetchall()
        con.close()
        return {r[0]: {"win_rate": r[1], "avg_return": r[2]} for r in rows}
    except Exception:
        return {}


def _save_report_row(
    report_date: date,
    tape: str,
    mcclellan: float | None,
    n_cand: int,
    n_multi: int,
    top5: list[str],
    db_path: str | None = None,
) -> None:
    try:
        import duckdb
        from quantlab.storage import DB_PATH, _ensure_schema
        con = duckdb.connect(db_path or str(DB_PATH))
        _ensure_schema(con)
        con.execute(
            """
            INSERT OR REPLACE INTO daily_reports
                (date, tape, mcclellan, candidates, multi_day, top_symbols, generated_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            [report_date.isoformat(), tape, mcclellan, n_cand, n_multi,
             json.dumps(top5)],
        )
        con.close()
    except Exception as exc:
        print(f"[generate_report] daily_reports insert failed: {exc}")


# ── Public API ─────────────────────────────────────────────────────────────────

_DEFAULT_DIR = Path("/mnt/c/Users/hadda/Desktop/Daily Report")


def generate(
    report_date: date | None = None,
    reports_dir: Path | None = None,
    db_path: str | None = None,
) -> Path:
    """
    Build and write the PDF report; persist the DuckDB daily_reports row.

    Args:
        report_date: Date for the report (default: today).
        reports_dir: Output directory (default: Desktop path for live use).
        db_path:     Override DuckDB path (used in tests).

    Returns:
        Path to the written PDF file.
    """
    from quantlab.watchlist import InstitutionalWatchlist
    from quantlab.signals.breadth import get_latest_snapshot

    report_date = report_date or date.today()
    out_dir     = Path(reports_dir) if reports_dir else _DEFAULT_DIR
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / f"{report_date.isoformat()}_watchlist.pdf"

    # ── Fetch data ─────────────────────────────────────────────────────────────
    iwl          = InstitutionalWatchlist(db_path=db_path)
    candidates   = iwl.get_candidates()
    breadth      = get_latest_snapshot()
    symbols      = [c["symbol"] for c in candidates]
    backtest_map = _load_backtest_map(symbols, db_path)
    abt_entry    = _load_abt_entry(db_path)
    n_multi      = sum(1 for c in candidates if c.get("consecutive_days", 1) >= 2)

    # ── Build PDF ──────────────────────────────────────────────────────────────
    ts     = datetime.now().strftime("%Y-%m-%d %H:%M")
    footer = _make_footer(ts)
    S      = _styles()

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=letter,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN,  bottomMargin=0.75 * inch,
        title=f"STH Capital — QuantLab Daily Briefing  {report_date}",
        author="STH Capital",
        subject="Institutional Pre-Breakout Watchlist",
    )

    story: list = []
    story += _page1(S, report_date, breadth, len(candidates), n_multi, abt_entry)
    if candidates:
        story += _candidate_table(S, candidates, backtest_map)
        story += _alert_section(S, candidates, backtest_map)

    doc.build(story, onFirstPage=footer, onLaterPages=footer)

    print(f"  PDF report → {out_path}  ({len(candidates)} candidates)")

    # ── Persist DuckDB row ─────────────────────────────────────────────────────
    tape      = breadth.tape if breadth else "N/A"
    mcclellan = breadth.mcclellan_oscillator if breadth else None
    _save_report_row(report_date, tape, mcclellan, len(candidates), n_multi,
                     symbols[:5], db_path)

    return out_path


def main() -> None:
    parser = ArgumentParser(
        description="Generate STH Capital daily briefing PDF report."
    )
    parser.add_argument("--date", default=None,
                        help="Report date YYYY-MM-DD (default: today)")
    parser.add_argument("--reports-dir", default=None,
                        help="Override output directory (default: Desktop)")
    args = parser.parse_args()

    report_date = date.fromisoformat(args.date) if args.date else date.today()
    reports_dir = Path(args.reports_dir) if args.reports_dir else None

    generate(report_date=report_date, reports_dir=reports_dir)


if __name__ == "__main__":
    main()
