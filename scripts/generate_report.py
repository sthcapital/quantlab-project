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
C_BLUE   = HexColor("#74b9ff")       # RECOVERY state
C_NGRAY  = HexColor("#636e72")       # NEUTRAL state

# Row shading for candidate table
_ROW_GREEN  = HexColor("#d4f5e9")   # Stage 2 + options + 2+ days
_ROW_YELLOW = HexColor("#fff9e0")   # Stage 2, no options or single day

_TAPE_COLORS = {
    "BULL":       C_GREEN,
    "CORRECTION": C_YELLOW,
    "RECOVERY":   C_BLUE,
    "NEUTRAL":    C_NGRAY,
    "BEAR":       C_RED,
}

PAGE_W, PAGE_H = letter
MARGIN = 0.75 * inch
CONTENT_W = PAGE_W - 2 * MARGIN


# ── Paragraph styles ──────────────────────────────────────────────────────────

def _styles() -> dict:
    S = {}
    S["title"] = ParagraphStyle(
        "RLTitle", fontName="Helvetica-Bold", fontSize=20,
        spaceAfter=6, textColor=C_NAVY, alignment=TA_CENTER,
    )
    S["date_line"] = ParagraphStyle(
        "RLDateLine", fontName="Helvetica", fontSize=10,
        spaceBefore=4, spaceAfter=16, textColor=C_TGRAY, alignment=TA_CENTER,
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


# ── Pure-computation helpers (testable without ReportLab) ────────────────────

def momentum_interpretation(up: int, dn: int, up_34: int = 0, dn_34: int = 0) -> str:
    """
    Generate a 1-2 sentence market interpretation from monthly and 34-day momentum counts.

    Args:
        up:    Stocks up ≥ 25% in the last 21 trading days.
        dn:    Stocks down ≥ 25% in the last 21 trading days.
        up_34: Stocks up ≥ 13% in the last 34 trading days.
        dn_34: Stocks down ≥ 13% in the last 34 trading days.
    """
    ratio = up / max(dn, 1)
    if ratio > 2.0:
        main = (
            f"Strong accumulation — {up} stocks made 25%+ monthly gains "
            f"vs {dn} with similar losses. "
            "Institutional money is actively deploying."
        )
    elif ratio >= 1.2:
        main = (
            "Moderate positive momentum — more stocks advancing "
            "strongly than declining. "
            "Market internals support selective long exposure."
        )
    elif ratio >= 0.8:
        main = (
            "Mixed momentum — advances and declines roughly balanced. "
            "Wait for clearer directional signal before adding exposure."
        )
    else:
        main = (
            f"Distribution in progress — {dn} stocks declining sharply "
            f"vs {up} advancing. Reduce risk, protect capital."
        )

    if up_34 > dn_34:
        confirm = "34-day momentum confirms positive trend."
    elif dn_34 > up_34:
        confirm = "34-day momentum confirms negative trend."
    else:
        confirm = "34-day momentum is neutral."

    return f"{main} {confirm}"


def _compute_r_multiple(
    entry_price: float | None,
    atr_stop: float | None,
    current_price: float | None,
) -> float | None:
    """Return R-multiple (current - entry) / (entry - stop), or None if inputs invalid."""
    if not (entry_price and atr_stop and current_price):
        return None
    risk = entry_price - atr_stop
    if risk <= 0:
        return None
    return (current_price - entry_price) / risk


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


def _opts_cell(opts_signal, fresh: dict | None) -> str:
    """
    Opts column with flag freshness: "F" for a fresh flag (streak 1),
    "FdN" for the Nth consecutive flagged session — first-day flow
    (positioning starting while price still bases) reads differently from
    campaign confirmation / crowding risk.

    The marker is ASCII on purpose: the report's Helvetica Type1 font has no
    U+2691 (⚑) glyph, which rendered as "n" in the generated PDF ("nd2"
    where "⚑d2" was meant).

    Floor-blocked and put-dominated names are not flagged, so they keep
    their existing treatment (no marker); rows without snapshot freshness
    fall back to the legacy ✓/– tick.
    """
    if fresh and fresh.get("unusual_flag"):
        streak = fresh.get("flag_streak") or 1
        return "F" if streak <= 1 else f"Fd{streak}"
    if fresh is not None:
        return "–"   # gated today, not flagged (incl. floor/PCR-blocked)
    return _tick(opts_signal)


def _stage_lbl(n: int) -> str:
    return {1: "1·Base", 2: "2", 3: "3·Top", 4: "4·Dec"}.get(n, "?")


def _bv_ratio(v) -> str:
    """Format the raw Weinstein breakout-volume ratio (e.g. 2.1x).

    None/NULL means the ratio was not measurable (no bar history) and must
    render as "—" — never as a number.
    """
    if v is None:
        return "—"
    try:
        return f"{float(v):.1f}x"
    except (TypeError, ValueError):
        return "—"


# Display overflow cap for YoY percentages: beyond ±999% the precise figure
# carries no signal at table resolution.  Data-quality suppression happens
# UPSTREAM (quarantine → NULL → "—"); display only handles overflow, so real
# hypergrowth (SNDK Rev +251%) renders as the number it is.
_YOY_DISPLAY_CAP = 9.99


def _rev_pct(v) -> str:
    """Format raw revenue YoY: '—' for missing/quarantined, '>999%' overflow."""
    if v is None:
        return "—"
    try:
        f = float(v)
        if f > _YOY_DISPLAY_CAP:
            return ">999%"
        return f"{f * 100:+.1f}%"
    except (TypeError, ValueError):
        return "—"


def _eps_cell(fresh: dict | None) -> str:
    """EPS YoY display: raw % with '>999%' overflow, a distinct '−→+' marker
    for negative→positive transitions (never a fake +100.0%), '—' for
    missing/quarantined."""
    if not fresh:
        return "—"
    if fresh.get("turned_positive"):
        return "−→+"
    v = fresh.get("eps_yoy")
    if v is None:
        return "—"
    if v > _YOY_DISPLAY_CAP:
        return ">999%"
    return f"{v * 100:+.1f}%"


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
    open_positions: list[dict],
    vix_close: float | None = None,
    warning: str | None = None,
    options_rate: str | None = None,
    regime_gate=None,
    growth_funnel: str | None = None,
    excluded_defensives: str | None = None,
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
    p10   = (f"{breadth_snap.pct_above_10sma:.1f}%"
             if breadth_snap else "—")
    p20   = (f"{breadth_snap.pct_above_20sma:.1f}%"
             if breadth_snap else "—")
    p50   = (f"{breadth_snap.pct_above_50sma:.1f}%"
             if breadth_snap else "—")
    p200  = (f"{breadth_snap.pct_above_200sma:.1f}%"
             if breadth_snap else "—")
    spy_status = ("Above" if (breadth_snap and breadth_snap.spy_above_200sma) else "Below")
    vix_str    = f"{vix_close:.1f}" if vix_close is not None else "—"
    uvol_dvol_str = (f"{breadth_snap.uvol_dvol_ratio:.2f}"
                     if breadth_snap else "—")
    eq_pcr_str    = (f"{breadth_snap.equity_pcr:.2f}"
                     if breadth_snap else "—")
    pcr_regime_str = (breadth_snap.pcr_regime.replace("_", " ").title()
                      if breadth_snap else "—")
    up_25m_str  = (str(breadth_snap.up_25pct_month) if breadth_snap else "—")
    dn_25m_str  = (str(breadth_snap.dn_25pct_month) if breadth_snap else "—")
    up_13_34_str = (str(breadth_snap.up_13pct_34d)  if breadth_snap else "—")
    dn_13_34_str = (str(breadth_snap.dn_13pct_34d)  if breadth_snap else "—")
    try:
        from quantlab.providers.cboe import classify_vix_regime as _cvr
        vix_regime_str = _cvr(vix_close)[0].capitalize() if vix_close is not None else "—"
    except Exception:
        vix_regime_str = "—"
    tape_c = _TAPE_COLORS.get(tape, C_DGRAY)

    e: list = []

    # ── Title block ───────────────────────────────────────────────────────────
    e.append(Paragraph("STH Capital — QuantLab Daily Briefing", S["title"]))
    e.append(Spacer(1, 0.1 * inch))
    e.append(Paragraph(report_date.strftime("%A, %B %d, %Y"), S["date_line"]))
    e.append(Spacer(1, 0.25 * inch))
    e.append(HRFlowable(width="100%", thickness=1, color=C_ACCENT, spaceAfter=12))

    if warning:
        e.append(Paragraph(
            f"<font color='#d63031'><b>{warning}</b></font>", S["body"],
        ))
        e.append(Spacer(1, 0.1 * inch))

    if options_rate:
        e.append(Paragraph(f"<b>{options_rate}</b>", S["body"]))
        e.append(Spacer(1, 0.1 * inch))

    if growth_funnel:
        e.append(Paragraph(f"<b>Growth funnel:</b> {growth_funnel}", S["body"]))
        e.append(Spacer(1, 0.1 * inch))

    if excluded_defensives:
        e.append(Paragraph(
            f"<font color='#636e72'><i>{excluded_defensives}</i></font>", S["body"]))
        e.append(Spacer(1, 0.1 * inch))

    # ── Tape badge (colored box via single-cell Table) ────────────────────────
    tape_inner = Paragraph(
        f"<font color='white'><b>{tape}</b></font>",
        ParagraphStyle("TBadge", fontName="Helvetica-Bold", fontSize=18,
                       textColor=white, alignment=TA_CENTER),
    )
    # 2.0 in wide so "CORRECTION" (10 chars, 18pt bold ≈ 116 pt) fits with margin.
    # Outer column is 2.2 in so the inner 2.0 in badge sits inside after cell padding.
    tape_tbl = Table([[tape_inner]], colWidths=[2.0 * inch], rowHeights=[0.42 * inch])
    tape_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), tape_c),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
    ]))
    e.append(Table([[tape_tbl, Spacer(0.1 * inch, 0.1 * inch)]],
                   colWidths=[2.2 * inch, CONTENT_W - 2.2 * inch]))
    e.append(Spacer(1, 0.15 * inch))

    # ── Regime exposure gate — suppressed entries are visible, never silent ──
    if regime_gate is not None:
        _gate_c = "#d63031" if regime_gate.entered == 0 and regime_gate.qualified > 0 else "#2d3436"
        e.append(Paragraph(
            f"<font color='{_gate_c}'><b>Entry gate:</b> {regime_gate.summary()}</font>",
            S["body"],
        ))
        e.append(Spacer(1, 0.12 * inch))

    # ── Market Internals — grouped sections ───────────────────────────────────
    e.append(Paragraph("Market Internals", S["section"]))

    # SPY MA distance values
    _spy21_pct  = getattr(breadth_snap, "spy_pct_above_21ema",  0.0) if breadth_snap else None
    _spy50_pct  = getattr(breadth_snap, "spy_pct_above_50sma",  0.0) if breadth_snap else None
    _spy200_pct = getattr(breadth_snap, "spy_pct_above_200sma", 0.0) if breadth_snap else None

    def _spy_str(pct):
        if pct is None:
            return "—"
        d = "Above" if pct >= 0 else "Below"
        return f"{d} ({pct:+.1f}%)"

    def _spy_color(pct):
        if pct is None or pct == 0.0:
            return C_DGRAY
        return C_GREEN if pct >= 0 else C_RED

    spy21_str  = _spy_str(_spy21_pct)
    spy50_str  = _spy_str(_spy50_pct)
    spy200_str = _spy_str(_spy200_pct)
    spy21_c    = _spy_color(_spy21_pct)
    spy50_c    = _spy_color(_spy50_pct)
    spy200_c   = _spy_color(_spy200_pct)

    # Row layout:  [label, value, label, value]
    # Header rows span all 4 columns (navy bg, white text).
    # SPY trend value spans cols 1-3 (data in col 0 only in col 1).
    _HDR_ROWS = [0, 4, 7, 11, 14]
    _SPY_ROWS = [8, 9, 10]   # rows where value cell gets SPY color
    met_data = [
        # 0 — section header
        ["Breadth Momentum", "", "", ""],
        # 1-3
        ["McClellan Oscillator", mcl,         "McClellan Summation",  summ],
        ["10-Day Breadth Ratio", ratio,        "AD Line",              ad_ln],
        ["UVOL / DVOL",          uvol_dvol_str,"",                     ""],
        # 4 — section header
        ["Market Participation (% of stocks above MA)", "", "", ""],
        # 5-6
        ["% Above 10 SMA", p10, "% Above 20 SMA", p20],
        ["% Above 50 SMA", p50, "% Above 200 SMA", p200],
        # 7 — section header
        ["SPY Trend", "", "", ""],
        # 8-10
        ["SPY vs 21 EMA",  spy21_str,  "", ""],
        ["SPY vs 50 SMA",  spy50_str,  "", ""],
        ["SPY vs 200 SMA", spy200_str, "", ""],
        # 11 — section header
        ["Volatility & Sentiment", "", "", ""],
        # 12-13
        ["VIX Close",   vix_str,     "VIX Regime",   vix_regime_str],
        ["Equity PCR",  eq_pcr_str,  "PCR Regime",   pcr_regime_str],
        # 14 — section header
        ["Momentum", "", "", ""],
        # 15-16
        ["Up 25% / Month", up_25m_str,  "Dn 25% / Month", dn_25m_str],
        ["Up 13% / 34d",   up_13_34_str,"Dn 13% / 34d",   dn_13_34_str],
    ]
    col_w = [1.8 * inch, 1.0 * inch, 1.8 * inch, 1.2 * inch]

    _met_style: list = [
        ("FONTNAME",  (0, 0), (-1, -1), "Helvetica"),
        ("FONTNAME",  (0, 0), (0, -1),  "Helvetica-Bold"),
        ("FONTNAME",  (2, 0), (2, -1),  "Helvetica-Bold"),
        ("FONTSIZE",  (0, 0), (-1, -1), 8.5),
        ("ALIGN",     (1, 0), (1, -1),  "RIGHT"),
        ("ALIGN",     (3, 0), (3, -1),  "RIGHT"),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [C_LGRAY, white]),
        ("GRID",      (0, 0), (-1, -1), 0.4, C_MGRAY),
        ("PADDING",   (0, 0), (-1, -1), 6),
    ]
    # Section headers: span + navy bg + white bold text
    for _hr in _HDR_ROWS:
        _met_style += [
            ("SPAN",        (0, _hr), (-1, _hr)),
            ("BACKGROUND",  (0, _hr), (-1, _hr), C_NAVY),
            ("TEXTCOLOR",   (0, _hr), (-1, _hr), white),
            ("FONTNAME",    (0, _hr), (-1, _hr), "Helvetica-Bold"),
            ("ALIGN",       (0, _hr), (-1, _hr), "LEFT"),
        ]
    # SPY trend rows: span value across cols 1-3; apply directional color
    for _sr, _sc in zip(_SPY_ROWS, [spy21_c, spy50_c, spy200_c]):
        _met_style += [
            ("SPAN",      (1, _sr), (-1, _sr)),
            ("TEXTCOLOR", (1, _sr), (1,  _sr), _sc),
            ("FONTNAME",  (1, _sr), (1,  _sr), "Helvetica-Bold"),
        ]

    met_tbl = Table(met_data, colWidths=col_w)
    met_tbl.setStyle(TableStyle(_met_style))
    e.append(met_tbl)

    # Momentum interpretation text
    if breadth_snap:
        _interp = momentum_interpretation(
            breadth_snap.up_25pct_month, breadth_snap.dn_25pct_month,
            breadth_snap.up_13pct_34d,  breadth_snap.dn_13pct_34d,
        )
        e.append(Spacer(1, 0.06 * inch))
        e.append(Paragraph(_interp, S["small"]))

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

    # ── Open position P&L ─────────────────────────────────────────────────────
    if open_positions:
        e.append(Spacer(1, 0.15 * inch))
        e.append(Paragraph("Open Position Monitor", S["section"]))

        pos_data = [["Symbol", "Entry", "Current", "Stop", "Target", "R-Mult", "P&L", "Days", "Status"]]
        row_cmds: list = []

        for ri, pos in enumerate(open_positions, 1):
            sym    = pos.get("symbol", "—")
            ep     = pos.get("entry_price") or 0.0
            cp     = pos.get("current_price")
            unrl   = pos.get("unrealized_ret")
            days_w = pos.get("days_on_watch", 0)
            status = (pos.get("status") or "watching").upper()
            atr_stp = pos.get("atr_stop")
            tgt     = pos.get("target_price")
            if tgt is None and ep and atr_stp and ep > atr_stp:
                tgt = ep + 2 * (ep - atr_stp)

            cp_str   = f"${cp:.2f}"         if cp       is not None else "—"
            unrl_str = f"{unrl*100:+.2f}%"  if unrl     is not None else "—"
            stop_str = f"${atr_stp:.2f}"    if atr_stp  is not None else "—"
            tgt_str  = f"${tgt:.2f}"        if tgt      is not None else "—"
            pnl_c    = C_GREEN if (unrl or 0) >= 0 else C_RED

            r_mult = _compute_r_multiple(ep, atr_stp, cp)
            if r_mult is not None:
                rm_str = f"{r_mult:+.1f}R"
                rm_c   = C_GREEN if r_mult > 0 else (C_RED if r_mult < 0 else C_TGRAY)
            else:
                rm_str = "—"
                rm_c   = C_TGRAY

            pos_data.append([
                sym, f"${ep:.2f}", cp_str, stop_str, tgt_str, rm_str, unrl_str, str(days_w), status,
            ])
            row_cmds += [
                ("FONTNAME",  (0, ri), (0, ri), "Helvetica-Bold"),
                ("TEXTCOLOR", (5, ri), (5, ri), rm_c),
                ("FONTNAME",  (5, ri), (5, ri), "Helvetica-Bold"),
                ("TEXTCOLOR", (6, ri), (6, ri), pnl_c),
                ("FONTNAME",  (6, ri), (6, ri), "Helvetica-Bold"),
            ]

        pos_tbl = Table(
            pos_data,
            colWidths=[
                0.72*inch, 0.68*inch, 0.68*inch, 0.68*inch,
                0.68*inch, 0.65*inch, 0.82*inch, 0.52*inch, 0.72*inch,
            ],
        )
        pos_tbl.setStyle(TableStyle([
            ("BACKGROUND",     (0, 0), (-1, 0), C_NAVY),
            ("TEXTCOLOR",      (0, 0), (-1, 0), white),
            ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",       (0, 0), (-1, -1), 7.5),
            ("ALIGN",          (1, 0), (-1, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, C_LGRAY]),
            ("GRID",           (0, 0), (-1, -1), 0.4, C_MGRAY),
            ("PADDING",        (0, 0), (-1, -1), 5),
        ] + row_cmds))
        e.append(pos_tbl)

    return e


# ── Page 2+ — Candidate Table ─────────────────────────────────────────────────

def _candidate_table(
    S: dict,
    candidates: list[dict],
    backtest_map: dict,
    revenue_map: dict | None = None,
    freshness_map: dict | None = None,
    eps_map: dict | None = None,
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

    # Include Rev % column only when coverage > 20% of candidates
    rev_map   = revenue_map or {}
    n_cands   = len(candidates)
    n_rev_cov = sum(1 for en in candidates if rev_map.get(en["symbol"]) is not None)
    show_rev  = n_cands > 0 and (n_rev_cov / n_cands) > 0.20

    if show_rev:
        headers   = ["#", "Symbol", "Stage", "Days", "Expl", "Conv", "Opts", "VDU",
                     "EPS %", "Rev %", "PEG", "Notes"]
        cw        = [0.28, 0.72, 0.60, 0.40, 0.40, 0.50, 0.38, 0.38, 0.55, 0.55, 0.45, 1.07]
    else:
        headers   = ["#", "Symbol", "Stage", "Days", "Expl", "Conv", "Opts", "VDU",
                     "EPS %", "PEG", "Notes"]
        cw        = [0.28, 0.72, 0.60, 0.40, 0.40, 0.50, 0.38, 0.38, 0.62, 0.45, 1.55]
    col_widths = [w * inch for w in cw]
    _notes_col = len(headers) - 1   # last column index

    rows = [headers]
    row_cmds: list = []

    for idx, en in enumerate(candidates, 1):
        stage  = en.get("stage", 0)
        days   = en.get("consecutive_days", 1)
        opts   = en.get("options_signal", False)
        _fresh = freshness_map.get(en["symbol"]) if freshness_map else None
        opts_str = _opts_cell(opts, _fresh)
        bkt    = backtest_map.get(en["symbol"], {})
        wr     = (f"WR:{bkt['win_rate']*100:.0f}%"
                  if bkt.get("win_rate") is not None else "")
        rev_v  = rev_map.get(en["symbol"])
        rev_str = _rev_pct(rev_v)

        if show_rev:
            rows.append([
                str(idx),
                en["symbol"],
                _stage_lbl(stage),
                f"{days}d",
                _fmt(en.get("explosion_score")),
                _fmt(en.get("conviction_score")),
                opts_str,
                _tick(en.get("volume_dry_up")),
                _eps_cell((eps_map or {}).get(en["symbol"])),
                rev_str,
                _fmt(en.get("peg_score")),
                wr,
            ])
        else:
            rows.append([
                str(idx),
                en["symbol"],
                _stage_lbl(stage),
                f"{days}d",
                _fmt(en.get("explosion_score")),
                _fmt(en.get("conviction_score")),
                opts_str,
                _tick(en.get("volume_dry_up")),
                _eps_cell((eps_map or {}).get(en["symbol"])),
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
        ("BACKGROUND",    (0, 0), (-1, 0),           C_NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0),           white),
        ("FONTNAME",      (0, 0), (-1, 0),           "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1),          7.5),
        ("FONTNAME",      (1, 1), (1, -1),           "Helvetica-Bold"),
        ("ALIGN",         (0, 0), (-1, -1),          "CENTER"),
        ("ALIGN",         (1, 0), (1, -1),           "LEFT"),
        ("ALIGN",         (_notes_col, 0), (_notes_col, -1), "LEFT"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1),          [white, C_LGRAY]),
        ("GRID",          (0, 0), (-1, -1),          0.3, C_MGRAY),
        ("LINEBELOW",     (0, 0), (-1, 0),           1.0, C_NAVY),
        ("PADDING",       (0, 0), (-1, -1),          4),
        ("TOPPADDING",    (0, 0), (-1, 0),           6),
        ("BOTTOMPADDING", (0, 0), (-1, 0),           6),
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
    revenue_map: dict | None = None,
    eps_map: dict | None = None,
) -> list:
    top5 = [c for c in candidates if c.get("consecutive_days", 1) >= 2][:5]
    if not top5:
        return []

    rev_map = revenue_map or {}

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
            f"   |   EPS YoY: {_eps_cell((eps_map or {}).get(en['symbol']))}"
            f"   |   Rev YoY: {_rev_pct(rev_map.get(sym))}"
            f"   |   PEG: {_fmt(en.get('peg_score'))}"
        )
        detail_line = (
            f"{'✓ Options signal' if en.get('options_signal') else '– No options signal'}"
            f"   |   {'✓ Volume dry-up' if en.get('volume_dry_up') else '– Vol normal'}"
            f"   |   Brkout Vol: {_bv_ratio(en.get('breakout_volume_ratio'))}"
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


# ── Basing Candidates (Stage 1) section ──────────────────────────────────────

def _basing_table(
    S: dict,
    basing_cands: list[dict],
    revenue_map: dict | None = None,
    freshness_map: dict | None = None,
    eps_map: dict | None = None,
) -> list:
    """Render 'Basing Candidates — Weekend Watchlist' for Stage 1 stocks."""
    if not basing_cands:
        return []

    rev_map = revenue_map or {}

    e: list = []
    e.append(Spacer(1, 0.2 * inch))
    e.append(HRFlowable(width=CONTENT_W, thickness=0.5, color=C_MGRAY))
    e.append(Spacer(1, 0.08 * inch))
    e.append(Paragraph("Basing Candidates — Weekend Watchlist", S["section"]))
    e.append(Paragraph(
        "Stage 1 stocks building a base.  Watch for Stage 2 breakout confirmation before acting.",
        S["small"],
    ))
    e.append(Spacer(1, 0.08 * inch))

    headers = ["#", "Symbol", "Days", "Conv", "Opts", "VDU", "EPS %", "Rev %", "PEG", "Notes"]
    cw = [0.28, 0.72, 0.45, 0.50, 0.38, 0.38, 0.62, 0.55, 0.45, 1.75]
    col_widths = [w * inch for w in cw]

    rows = [headers]
    for idx, en in enumerate(basing_cands, 1):
        rows.append([
            str(idx),
            en["symbol"],
            f"{en.get('consecutive_days', 1)}d",
            _fmt(en.get("conviction_score")),
            _opts_cell(en.get("options_signal", False),
                       freshness_map.get(en["symbol"]) if freshness_map else None),
            _tick(en.get("volume_dry_up", False)),
            _eps_cell((eps_map or {}).get(en["symbol"])),
            _rev_pct(rev_map.get(en["symbol"])),
            _fmt(en.get("peg_score")),
            "",
        ])

    style_cmds = [
        ("BACKGROUND",    (0, 0), (-1, 0),  C_DGRAY),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 7.5),
        ("FONTNAME",      (1, 1), (1, -1),  "Helvetica-Bold"),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("ALIGN",         (1, 0), (1, -1),  "LEFT"),
        ("ALIGN",         (9, 0), (9, -1),  "LEFT"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [white, C_LGRAY]),
        ("GRID",          (0, 0), (-1, -1), 0.3, C_MGRAY),
        ("LINEBELOW",     (0, 0), (-1, 0),  1.0, C_DGRAY),
        ("PADDING",       (0, 0), (-1, -1), 4),
        ("TOPPADDING",    (0, 0), (-1, 0),  6),
        ("BOTTOMPADDING", (0, 0), (-1, 0),  6),
    ]

    tbl = Table(rows, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle(style_cmds))
    e.append(tbl)
    return e


# ── Data helpers ───────────────────────────────────────────────────────────────

def _load_open_positions(db_path: str | None = None) -> list[dict]:
    """Return all 'watching' watchlist entries for the Open Position Monitor."""
    try:
        import duckdb
        from quantlab.storage import DB_PATH, _ensure_schema
        con = duckdb.connect(db_path or str(DB_PATH))
        _ensure_schema(con)
        rows = con.execute(
            """
            SELECT symbol, entry_price, current_price, unrealized_ret,
                   days_on_watch, status,
                   COALESCE(atr_stop, NULL) AS atr_stop,
                   COALESCE(target_price, NULL) AS target_price
            FROM watchlist
            WHERE status = 'watching'
            ORDER BY conviction_score DESC
            """
        ).fetchall()
        con.close()
        cols = ["symbol", "entry_price", "current_price", "unrealized_ret",
                "days_on_watch", "status", "atr_stop", "target_price"]
        return [dict(zip(cols, row)) for row in rows]
    except Exception:
        pass
    return []


def _load_revenue_map(symbols: list[str], db_path: str | None = None) -> dict:
    """Return {symbol: revenue_yoy_pct} from the edgar_fundamentals cache."""
    if not symbols:
        return {}
    try:
        import duckdb
        from quantlab.storage import DB_PATH
        con = duckdb.connect(db_path or str(DB_PATH))
        ph  = ",".join("?" * len(symbols))
        rows = con.execute(
            f"SELECT symbol, revenue_growth FROM edgar_fundamentals "
            f"WHERE symbol IN ({ph}) "
            f"ORDER BY fetch_date DESC",
            symbols,
        ).fetchall()
        con.close()
        result: dict = {}
        for sym, rev in rows:
            if sym not in result:
                result[sym] = rev
        return result
    except Exception:
        return {}


def _load_eps_map(symbols: list[str], db_path: str | None = None) -> dict:
    """Return {symbol: {"eps_yoy", "turned_positive"}} from edgar_fundamentals.

    eps_yoy is the RAW period-matched YoY rate (uncapped; None = not
    computable / quarantined); turned_positive marks a negative→positive EPS
    transition (max-strength earnings event stored as NULL%).
    """
    if not symbols:
        return {}
    try:
        import duckdb
        from quantlab.storage import DB_PATH
        con = duckdb.connect(db_path or str(DB_PATH))
        ph  = ",".join("?" * len(symbols))
        rows = con.execute(
            f"SELECT symbol, eps_growth, eps_turned_positive "
            f"FROM edgar_fundamentals "
            f"WHERE symbol IN ({ph}) "
            f"ORDER BY fetch_date DESC",
            symbols,
        ).fetchall()
        con.close()
        result: dict = {}
        for sym, eps, tp in rows:
            if sym not in result:
                result[sym] = {"eps_yoy": eps, "turned_positive": bool(tp)}
        return result
    except Exception:
        return {}


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


def _load_options_freshness(
    report_date: date,
    db_path: str | None = None,
) -> dict[str, dict] | None:
    """
    Per-symbol flag freshness for the session: symbol → {unusual_flag,
    flag_streak, first_flagged_date, put_dominated}.  None when the session
    has no gated snapshot rows (pre-recalibration or monitor didn't run) —
    callers then fall back to the legacy ✓/– rendering.
    """
    try:
        import duckdb
        from quantlab.storage import DB_PATH
        con = duckdb.connect(str(db_path or DB_PATH), read_only=True)
        rows = con.execute(
            "SELECT symbol, unusual_flag, flag_streak, first_flagged_date, "
            "       put_dominated "
            "FROM options_snapshots "
            "WHERE snap_date = ? AND rel_score IS NOT NULL",
            [report_date],
        ).fetchall()
        con.close()
    except Exception:
        return None
    if not rows:
        return None
    return {
        sym: {
            "unusual_flag": flag,
            "flag_streak": streak,
            "first_flagged_date": first,
            "put_dominated": put_dom,
        }
        for sym, flag, streak, first, put_dom in rows
    }


def _options_signal_rate(
    report_date: date,
    db_path: str | None = None,
) -> str | None:
    """
    Render the day's options signal rate, e.g. "Options: 31/357 unusual, 8.7%".

    Counterpart to the zero-signal warning, for the opposite failure: a drift
    back toward saturation (the detector flagged 97% of monitored symbols on
    2026-06-11 before recalibration) must be visible in the header at a glance.

    Uses the recalibrated columns (rel_score / unusual_flag) when the session
    has them; falls back to the legacy unusual_calls count (marked "legacy")
    for pre-recalibration sessions.  Returns None when the session has no
    options_snapshots rows at all.
    """
    n_long_streak = 0
    try:
        import duckdb
        from quantlab.storage import DB_PATH
        con = duckdb.connect(str(db_path or DB_PATH), read_only=True)
        total, n_scored, n_flagged, n_legacy = con.execute(
            "SELECT COUNT(*), "
            "       COUNT(rel_score), "
            "       COALESCE(SUM(CASE WHEN unusual_flag THEN 1 ELSE 0 END), 0), "
            "       COALESCE(SUM(CASE WHEN unusual_calls THEN 1 ELSE 0 END), 0) "
            "FROM options_snapshots WHERE snap_date = ?",
            [report_date],
        ).fetchone()
        try:
            # Baseline-inflation watch: campaigns this long have materially
            # fed their own trailing baseline (column absent on old DBs)
            n_long_streak = con.execute(
                "SELECT COUNT(*) FROM options_snapshots "
                "WHERE snap_date = ? AND flag_streak >= 5",
                [report_date],
            ).fetchone()[0]
        except Exception:
            n_long_streak = 0
        con.close()
    except Exception:
        return None

    if not total:
        return None
    basis = _session_basis_suffix(report_date, db_path)
    if n_scored > 0:
        line = (f"Options: {n_flagged}/{n_scored} unusual, "
                f"{n_flagged / n_scored:.1%}{basis}")
        if n_long_streak:
            line += (f"  |  {n_long_streak} streak≥5 "
                     f"(baseline-inflation watch)")
        return line
    return (f"Options: {n_legacy}/{total} unusual, "
            f"{n_legacy / total:.1%} (legacy scorer){basis}")


def _session_basis_suffix(report_date: date, db_path: str | None = None) -> str:
    """
    Finalization-basis suffix for the options header line: " (final)" once the
    session has been rescored against the EOD flat file, otherwise
    " (intraday — finalizes overnight)".  Empty string only if the status table
    is unavailable (old DB) — silence beats a misleading "(final)".
    """
    try:
        from quantlab.options_finalize import get_session_status
        st = get_session_status(report_date, db_path=db_path)
    except Exception:
        return ""
    if st is None:
        return " (intraday — finalizes overnight)"
    return " (final)" if st.get("finalized") else " (intraday — finalizes overnight)"


def _stale_finalization_warning(db_path: str | None = None) -> str | None:
    """
    WARNING string when any options session is still unfinalized past noon the
    next trading day — finalization has stalled (most likely a credential /
    permission failure that must not silently persist).  None when all clear.
    """
    try:
        from quantlab.options_finalize import stale_unfinalized_sessions
        stale = stale_unfinalized_sessions(db_path=db_path)
    except Exception:
        return None
    if not stale:
        return None
    dates = ", ".join(d.isoformat() for d in stale)
    return (f"WARNING: options finalization STALLED — session(s) {dates} still "
            f"unfinalized past noon the next trading day; check POLYGON_S3 "
            f"credentials (manual: scripts/rescore_options_session.py --write).")


def _universe_gate_warning(report_date: date, db_path: str | None = None) -> str | None:
    """
    WARNING string when the universe sanity gate refused the build for
    ``report_date`` (universe_history.gate_accepted = FALSE) — the scan ran
    on a prior day's universe.  None when the build was accepted or absent.
    """
    try:
        import duckdb
        from quantlab.storage import DB_PATH
        con = duckdb.connect(str(db_path or DB_PATH), read_only=True)
        row = con.execute(
            "SELECT final_count FROM universe_history "
            "WHERE date = ? AND gate_accepted = FALSE",
            [report_date],
        ).fetchone()
        con.close()
    except Exception:
        return None
    if row is None:
        return None
    return (
        f"WARNING: UNIVERSE GATE refused the {report_date} build "
        f"(degenerate count {row[0]:,}) — scan ran on the prior day's "
        f"universe; streak counters treated this day as neutral."
    )


def _growth_funnel_line(report_date: date, db_path: str | None = None) -> str | None:
    """One-line growth pre-filter funnel for the report header, or None.

    Reads the persisted growth_universe rows for ``report_date`` and renders the
    sequential per-gate counts (binding constraint always visible).  When zero
    names are growth-qualified — possible in a deep correction — that is stated
    explicitly rather than hidden; basing/watchlist panels still render below.
    """
    try:
        from quantlab.growth_filter import load_growth_funnel
        f = load_growth_funnel(report_date, db_path)
    except Exception:
        return None
    if f is None:
        return None
    line = f.render()
    if f.growth_qualified == 0:
        line += "  —  ZERO growth-qualified (correction tape); see basing/watchlist below"
    # Explain acceleration's silence so a dormant signal isn't read as "nothing
    # accelerating" (it qualifies 0 until names accrue ≥5q of stored history).
    line += f"<br/>{f.accel_status()}"
    return line


def _excluded_defensive_line(report_date: date, db_path: str | None = None) -> str | None:
    """Optional 'tape character' context: top defensive names the growth filter
    excluded at a hard gate.  OFF by default — enabled via scanner config
    growth_filter.show_excluded_panel.  Market context only, never candidates.
    """
    try:
        from quantlab.utils import get_config
        if not get_config("scanner").get("growth_filter", {}).get("show_excluded_panel", False):
            return None
        from quantlab.growth_filter import load_excluded_defensive
        rows = load_excluded_defensive(report_date, limit=10, db_path=db_path)
    except Exception:
        return None
    if not rows:
        return None
    parts = []
    for r in rows:
        cap = f"{r['market_cap']/1e9:.0f}B" if r.get("market_cap") else "—"
        parts.append(f"{r['symbol']} ({cap})")
    return "Excluded defensives (tape character): " + ", ".join(parts)


def _options_pipeline_warning(
    candidates: list[dict],
    basing_cands: list[dict],
    report_date: date,
    db_path: str | None = None,
) -> str | None:
    """
    Return a WARNING string when zero watchlist entries carry an options signal.

    Universe-wide zero is treated as a pipeline fault until proven otherwise:
    when the intraday monitor recorded options_snapshots rows for the session
    but no entry is flagged, the signal was detected and then lost downstream.
    Returns None when at least one entry has options_signal=True.
    """
    n_flagged = sum(
        1 for c in [*candidates, *basing_cands] if c.get("options_signal")
    )
    if n_flagged > 0:
        return None

    snap_total, snap_unusual = 0, 0
    try:
        import duckdb
        from quantlab.storage import DB_PATH
        con = duckdb.connect(str(db_path or DB_PATH), read_only=True)
        snap_total, snap_unusual = con.execute(
            "SELECT COUNT(*), "
            "       COALESCE(SUM(CASE WHEN unusual_calls THEN 1 ELSE 0 END), 0) "
            "FROM options_snapshots WHERE snap_date = ?",
            [report_date],
        ).fetchone()
        con.close()
    except Exception:
        pass

    if snap_total > 0:
        return (
            f"WARNING: 0 options signals in this report, but the intraday monitor "
            f"recorded {snap_total} snapshots ({snap_unusual} unusual) for "
            f"{report_date} — options signal pipeline fault likely."
        )
    return (
        f"WARNING: 0 options signals and no intraday monitor data for "
        f"{report_date} — options monitor may not have run."
    )


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
    # Main table: Stage 2 advancing stocks only (defensive filter; upsert already routes)
    _cands_pre   = [c for c in iwl.get_candidates() if c.get("stage", 0) == 2]
    # Basing section: Stage 1 stocks from the separate basing_watchlist table
    basing_cands = iwl.get_basing_candidates()
    breadth      = get_latest_snapshot()

    # EPS gate: exclude symbols with deeply negative earnings growth at report layer.
    # This prevents fundamentally deteriorating stocks (e.g. eps_growth < -10%) from
    # appearing in the candidate table even if they passed technical conviction thresholds.
    _pre_syms    = [c["symbol"] for c in _cands_pre]
    _basing_syms = [c["symbol"] for c in basing_cands]
    eps_map      = _load_eps_map(_pre_syms + _basing_syms, db_path)

    def _passes_eps_gate(sym: str) -> bool:
        e = eps_map.get(sym)
        if e is None:
            return True                      # no data — gate can't judge
        if e.get("turned_positive"):
            return True                      # max-strength earnings event
        v = e.get("eps_yoy")
        return v is None or v >= -0.10

    candidates = [c for c in _cands_pre if _passes_eps_gate(c["symbol"])]

    symbols         = [c["symbol"] for c in candidates]
    backtest_map    = _load_backtest_map(symbols, db_path)
    revenue_map     = _load_revenue_map(symbols, db_path)
    basing_symbols  = [c["symbol"] for c in basing_cands]
    basing_rev_map  = _load_revenue_map(basing_symbols, db_path)
    open_positions  = _load_open_positions(db_path)
    n_multi         = sum(1 for c in candidates if c.get("consecutive_days", 1) >= 2)

    # VIX close (non-fatal; defaults to None → shown as "—" in report)
    vix_close: float | None = None
    try:
        from datetime import timedelta as _td
        from quantlab.providers.cboe import fetch_vix_history
        _vix_bars = fetch_vix_history(report_date - _td(days=7), report_date)
        if _vix_bars:
            vix_close = _vix_bars[-1].close
    except Exception:
        pass

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

    # Universe-wide zero options signals must never pass silently as normal
    opts_warning = _options_pipeline_warning(candidates, basing_cands,
                                             report_date, db_path)
    if opts_warning:
        print(f"  {opts_warning}")

    # Universe sanity-gate refusal — the scan ran on a prior day's universe
    gate_warning = _universe_gate_warning(report_date, db_path)
    if gate_warning:
        print(f"  {gate_warning}")

    # Finalization stalled past the noon-next-day deadline — credential/permission
    # failure must not silently persist (item 3c loud alert)
    stale_warning = _stale_finalization_warning(db_path)
    if stale_warning:
        print(f"  {stale_warning}")

    header_warnings = [w for w in (opts_warning, gate_warning, stale_warning) if w]
    header_warning = "<br/>".join(header_warnings) if header_warnings else None

    # Day's options signal rate — saturation drift must be visible immediately
    opts_rate = _options_signal_rate(report_date, db_path)
    if opts_rate:
        print(f"  {opts_rate}")

    # Regime gate decision from today's scan (None when no scan ran today)
    regime_gate = None
    try:
        from quantlab.risk.regime_policy import load_regime_gate
        regime_gate = load_regime_gate(report_date, db_path)
    except Exception:
        pass
    if regime_gate:
        print(f"  regime gate: {regime_gate.summary()}")

    # Flag freshness: F fresh / FdN streak rendering in the Opts columns
    freshness_map = _load_options_freshness(report_date, db_path)

    growth_funnel = _growth_funnel_line(report_date, db_path)
    if growth_funnel:
        print(f"  growth funnel: {growth_funnel}")
    excluded_defensives = _excluded_defensive_line(report_date, db_path)

    story: list = []
    story += _page1(S, report_date, breadth, len(candidates), n_multi, open_positions,
                    vix_close=vix_close, warning=header_warning,
                    options_rate=opts_rate, regime_gate=regime_gate,
                    growth_funnel=growth_funnel,
                    excluded_defensives=excluded_defensives)
    if candidates:
        story += _candidate_table(S, candidates, backtest_map, revenue_map,
                                  freshness_map=freshness_map, eps_map=eps_map)
        story += _alert_section(S, candidates, backtest_map, revenue_map,
                                eps_map=eps_map)
    if basing_cands:
        story += _basing_table(S, basing_cands, basing_rev_map,
                               freshness_map=freshness_map, eps_map=eps_map)

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
