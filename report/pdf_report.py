"""
Phase 9 — Faculty Feedback Report (PDF).

The IRB protocol promises each participating instructor a "Faculty Feedback
Report": a concrete, actionable summary that translates the COPUS AI analysis
into something useful for the professor. This module builds that multi-page PDF
from the per-lecture and professor-level outputs `run.py` already produces.

Charts are rendered with plotly and exported to static PNG via kaleido, then
embedded with reportlab. Per-lecture narratives and recommendations come from
Gemini when credentials are available; when they are not (offline, no ADC), a
deterministic data-driven fallback keeps the report fully renderable.

Public entry point: `generate_faculty_report(...)`. A `main()` CLI wraps it so
a report can be built directly from a `run.py` output directory.
"""
import os
import argparse
from datetime import date
from xml.sax.saxutils import escape as xml_escape

import pandas as pd
import plotly.graph_objects as go

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    KeepTogether, SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
)

from report.dashboard import add_timeline_traces, CODE_COLORS

# COPUS instructor codes and their meanings (from the protocol).
CODE_MEANINGS = {
    "Lec": "Lecturing — presenting content",
    "RtW": "Real-time writing on board/doc camera",
    "FUp": "Follow-up/feedback on activity to class",
    "PQ":  "Posing a non-clicker question",
    "CQ":  "Asking a clicker question",
    "AnQ": "Answering student questions",
    "MG":  "Moving through class guiding student work",
    "1o1": "One-on-one extended discussion",
    "D/V": "Showing demo, experiment, simulation, video",
    "Adm": "Administration",
    "W":   "Waiting",
    "O":   "Other",
}

# Codes that represent active-learning / interactive instruction, as opposed to
# transmission-style teaching. Used for the lecturing-vs-active-learning split.
ACTIVE_LEARNING_CODES = {"FUp", "PQ", "CQ", "AnQ", "MG", "1o1", "D/V"}
# Passive / transmission-style codes (the rest that count as "teaching time").
PASSIVE_CODES = {"Lec", "RtW", "Adm", "W"}

from validate.convert_copus_sheet import INSTRUCTOR_CODES
from validate.validator import (
    AC1_CLEAR_LABEL, KAPPA_CLEAR_LABEL, MEAN_ROW_LABEL, SUMMARY_LABELS, mean_kappa,
)
from report.feedback_sections import (
    DEFAULT_DATA_DIR, code_shares, cucei_context_lines, cucei_profile,
    DEFAULT_GOLDEN_DIR, cucei_recommendation, golden_lectures, linking_observations,
)

# Golden-comparison chart: the report's navy for the professor, a muted gold for
# the reference lectures.
PROFESSOR_COLOR = "#003366"
GOLDEN_COLOR = "#C9A227"

# Reliability bands for Cohen's kappa (Landis & Koch).
KAPPA_BANDS = [
    (0.81, "Almost perfect", colors.HexColor("#2E7D32")),
    (0.61, "Substantial",    colors.HexColor("#66BB6A")),
    (0.41, "Moderate",       colors.HexColor("#FBC02D")),
    (0.21, "Fair",           colors.HexColor("#FB8C00")),
    (0.00, "Slight",         colors.HexColor("#EF5350")),
    (-1.0, "Poor / none",    colors.HexColor("#C62828")),
]


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def _explode_codes(df):
    """Yield every (row, code) pair, splitting the pipe-delimited codes cell."""
    for _, r in df.iterrows():
        cell = r.get("copus_codes")
        if pd.isna(cell) or not str(cell).strip():
            continue
        for code in str(cell).split("|"):
            code = code.strip()
            if code:
                yield r, code


def code_frequency(df):
    """
    Per-code window counts for one lecture's results DataFrame.

    Returns a DataFrame [code, windows_present, pct_of_windows] sorted by
    frequency. "% of total windows" uses the number of windows in the lecture
    (a window can carry several codes, so percentages can sum past 100%).
    """
    total_windows = len(df)
    counts = {}
    for _, code in _explode_codes(df):
        counts[code] = counts.get(code, 0) + 1
    rows = [
        {
            "code": code,
            "windows_present": n,
            "pct_of_windows": round(100.0 * n / total_windows, 1) if total_windows else 0.0,
        }
        for code, n in counts.items()
    ]
    rows.sort(key=lambda r: r["windows_present"], reverse=True)
    return pd.DataFrame(rows, columns=["code", "windows_present", "pct_of_windows"])


def active_learning_split(df):
    """
    Fraction of code-instances that are active-learning vs passive for one
    lecture. Returns (active_pct, passive_pct) over classified code instances
    (instances whose code falls in neither set are ignored).
    """
    active = passive = 0
    for _, code in _explode_codes(df):
        if code in ACTIVE_LEARNING_CODES:
            active += 1
        elif code in PASSIVE_CODES:
            passive += 1
    total = active + passive
    if not total:
        return 0.0, 0.0
    return round(100.0 * active / total, 1), round(100.0 * passive / total, 1)


def _kappa_band(value):
    """Map a kappa value to (label, color). Non-numeric → neutral 'N/A'."""
    v = pd.to_numeric(value, errors="coerce")
    if pd.isna(v):
        return "N/A", colors.HexColor("#9E9E9E")
    for threshold, label, color in KAPPA_BANDS:
        if v >= threshold:
            return label, color
    return "Poor / none", colors.HexColor("#C62828")


def load_kappa_dict(kappa_results):
    """
    Normalize the `kappa_results` argument into
        {comparison_name: {code: kappa}}.

    Accepts either that dict directly, or a path to a `combined_kappa.csv`
    (columns: code, ai_vs_sofia_kappa).
    """
    if kappa_results is None:
        return {}
    if isinstance(kappa_results, dict):
        return kappa_results
    # Treat as a combined_kappa.csv path.
    df = pd.read_csv(kappa_results)
    out = {}
    for col, name in (
        ("ai_vs_sofia_kappa", "AI vs Sofia"),
    ):
        if col in df.columns:
            # N/A rows are KEPT: a code neither rater used must still show as
            # N/A in the report, not silently disappear from the table.
            out[name] = {
                r["code"]: ("N/A" if str(r[col]).strip() in ("", "N/A", "nan")
                            else r[col])
                for _, r in df.iterrows()
                # Threshold-count rows are counts of codes, not kappas.
                if r["code"] not in SUMMARY_LABELS
            }
    return out


# --------------------------------------------------------------------------- #
# Chart rendering (plotly -> PNG via kaleido)
# --------------------------------------------------------------------------- #
def render_timeline_png(results_csv, png_path, width=800, height=360):
    """
    Render a lecture's behavioral timeline to a static PNG for PDF embedding.
    Reuses the dashboard's timeline traces so the PDF matches the HTML view.
    """
    df = pd.read_csv(results_csv)
    fig = go.Figure()
    add_timeline_traces(fig, df)
    fig.update_layout(
        xaxis_title="Time (seconds)",
        yaxis_title="COPUS Code",
        barmode="overlay",
        plot_bgcolor="white",
        margin=dict(l=50, r=20, t=20, b=40),
        showlegend=False,
    )
    os.makedirs(os.path.dirname(png_path) or ".", exist_ok=True)
    fig.write_image(png_path, width=width, height=height)
    return png_path


def render_active_learning_png(active_pct, passive_pct, png_path,
                               width=360, height=300):
    """Small donut of the active-learning vs lecturing split for the summary."""
    fig = go.Figure(go.Pie(
        labels=["Active learning", "Lecturing / passive"],
        values=[active_pct, passive_pct],
        hole=0.55,
        marker_colors=[CODE_COLORS.get("PQ", "#C44E52"),
                       CODE_COLORS.get("Lec", "#4C72B0")],
        textinfo="label+percent",
        sort=False,
    ))
    fig.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    os.makedirs(os.path.dirname(png_path) or ".", exist_ok=True)
    fig.write_image(png_path, width=width, height=height)
    return png_path


# --------------------------------------------------------------------------- #
# AI-generated narrative (Gemini, with an offline fallback)
# --------------------------------------------------------------------------- #
def _gemini_text(prompt, model="gemini-2.5-flash"):
    """
    Best-effort single-shot Gemini call. Returns the text, or None if the client
    can't be built or the call fails (offline / no credentials).
    """
    try:
        from classify.classifier import get_gemini_client
        client = get_gemini_client()
        resp = client.models.generate_content(model=model, contents=prompt)
        return (resp.text or "").strip() or None
    except Exception as e:  # noqa: BLE001 — narrative is optional, never fatal
        print(f"[pdf_report] Gemini narrative unavailable, using fallback: {e}")
        return None


def _fallback_lecture_narrative(freq, active_pct, passive_pct):
    """Deterministic per-lecture summary when Gemini is unavailable."""
    if freq.empty:
        return "No instructor codes were recorded for this lecture."
    top = freq.iloc[0]
    top_meaning = CODE_MEANINGS.get(top["code"], top["code"])
    parts = [
        f"This lecture was predominantly characterized by "
        f"'{top['code']}' ({top_meaning.lower()}), present in "
        f"{top['pct_of_windows']}% of windows.",
        f"Active-learning behaviors accounted for about {active_pct}% of "
        f"classified instructor time, versus {passive_pct}% lecturing/passive.",
    ]
    if len(freq) > 1:
        others = ", ".join(freq["code"].iloc[1:4])
        parts.append(f"Other observed behaviors included {others}.")
    return " ".join(parts)


def lecture_narrative(results_csv, freq, active_pct, passive_pct, survey_summary=None):
    """
    3–4 sentence pedagogical summary of one lecture. Tries Gemini first; on any
    failure returns a data-driven fallback so the report always has content.
    """
    freq_lines = "\n".join(
        f"- {r['code']} ({CODE_MEANINGS.get(r['code'], r['code'])}): "
        f"{r['windows_present']} windows ({r['pct_of_windows']}%)"
        for _, r in freq.iterrows()
    ) or "- (no codes recorded)"
    prompt = (
        "You are a pedagogical consultant analyzing COPUS classroom observation "
        "data for a STEM instructor. Based on the code frequencies below, write a "
        "3-4 sentence summary of this lecture's teaching style. Be specific and "
        "constructive; do not evaluate or grade the instructor.\n\n"
        f"Active-learning share: {active_pct}%. Lecturing/passive share: {passive_pct}%.\n"
        f"Code frequencies:\n{freq_lines}\n"
    )
    if survey_summary:
        prompt += f"\nStudent survey context:\n{survey_summary}\n"
    text = _gemini_text(prompt)
    if not text:
        return _fallback_lecture_narrative(freq, active_pct, passive_pct)
    return text


def _fallback_recommendations(overall_freq, overall_active_pct):
    """Deterministic recommendations when Gemini is unavailable."""
    recs = []
    present = set(overall_freq["code"]) if not overall_freq.empty else set()
    if overall_active_pct < 30:
        recs.append(
            "Active-learning behaviors made up a small share of instructor time. "
            "Consider adding brief posed questions (PQ) or clicker questions (CQ) "
            "to interrupt extended lecturing segments."
        )
    if "PQ" not in present and "CQ" not in present:
        recs.append(
            "No questioning behaviors (PQ/CQ) were observed. Introducing a "
            "question in the first 20 minutes can surface student misconceptions "
            "early."
        )
    if "MG" not in present:
        recs.append(
            "Moving through the class to guide student work (MG) was not "
            "observed; circulating during any activity increases opportunities "
            "for informal feedback."
        )
    if not recs:
        recs.append(
            "The lectures show a balanced mix of behaviors. Continuing to vary "
            "questioning and guided-work segments will help sustain engagement."
        )
    return recs


def recommendations(overall_freq, overall_active_pct, survey_summary=None,
                    shares=None, cucei_rows=None):
    """
    2–3 constructive, framed-as-suggestion recommendations across all lectures.
    Returns a list of strings. Gemini first, deterministic fallback otherwise.

    With CUCEI scores (cucei_rows), Gemini is shown each score against its
    comparison point and may tie one recommendation to the lowest-rated
    dimension; the fallback leads with cucei_recommendation() instead.
    """
    freq_lines = "\n".join(
        f"- {r['code']} ({CODE_MEANINGS.get(r['code'], r['code'])}): "
        f"{r['windows_present']} windows"
        for _, r in overall_freq.iterrows()
    ) or "- (no codes recorded)"
    prompt = (
        "You are a pedagogical consultant. Based on the aggregated COPUS "
        "behavioral data below (across all analyzed lectures), write 2-3 "
        "specific, constructive recommendations for the instructor. Frame each "
        "as a suggestion, not an evaluation, consistent with COPUS philosophy. "
        "Return each recommendation on its own line, no numbering.\n\n"
        f"Overall active-learning share: {overall_active_pct}%.\n"
        f"Aggregated code frequencies:\n{freq_lines}\n"
    )
    if survey_summary:
        prompt += f"\nStudent survey context:\n{survey_summary}\n"
    cucei_lines = cucei_context_lines(cucei_rows)
    if cucei_lines:
        prompt += (
            "\nStudent perceptions (CUCEI, 1-4 scale, higher is more favorable):\n"
            + "\n".join(cucei_lines) + "\n"
            "If any dimension is below its comparison point, you may tie AT MOST ONE "
            "recommendation to the lowest such dimension, naming a concrete, "
            "observable instructor behavior. Describe the score as students' "
            "perception of the course; do not claim that any behavior caused it.\n")

    cucei_rec = cucei_recommendation(shares or {}, cucei_rows)

    def fallback():
        recs = _fallback_recommendations(overall_freq, overall_active_pct)
        # The CUCEI-grounded suggestion leads: it is the only one built from
        # this professor's own students' responses.
        return ([cucei_rec] + recs)[:3] if cucei_rec else recs

    text = _gemini_text(prompt)
    if not text:
        return fallback()
    recs = [ln.strip(" -•\t") for ln in text.splitlines() if ln.strip()]
    return recs or fallback()


def _survey_summary_text(survey_data):
    """
    Normalize survey_data (dict or path to survey_analysis.csv) into a short
    text block for prompts and a DataFrame for the survey page. Returns
    (summary_text, dataframe) or (None, None) when no survey is provided.
    """
    if survey_data is None:
        return None, None
    if isinstance(survey_data, str):
        try:
            df = pd.read_csv(survey_data)
        except Exception as e:  # noqa: BLE001
            print(f"[pdf_report] could not read survey {survey_data}: {e}")
            return None, None
    elif isinstance(survey_data, pd.DataFrame):
        df = survey_data
    elif isinstance(survey_data, dict):
        df = pd.DataFrame(survey_data)
    else:
        return None, None
    if df.empty:
        return None, df
    lines = []
    for _, r in df.iterrows():
        q = r.get("question", "")
        if str(r.get("type", "")) == "numeric":
            lines.append(f"- {q}: mean {r.get('mean')} (n={r.get('n')})")
        else:
            lines.append(f"- {q}: top response {r.get('top_response')} (n={r.get('n')})")
    return "\n".join(lines), df


# --------------------------------------------------------------------------- #
# PDF assembly
# --------------------------------------------------------------------------- #
def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        "CoverTitle", parent=styles["Title"], fontSize=28, spaceAfter=18,
        textColor=colors.HexColor("#003366")))
    styles.add(ParagraphStyle(
        "CoverMeta", parent=styles["Normal"], fontSize=14, spaceAfter=6,
        alignment=1))
    styles.add(ParagraphStyle(
        "Disclaimer", parent=styles["Normal"], fontSize=9, alignment=1,
        textColor=colors.grey, spaceBefore=24))
    styles.add(ParagraphStyle(
        "SectionH", parent=styles["Heading1"], fontSize=18,
        textColor=colors.HexColor("#003366"), spaceAfter=10))
    styles.add(ParagraphStyle(
        "SubH", parent=styles["Heading2"], fontSize=13,
        textColor=colors.HexColor("#003366"), spaceBefore=14, spaceAfter=2))
    styles.add(ParagraphStyle(
        "Narrative", parent=styles["Normal"], fontSize=10.5, leading=15,
        spaceBefore=8, spaceAfter=8))
    return styles


def _freq_table(freq, styles):
    """A reportlab Table of code | meaning | windows | % for one lecture."""
    data = [["Code", "Meaning", "Windows", "% of Windows"]]
    for _, r in freq.iterrows():
        data.append([
            r["code"],
            CODE_MEANINGS.get(r["code"], "—"),
            str(int(r["windows_present"])),
            f"{r['pct_of_windows']}%",
        ])
    if len(data) == 1:
        data.append(["—", "No codes recorded", "0", "0%"])
    table = Table(data, colWidths=[0.7 * inch, 3.2 * inch, 0.9 * inch, 1.1 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#003366")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F5F8")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return table


def _kappa_table(kappa_dict):
    """
    Validation table: one row per code, one column per comparison, cells colored
    by reliability band. `kappa_dict` is {comparison: {code: kappa}}.
    """
    comparisons = list(kappa_dict.keys())
    codes = sorted({c for m in kappa_dict.values() for c in m})
    # MEAN is a summary row, not a COPUS code — keep it pinned at the bottom
    # instead of alphabetized in among Adm/AnQ/CQ.
    if MEAN_ROW_LABEL in codes:
        codes = [c for c in codes if c != MEAN_ROW_LABEL] + [MEAN_ROW_LABEL]
    header = ["Code"] + comparisons
    data = [header]
    for code in codes:
        row = [code]
        for comp in comparisons:
            v = kappa_dict[comp].get(code, "N/A")
            vnum = pd.to_numeric(v, errors="coerce")
            row.append("N/A" if pd.isna(vnum) else f"{vnum:.2f}")
        data.append(row)

    col_widths = [0.9 * inch] + [ (5.4 / max(len(comparisons), 1)) * inch
                                  for _ in comparisons]
    table = Table(data, colWidths=col_widths)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#003366")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ]
    # Color kappa cells by reliability band.
    for ri, code in enumerate(codes, start=1):
        for ci, comp in enumerate(comparisons, start=1):
            _, color = _kappa_band(kappa_dict[comp].get(code, "N/A"))
            style.append(("TEXTCOLOR", (ci, ri), (ci, ri), color))
        if code == MEAN_ROW_LABEL:
            style.append(("LINEABOVE", (0, ri), (-1, ri), 1,
                          colors.HexColor("#003366")))
            style.append(("FONTNAME", (0, ri), (-1, ri), "Helvetica-Bold"))
    table.setStyle(TableStyle(style))
    return table


def _legend_flowable(styles):
    """A small reliability-band legend paragraph for the validation page."""
    bits = []
    for _, label, color in KAPPA_BANDS[:-1]:
        bits.append(f'<font color="#{color.hexval()[2:]}">■ {label}</font>')
    return Paragraph("Reliability bands (κ and AC1): " + "  ".join(bits),
                     styles["Normal"])


def render_golden_png(prof_shares, golden_shares, png_path, width=760, height=380,
                      golden_label="Reference lecture"):
    """
    Grouped horizontal bars: % of windows per COPUS code, this professor vs ONE
    golden reference lecture. Codes appear in protocol order; a code shows when
    either profile used it.
    """
    codes = [c for c in INSTRUCTOR_CODES
             if prof_shares.get(c, 0) > 0 or golden_shares.get(c, 0) > 0]
    codes = list(reversed(codes))  # plotly draws the first category at the bottom
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=codes, x=[golden_shares.get(c, 0) for c in codes], orientation="h",
        name=golden_label, marker_color=GOLDEN_COLOR,
        text=[f"{golden_shares.get(c, 0):g}%" for c in codes], textposition="outside"))
    fig.add_trace(go.Bar(
        y=codes, x=[prof_shares.get(c, 0) for c in codes], orientation="h",
        name="Your lectures", marker_color=PROFESSOR_COLOR,
        text=[f"{prof_shares.get(c, 0):g}%" for c in codes], textposition="outside"))
    fig.update_layout(
        barmode="group", plot_bgcolor="white",
        xaxis=dict(title="% of analyzed 2-minute segments", range=[0, 115],
                   showgrid=True, gridcolor="#EEEEEE"),
        legend=dict(orientation="h", y=1.08, x=0),
        margin=dict(l=50, r=20, t=30, b=40), font=dict(size=12))
    os.makedirs(os.path.dirname(png_path) or ".", exist_ok=True)
    fig.write_image(png_path, width=width, height=height)
    return png_path


def _band_hex(value):
    return "#" + _kappa_band(value)[1].hexval()[2:]


def _fmt(value, spec=".2f"):
    v = pd.to_numeric(value, errors="coerce")
    return "N/A" if pd.isna(v) else format(float(v), spec)


def _table_style(extra=None):
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#003366")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F5F8")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    return TableStyle(style + (extra or []))


def _cucei_table(rows, styles):
    """Dimension | Mean | SD | N | What it means (one line)."""
    cell = ParagraphStyle("cucei_cell", parent=styles["Normal"], fontSize=8.5, leading=10.5)
    data = [["Dimension", "Mean (1–4)", "SD", "N", "Interpretation"]]
    for r in rows:
        data.append([r["dimension"], _fmt(r["mean"]), _fmt(r["sd"]),
                     str(r["n"]) if r["n"] is not None else "—",
                     Paragraph(r["interpretation"], cell)])
    t = Table(data, colWidths=[1.35 * inch, 0.8 * inch, 0.5 * inch, 0.4 * inch, 3.65 * inch])
    t.setStyle(_table_style([("ALIGN", (1, 0), (3, -1), "CENTER")]))
    return t


def _validation_table(df):
    """
    Per-code validation from the combined_kappa.csv written by run.py:
    Code | Human windows | AI windows | Agreement | kappa | AC1.
    kappa and AC1 are colored by the same reliability bands.
    """
    rows = df[~df["code"].isin(SUMMARY_LABELS) & (df["code"] != MEAN_ROW_LABEL)]
    has_ac1 = "ai_vs_sofia_ac1" in df.columns
    header = ["Code", "Human", "AI", "Agree", "κ"] + (["AC1"] if has_ac1 else [])
    data = [header]
    style_extra = [("ALIGN", (1, 0), (-1, -1), "CENTER")]
    grey = colors.HexColor("#9E9E9E")
    for i, (_, r) in enumerate(rows.iterrows(), start=1):
        pct = pd.to_numeric(r.get("pct_agreement"), errors="coerce")
        # A behavior the human never recorded has no real agreement to grade:
        # AC1 there is ~0.95 from shared absences alone. Grey, not green.
        observed = (pd.to_numeric(r.get("n_human_marked"), errors="coerce") or 0) > 0
        band = (lambda v: colors.HexColor(_band_hex(v))) if observed else (lambda v: grey)
        line = [r["code"],
                _fmt(r.get("n_human_marked"), ".0f"),
                _fmt(r.get("n_ai_marked"), ".0f"),
                "N/A" if pd.isna(pct) else f"{pct:.0f}%",
                _fmt(r["ai_vs_sofia_kappa"])]
        style_extra.append(("TEXTCOLOR", (4, i), (4, i), band(r["ai_vs_sofia_kappa"])))
        if has_ac1:
            line.append(_fmt(r["ai_vs_sofia_ac1"]))
            style_extra.append(("TEXTCOLOR", (5, i), (5, i), band(r["ai_vs_sofia_ac1"])))
        data.append(line)
    widths = [0.8, 0.8, 0.8, 0.9, 0.9] + ([0.9] if has_ac1 else [])
    t = Table(data, colWidths=[w * inch for w in widths])
    t.setStyle(_table_style(style_extra))
    return t


def _clearing_summary(df):
    """'κ ≥ 0.7 on a of b behaviors ... AC1 ≥ 0.7 on c of b' from the summary rows."""
    s = df.set_index("code")
    parts = []
    for label, col, name in ((KAPPA_CLEAR_LABEL, "ai_vs_sofia_kappa", "κ"),
                             (AC1_CLEAR_LABEL, "ai_vs_sofia_ac1", "AC1")):
        if label in s.index and col in s.columns:
            cleared = _fmt(s.loc[label, col], ".0f")
            denom = _fmt(s.loc[label].get("n_codes_human_observed"), ".0f")
            parts.append(f"{name} ≥ 0.7 on <b>{cleared} of {denom}</b>")
    if not parts:
        return None
    return ("The AI reached the standard COPUS reliability threshold (Smith et al., "
            "2013) with " + " and ".join(parts) + " of the behaviors the human "
            "observer recorded in your class.")


def generate_faculty_report(
    professor_name,
    course_name,
    semester,
    lecture_results,
    kappa_results,
    survey_data,
    output_path,
    tmp_dir=None,
    professor_id="",
    data_dir=None,
    golden_dir=None,
):
    """
    Produce the Faculty Feedback Report PDF and return `output_path`.

    Sections, in order:
      1. Header: professor, course, date, number of lectures analyzed
      2. Behavioral Profile (summary, code frequencies, per-lecture timelines)
      3. Student Perceptions (CUCEI): 7 dimensions, mean / SD / N + one line each
      4. Comparison with each golden reference lecture (one chart per lecture,
         with its instructor's name and YouTube link; never pooled)
      5. Linking what you did to how students experienced it (2-3 observations)
      6. Validation: kappa and Gwet's AC1 per code
      7. Recommendations

    Sections 3-5 show a short "not available yet" note when their input
    (CUCEI scores in data/, golden/golden_lectures.csv) is missing, so the report
    always builds.

    Parameters
    ----------
    professor_name, course_name, semester : str
        Header labels. Blank ones are left out rather than printed empty.
    lecture_results : list[dict]
        One dict per lecture with keys `lecture_id` and `results_csv_path`.
    kappa_results : dict | str
        {comparison: {code: kappa}} or a path to `combined_kappa.csv`.
    survey_data : dict | str | None
        Generic survey summary, used only as context for the narrative text.
    output_path : str
        Where to write the PDF.
    tmp_dir : str | None
        Where to write intermediate chart PNGs (defaults to alongside the PDF).
    professor_id : str
        Looks up this professor's CUCEI scores.
    data_dir : str | None
        Folder holding cucei/ and cucei_scores.csv (default: the repo's data/).
    golden_dir : str | None
        Folder holding golden_lectures.csv and one results folder per reference
        lecture (default: the repo's golden/).
    """
    data_dir = data_dir or DEFAULT_DATA_DIR
    styles = _styles()
    tmp_dir = tmp_dir or os.path.join(os.path.dirname(output_path) or ".",
                                      "_report_assets")
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    survey_summary, _ = _survey_summary_text(survey_data)

    # Load every lecture's results once; skip missing/empty CSVs gracefully.
    lectures = []
    for lr in lecture_results:
        csv_path = lr.get("results_csv_path")
        if not csv_path or not os.path.exists(csv_path):
            print(f"[pdf_report] skipping {lr.get('lecture_id')}: missing {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        freq = code_frequency(df)
        a_pct, p_pct = active_learning_split(df)
        lectures.append({
            "lecture_id": lr.get("lecture_id", "lecture"),
            "csv_path": csv_path,
            "df": df, "freq": freq,
            "active_pct": a_pct, "passive_pct": p_pct,
        })

    if lectures:
        all_df = pd.concat([l["df"] for l in lectures], ignore_index=True)
    else:
        all_df = pd.DataFrame(columns=["copus_codes"])
    overall_freq = code_frequency(all_df)
    overall_active, overall_passive = active_learning_split(all_df)
    shares = code_shares(all_df)

    cucei_rows = cucei_profile(professor_id, data_dir)
    goldens = golden_lectures(golden_dir or DEFAULT_GOLDEN_DIR)

    # CUCEI scores give the narrative/recommendation prompts real student context.
    if cucei_rows:
        cucei_text = "\n".join(cucei_context_lines(cucei_rows))
        survey_summary = "\n".join(x for x in (survey_summary, cucei_text) if x)
    # recommendations() gets the CUCEI rows directly; keep them out of its
    # generic survey text so they are not listed twice in the prompt.
    rec_survey_summary, _ = _survey_summary_text(survey_data)

    n_lect = len(lectures)
    who = professor_name or "this instructor"
    story = []

    # ---- 1. Header --------------------------------------------------------- #
    story.append(Spacer(1, 1.4 * inch))
    story.append(Paragraph("Faculty Feedback Report", styles["CoverTitle"]))
    for line in (professor_name, course_name, semester):
        if line:
            story.append(Paragraph(line, styles["CoverMeta"]))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(
        f"{n_lect} lecture{'s' if n_lect != 1 else ''} analyzed · "
        f"generated {date.today().strftime('%B %d, %Y').replace(' 0', ' ')}",
        styles["CoverMeta"]))
    story.append(Paragraph(
        "University of South Florida — COPUS Classroom Analytics",
        styles["CoverMeta"]))
    story.append(Paragraph(
        "Generated by the Bridging the Feedback Gap AI Tool. "
        "For pedagogical development only.", styles["Disclaimer"]))
    story.append(PageBreak())

    # ---- 2. Behavioral Profile --------------------------------------------- #
    story.append(Paragraph("Behavioral Profile", styles["SectionH"]))
    if not overall_freq.empty:
        most = overall_freq.iloc[0]["code"]
        least = overall_freq.iloc[-1]["code"]
    else:
        most = least = "—"
    context = ", ".join(x for x in (course_name, semester) if x)
    story.append(Paragraph(
        f"This report summarizes an AI-assisted COPUS analysis of {n_lect} "
        f"lecture{'s' if n_lect != 1 else ''} for {who}"
        f"{f' ({context})' if context else ''}. Each lecture was divided into "
        f"2-minute segments and classified against the 12 COPUS instructor codes. "
        f"Across all lectures, roughly {overall_active}% of classified instructor "
        f"time reflected active-learning behaviors and {overall_passive}% reflected "
        f"lecturing or passive instruction. The most frequently observed code was "
        f"<b>{most}</b> and the least frequent was <b>{least}</b>.",
        styles["Narrative"]))

    if not overall_freq.empty:
        split_png = os.path.join(tmp_dir, "summary_split.png")
        render_active_learning_png(overall_active, overall_passive, split_png)
        story.append(Spacer(1, 0.1 * inch))
        story.append(Image(split_png, width=3.0 * inch, height=2.5 * inch))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("Overall code frequency (all lectures)", styles["Narrative"]))
    story.append(_freq_table(overall_freq, styles))
    story.append(PageBreak())

    for lect in lectures:
        story.append(Paragraph(
            f"Lecture: {lect['lecture_id']}", styles["SectionH"]))
        png = os.path.join(tmp_dir, f"timeline_{lect['lecture_id']}.png")
        try:
            render_timeline_png(lect["csv_path"], png)
            story.append(Image(png, width=6.5 * inch, height=2.9 * inch))
        except Exception as e:  # noqa: BLE001
            print(f"[pdf_report] timeline render failed for "
                  f"{lect['lecture_id']}: {e}")
            story.append(Paragraph("(timeline chart unavailable)", styles["Narrative"]))
        story.append(Spacer(1, 0.1 * inch))
        story.append(_freq_table(lect["freq"], styles))
        story.append(Spacer(1, 0.1 * inch))
        narrative = lecture_narrative(
            lect["csv_path"], lect["freq"],
            lect["active_pct"], lect["passive_pct"], survey_summary)
        story.append(Paragraph(narrative, styles["Narrative"]))
        story.append(PageBreak())

    # ---- 3. Student Perceptions (CUCEI) ------------------------------------ #
    story.append(Paragraph("Student Perceptions (CUCEI)", styles["SectionH"]))
    if cucei_rows:
        n_max = max((r["n"] or 0) for r in cucei_rows)
        story.append(Paragraph(
            f"Your students completed the College and University Classroom Environment "
            f"Inventory (CUCEI; up to {n_max} respondents). Each dimension is scored "
            f"from 1 (strongly disagree) to 4 (strongly agree); higher is more "
            f"favorable. Each dimension counts only students who answered all of "
            f"its items.", styles["Narrative"]))
        story.append(_cucei_table(cucei_rows, styles))
    else:
        story.append(Paragraph(
            "<i>Student survey (CUCEI) results for this course have not been added "
            "yet. This section will show your seven classroom-environment scores "
            "once they are available.</i>", styles["Narrative"]))
    story.append(PageBreak())

    # ---- 4. Golden reference comparisons ----------------------------------- #
    # One comparison per reference lecture: each is by a different instructor,
    # so they are never pooled into a single "golden" profile.
    story.append(Paragraph("Comparison with Reference Lectures", styles["SectionH"]))
    if goldens and shares:
        n_g = len(goldens)
        story.append(Paragraph(
            f"Below, your behavioral profile is set beside {n_g} reference "
            f"lecture{'s' if n_g != 1 else ''}, each by a different instructor and "
            f"chosen as an example of active-learning practice. Each is shown on its "
            f"own, with a link so you can watch it. Bars show the share of analyzed "
            f"2-minute segments in which each behavior appeared; a segment can show "
            f"several behaviors. These are reference points, not targets.",
            styles["Narrative"]))
        for n, g in enumerate(goldens, start=1):
            who = g["professor_name"] or f"Reference lecture {n}"
            heading = f"Reference lecture {n}: {xml_escape(who)}"
            if g["lecture_title"]:
                heading += f" — <i>{xml_escape(g['lecture_title'])}</i>"
            block = [Paragraph(heading, styles["SubH"])]
            if g["youtube_url"]:
                url = xml_escape(g["youtube_url"], {'"': "&quot;"})
                block.append(Paragraph(
                    f'Watch the lecture: <link href="{url}" color="blue">'
                    f'<u>{url}</u></link> ({g["n_windows"]} segments analyzed)',
                    styles["Narrative"]))
            png = os.path.join(tmp_dir, f"golden_comparison_{n}.png")
            try:
                render_golden_png(shares, g["shares"], png, golden_label=who)
                block.append(Image(png, width=6.3 * inch, height=3.15 * inch))
            except Exception as e:  # noqa: BLE001
                print(f"[pdf_report] golden chart {n} failed: {e}")
            g_active, _ = active_learning_split(g["results"])
            diffs = sorted(((g["shares"].get(c, 0) - shares.get(c, 0), c)
                            for c in ACTIVE_LEARNING_CODES), reverse=True)
            gap_pct, gap_code = diffs[0]
            gap_line = (f" The largest difference is <b>{gap_code}</b> "
                        f"({CODE_MEANINGS[gap_code].lower()}): "
                        f"{g['shares'].get(gap_code, 0):g}% of this lecture's segments "
                        f"versus {shares.get(gap_code, 0):g}% of yours."
                        if gap_pct >= 10 else "")
            block.append(Paragraph(
                f"Active-learning behaviors made up <b>{overall_active}%</b> of coded "
                f"instructor behavior in your lectures and <b>{g_active}%</b> in this "
                f"one (the same measure as the chart on the Behavioral Profile "
                f"page).{gap_line}", styles["Narrative"]))
            story.append(KeepTogether(block))
    else:
        story.append(Paragraph(
            "<i>The reference (golden) lectures have not been added yet. This "
            "section will compare your behavioral profile with each of them once "
            "they are available.</i>", styles["Narrative"]))
    story.append(PageBreak())

    # ---- 5. Behavior-perception linking ------------------------------------ #
    story.append(Paragraph("What You Did and How Students Experienced It",
                           styles["SectionH"]))
    observations = linking_observations(shares, cucei_rows)
    if observations:
        for obs in observations:
            story.append(Paragraph(f"• {obs}", styles["Narrative"]))
        story.append(Paragraph(
            "<i>These pairings are observations, not measured effects: the survey "
            "asks about the course as a whole, and the behaviors come from a sample "
            "of class time.</i>", styles["Disclaimer"]))
    else:
        story.append(Paragraph(
            "<i>Available once student survey (CUCEI) results are added.</i>",
            styles["Narrative"]))
    story.append(Spacer(1, 0.3 * inch))

    # ---- 6. Validation ------------------------------------------------------ #
    kappa_df = (pd.read_csv(kappa_results)
                if isinstance(kappa_results, str) and os.path.exists(kappa_results)
                else None)
    validation = [Paragraph("How Reliable Is This Analysis?", styles["SectionH"])]
    if kappa_df is not None and "ai_vs_sofia_kappa" in kappa_df.columns:
        validation.append(Paragraph(
            "Part of each lecture was also coded by a trained human observer. The "
            "table compares the AI with that observer, per behavior: how many "
            "segments each marked, how often they agreed, Cohen's κ, and Gwet's AC1. "
            "AC1 stays meaningful when a behavior fills almost every segment (such as "
            "lecturing), where κ can fall to near zero despite near-perfect agreement. "
            "N/A means neither observer recorded that behavior; grey values are "
            "behaviors only the AI recorded.", styles["Narrative"]))
        validation.append(_validation_table(kappa_df))
        summary = _clearing_summary(kappa_df)
        if summary:
            validation.append(Paragraph(summary, styles["Narrative"]))
        validation.append(_legend_flowable(styles))
    elif kappa_results:
        kappa_dict = load_kappa_dict(kappa_results)
        validation.append(Paragraph(
            "Cohen's κ between the AI classifier and the human coder, per COPUS code, "
            "pooled across lectures.", styles["Narrative"]))
        validation.append(_kappa_table(kappa_dict))
        validation.append(Spacer(1, 0.15 * inch))
        validation.append(_legend_flowable(styles))
    else:
        validation.append(Paragraph(
            "No validation data was provided for this report.", styles["Narrative"]))
    story.append(KeepTogether(validation))
    story.append(PageBreak())

    # ---- 7. Recommendations ------------------------------------------------- #
    story.append(Paragraph("Recommendations", styles["SectionH"]))
    story.append(Paragraph(
        "The following suggestions are framed as opportunities, not "
        "evaluations, consistent with the COPUS philosophy of describing "
        "(rather than judging) classroom practice.", styles["Narrative"]))
    for rec in recommendations(overall_freq, overall_active, rec_survey_summary,
                               shares=shares, cucei_rows=cucei_rows):
        story.append(Paragraph(f"• {rec}", styles["Narrative"]))

    doc = SimpleDocTemplate(
        output_path, pagesize=letter,
        leftMargin=0.9 * inch, rightMargin=0.9 * inch,
        topMargin=0.9 * inch, bottomMargin=0.9 * inch,
        title="Faculty Feedback Report", author="Bridging the Feedback Gap")
    doc.build(story)
    print(f"Faculty report saved → {output_path}")
    return output_path


# --------------------------------------------------------------------------- #
# CLI — build a report straight from a run.py output directory
# --------------------------------------------------------------------------- #
def _discover_lectures(output_dir):
    """
    Find per-lecture results CSVs under a run.py output dir. A lecture dir is any
    subdir containing a results CSV (prefers results_multimodal.csv).
    """
    lectures = []
    for name in sorted(os.listdir(output_dir)):
        ldir = os.path.join(output_dir, name)
        if not os.path.isdir(ldir):
            continue
        for candidate in ("results_multimodal.csv", "results.csv"):
            csv_path = os.path.join(ldir, candidate)
            if os.path.exists(csv_path):
                lectures.append({
                    "lecture_id": name,
                    "results_csv_path": csv_path,
                    "dashboard_html_path": os.path.join(ldir, "dashboard.html"),
                })
                break
    return lectures


def main():
    parser = argparse.ArgumentParser(
        description="Phase 9 — build a Faculty Feedback Report PDF from a "
                    "run.py output directory.")
    parser.add_argument("--output-dir", required=True,
                        help="run.py output dir, e.g. output/dr_smith")
    parser.add_argument("--professor", default="",
                        help="Default: professor_name from lecture_professor_mapping.csv")
    parser.add_argument("--course", default="",
                        help="Default: course_name from lecture_professor_mapping.csv")
    parser.add_argument("--data-dir", default=None,
                        help="Folder with cucei/ and cucei_scores.csv (default: repo data/)")
    parser.add_argument("--semester", default="")
    parser.add_argument("--pdf", default=None,
                        help="Output PDF path (default: <output-dir>/faculty_report.pdf)")
    args = parser.parse_args()

    lectures = _discover_lectures(args.output_dir)
    if not lectures:
        parser.error(f"no lecture results CSVs found under {args.output_dir}")

    kappa_csv = os.path.join(args.output_dir, "combined_kappa.csv")
    kappa_arg = kappa_csv if os.path.exists(kappa_csv) else None
    survey_csv = os.path.join(args.output_dir, "survey_analysis.csv")
    survey_arg = survey_csv if os.path.exists(survey_csv) else None
    pdf_path = args.pdf or os.path.join(args.output_dir, "faculty_report.pdf")

    from report.feedback_sections import resolve_identity
    professor_id, professor_name, course_name = resolve_identity(
        args.output_dir, {"professor": args.professor, "course": args.course})
    generate_faculty_report(
        professor_name=professor_name,
        course_name=course_name,
        professor_id=professor_id,
        data_dir=args.data_dir,
        semester=args.semester,
        lecture_results=lectures,
        kappa_results=kappa_arg,
        survey_data=survey_arg,
        output_path=pdf_path,
    )


if __name__ == "__main__":
    main()
