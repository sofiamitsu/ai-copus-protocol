import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# COPUS code colors — each code gets a distinct color
CODE_COLORS = {
    "Lec": "#4C72B0",
    "RtW": "#DD8452",
    "FUp": "#55A868",
    "PQ":  "#C44E52",
    "AnQ": "#8172B2",
    "MG":  "#937860",
    "D/V": "#DA8BC3",
    "L":   "#8C8C8C",
    "Ind": "#CCB974",
    "CG":  "#64B5CD",
    "WG":  "#2CA02C",
    "SQ":  "#FF7F0E",
}

def add_timeline_traces(fig, df, row=None, col=None, seen_codes=None):
    """
    Adds one horizontal timeline bar per (window, code) from a results DataFrame
    to `fig`. When row/col are given, traces go to that subplot cell; otherwise
    they go to the single-plot figure. `seen_codes` (a set) dedupes legend
    entries across calls — pass a shared set when stacking multiple subplots.
    """
    if seen_codes is None:
        seen_codes = set()
    add_kwargs = {}
    if row is not None:
        add_kwargs = {"row": row, "col": col or 1}

    for _, r in df.iterrows():
        codes = r["copus_codes"].split("|") if pd.notna(r["copus_codes"]) else []
        start = r["window_start"]
        end = r["window_end"]
        duration = end - start

        for code in codes:
            color = CODE_COLORS.get(code, "#999999")
            fig.add_trace(go.Bar(
                x=[duration],
                y=[code],
                base=start,
                orientation="h",
                marker_color=color,
                name=code,
                legendgroup=code,
                hovertemplate=(
                    f"<b>{code}</b><br>"
                    f"Window: {start}s → {end}s<br>"
                    f"Reasoning: {str(r.get('reasoning', ''))[:200]}"
                    "<extra></extra>"
                ),
                showlegend=code not in seen_codes,
            ), **add_kwargs)
            seen_codes.add(code)


def generate_dashboard(results_csv, output_html):
    """
    Reads results.csv and generates an interactive HTML timeline
    showing COPUS codes across all 2-minute windows.
    """
    df = pd.read_csv(results_csv)

    fig = go.Figure()
    add_timeline_traces(fig, df)

    fig.update_layout(
        title=f"COPUS Behavioral Timeline — {df['lecture_id'].iloc[0]}",
        xaxis_title="Time (seconds)",
        yaxis_title="COPUS Code",
        barmode="overlay",
        height=500,
        xaxis=dict(showgrid=True),
        legend_title="COPUS Codes",
        plot_bgcolor="white"
    )

    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    fig.write_html(output_html)
    print(f"Dashboard saved → {output_html}")


ARM_ORDER = ["multimodal", "vision_only", "audio_only", "transcript_only"]

# Distinct colors for arms in the kappa bar chart
ARM_COLORS = {
    "multimodal": "#4C72B0",
    "vision_only": "#DD8452",
    "audio_only": "#55A868",
    "transcript_only": "#C44E52",
}


def generate_comparison_dashboard(arm_csvs, comparison_table_csv, output_html):
    """
    Builds a stacked comparison dashboard:
      - one behavioral-timeline subplot per arm (rows 1..N), and
      - a grouped bar chart of Cohen's κ per code per arm (final row).

    arm_csvs: {arm_name: results_csv_path} for ONE lecture (timelines are
              per-lecture). comparison_table_csv: wide table from
              validate.validator.compute_comparison_table (may be pooled).
    """
    arms = [a for a in ARM_ORDER if a in arm_csvs]
    n_rows = len(arms) + 1
    titles = [f"Arm: {a}" for a in arms] + ["Cohen's κ per code per arm"]

    fig = make_subplots(
        rows=n_rows, cols=1,
        subplot_titles=titles,
        vertical_spacing=0.06,
        row_heights=[1.0] * len(arms) + [1.4],
    )

    seen_codes = set()
    for i, arm in enumerate(arms, start=1):
        df = pd.read_csv(arm_csvs[arm])
        add_timeline_traces(fig, df, row=i, col=1, seen_codes=seen_codes)

    # Final row: grouped κ bar chart from the comparison table.
    table = pd.read_csv(comparison_table_csv)
    kappa_row = n_rows
    for arm in arms:
        col = f"{arm}_kappa"
        if col not in table.columns:
            continue
        codes, vals = [], []
        for _, r in table.iterrows():
            v = pd.to_numeric(r[col], errors="coerce")
            if pd.notna(v):
                codes.append(r["code"])
                vals.append(v)
        fig.add_trace(go.Bar(
            x=codes, y=vals, name=arm,
            marker_color=ARM_COLORS.get(arm, "#999999"),
            legendgroup=arm, showlegend=True,
            hovertemplate=f"<b>{arm}</b><br>%{{x}}: κ=%{{y:.3f}}<extra></extra>",
        ), row=kappa_row, col=1)

    fig.update_layout(
        title="COPUS Ablation Comparison — modality arms vs human coding",
        barmode="group",
        height=280 * n_rows,
        plot_bgcolor="white",
        legend_title="COPUS Codes / Arms",
    )
    for i in range(1, len(arms) + 1):
        fig.update_xaxes(title_text="Time (seconds)", row=i, col=1)
        fig.update_yaxes(title_text="Code", row=i, col=1)
    fig.update_xaxes(title_text="COPUS Code", row=kappa_row, col=1)
    fig.update_yaxes(title_text="Cohen's κ", row=kappa_row, col=1)

    os.makedirs(os.path.dirname(output_html) or ".", exist_ok=True)
    fig.write_html(output_html)
    print(f"Comparison dashboard saved → {output_html}")