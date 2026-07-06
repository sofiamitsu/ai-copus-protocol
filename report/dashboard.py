import pandas as pd
import plotly.graph_objects as go
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

def generate_dashboard(results_csv, output_html):
    """
    Reads results.csv and generates an interactive HTML timeline
    showing COPUS codes across all 2-minute windows.
    """
    df = pd.read_csv(results_csv)

    fig = go.Figure()

    # For each window, draw a bar for each COPUS code present
    for _, row in df.iterrows():
        codes = row["copus_codes"].split("|") if pd.notna(row["copus_codes"]) else []
        start = row["window_start"]
        end = row["window_end"]
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
                hovertemplate=(
                    f"<b>{code}</b><br>"
                    f"Window: {start}s → {end}s<br>"
                    f"Reasoning: {row.get('reasoning', '')[:200]}"
                    "<extra></extra>"
                ),
                showlegend=code not in [t.name for t in fig.data]
            ))

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