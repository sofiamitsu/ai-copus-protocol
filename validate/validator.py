import pandas as pd
import os
from sklearn.metrics import cohen_kappa_score


def load_human_codes(csv_path):
    """
    Loads long-format human coding CSV.
    Expected columns: lecture_id, window_index, window_start, window_end, copus_codes
    """
    df = pd.read_csv(csv_path)
    rows = []
    for _, row in df.iterrows():
        codes = row["copus_codes"].split("|") if pd.notna(row["copus_codes"]) and row["copus_codes"] != "" else []
        rows.append({
            "window_index": int(row["window_index"]),
            "codes": codes
        })
    return rows


def load_ai_codes(csv_path):
    """
    Loads results.csv from the pipeline.
    Expected columns: lecture_id, window_index, window_start, window_end, arm, copus_codes
    """
    df = pd.read_csv(csv_path)
    rows = []
    for _, row in df.iterrows():
        codes = row["copus_codes"].split("|") if pd.notna(row["copus_codes"]) and row["copus_codes"] != "" else []
        rows.append({
            "window_index": int(row["window_index"]),
            "codes": codes
        })
    return rows


def compute_kappa(human_csv, ai_csv, output_dir):
    """
    Computes Cohen's Kappa for each COPUS code comparing
    human ground truth vs AI classifications.
    Only compares windows present in BOTH human and AI.
    """
    os.makedirs(output_dir, exist_ok=True)

    human_windows = load_human_codes(human_csv)
    ai_windows = load_ai_codes(ai_csv)

    # Only compare windows that exist in both
    human_indices = set(w["window_index"] for w in human_windows)
    ai_indices = set(w["window_index"] for w in ai_windows)
    shared_indices = sorted(human_indices & ai_indices)

    print(f"Human windows: {len(human_indices)}")
    print(f"AI windows: {len(ai_indices)}")
    print(f"Shared windows for comparison: {shared_indices}")

    # Get all codes that appear in either human or AI
    all_codes = set()
    for w in human_windows:
        all_codes.update(w["codes"])
    for w in ai_windows:
        all_codes.update(w["codes"])
    all_codes.discard("")

    print(f"\n{'Code':<8} {'Kappa':>8} {'Human+':>8} {'AI+':>8}")
    print("-" * 38)

    kappa_results = []

    for code in sorted(all_codes):
        human_labels = []
        ai_labels = []

        for idx in shared_indices:
            h = next(w for w in human_windows if w["window_index"] == idx)
            a = next(w for w in ai_windows if w["window_index"] == idx)
            human_labels.append(1 if code in h["codes"] else 0)
            ai_labels.append(1 if code in a["codes"] else 0)

        human_pos = sum(human_labels)
        ai_pos = sum(ai_labels)

        # Need variation in at least one rater to compute kappa
        if len(set(human_labels + ai_labels)) < 2:
            kappa = "N/A"
        else:
            try:
                kappa = round(cohen_kappa_score(human_labels, ai_labels), 3)
            except Exception as e:
                kappa = f"ERR: {e}"

        print(f"{code:<8} {str(kappa):>8} {human_pos:>8} {ai_pos:>8}")
        kappa_results.append({
            "code": code,
            "kappa": kappa,
            "human_positive_windows": human_pos,
            "ai_positive_windows": ai_pos
        })

    # Save results
    results_df = pd.DataFrame(kappa_results)
    out_path = os.path.join(output_dir, "kappa_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nKappa results saved → {out_path}")

    return results_df