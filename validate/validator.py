import math
import pandas as pd
import os
from sklearn.metrics import cohen_kappa_score

# Label for the summary row appended to every kappa table.
MEAN_ROW_LABEL = "MEAN"


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
    Expected columns: lecture_id, window_index, window_start, window_end, arm,
    model, copus_codes
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


def mean_kappa(values):
    """
    Mean of the numeric kappa values, skipping "N/A" and "ERR:" entries.

    Codes that neither rater used contribute no information and must not drag
    the average toward zero -- that is the whole point of reporting them as N/A
    rather than 0.00.
    """
    nums = []
    for v in values:
        f = pd.to_numeric(v, errors="coerce")
        if pd.notna(f):
            nums.append(float(f))
    if not nums:
        return "N/A"
    return round(sum(nums) / len(nums), 3)


def kappa_by_code(human_windows, ai_windows):
    """
    Computes Cohen's Kappa per COPUS code given already-loaded window lists
    (each a list of {"window_index", "codes"} dicts). Only compares windows
    present in BOTH sets. Returns a list of dicts:
      {"code", "kappa", "human_positive_windows", "ai_positive_windows"}
    sorted by code.

    Kappa is "N/A" only when it is mathematically undefined rather than low:
      - NEITHER rater ever marked the code (no observations at all), or
      - neither rater varies on the code (e.g. both marked every window).

    A code used by exactly one rater is NOT N/A. cohen_kappa_score returns
    exactly 0.0 there, and that zero is the correct reading: the two raters
    agreed on nothing for that code. Reporting it as N/A hides one-sided
    over-fires (Sofia n=0 vs AI n=11 on AnQ) behind a blank cell.

    Window lists may pool multiple lectures; window_index collisions across
    lectures are handled by pairing on position within the shared-index order
    per (human, ai) pair --- callers that pool should pass matching-length,
    aligned lists (see compute_comparison_table).
    """
    human_indices = [w["window_index"] for w in human_windows]
    ai_indices = set(w["window_index"] for w in ai_windows)
    # Pair human rows to AI rows on shared window_index, preserving human order.
    ai_by_index = {}
    for w in ai_windows:
        ai_by_index.setdefault(w["window_index"], []).append(w)

    all_codes = set()
    for w in human_windows:
        all_codes.update(w["codes"])
    for w in ai_windows:
        all_codes.update(w["codes"])
    all_codes.discard("")

    # Build aligned (human, ai) row pairs over shared indices.
    pairs = []
    ai_cursor = {}
    for w in human_windows:
        idx = w["window_index"]
        if idx not in ai_by_index:
            continue
        bucket = ai_by_index[idx]
        pos = ai_cursor.get(idx, 0)
        if pos >= len(bucket):
            continue
        pairs.append((w, bucket[pos]))
        ai_cursor[idx] = pos + 1

    n = len(pairs)
    results = []
    for code in sorted(all_codes):
        human_labels = [1 if code in h["codes"] else 0 for h, _ in pairs]
        ai_labels = [1 if code in a["codes"] else 0 for _, a in pairs]
        human_pos = sum(human_labels)
        ai_pos = sum(ai_labels)

        if n == 0:
            kappa = "N/A"
        elif human_pos == 0 and ai_pos == 0:
            # NEITHER rater used this code -- there are no observations at all,
            # so kappa is genuinely undefined. When only ONE rater is at zero the
            # code IS defined: sklearn returns 0.0, and that 0.0 is a real
            # finding (one rater over- or under-fired the code on every window),
            # not an artifact. Suppressing it hid an 11-window AnQ over-fire.
            kappa = "N/A"
        elif len(set(human_labels + ai_labels)) < 2:
            # No variation in either rater.
            kappa = "N/A"
        else:
            if human_pos == n or ai_pos == n:
                # Mirror image of the case above: a rater who marked EVERY
                # window is also constant, so kappa is depressed by prevalence
                # rather than by disagreement. Reported, but flagged.
                print(f"[warn] {code}: a rater marked all {n} windows "
                      f"(human={human_pos}, ai={ai_pos}); kappa is prevalence-"
                      f"depressed, not a disagreement measure.")
            try:
                kappa = cohen_kappa_score(human_labels, ai_labels)
                # sklearn returns nan when the (1 - pe) denominator is 0.
                kappa = "N/A" if math.isnan(kappa) else round(kappa, 3)
            except Exception as e:
                kappa = f"ERR: {e}"

        results.append({
            "code": code,
            "kappa": kappa,
            "human_positive_windows": human_pos,
            "ai_positive_windows": ai_pos,
        })
    return results


def compute_kappa(csv_a, csv_b, output_dir, output_filename="kappa_results.csv",
                  label_a="A", label_b="B"):
    """
    Computes Cohen's Kappa for each COPUS code comparing two long-format code
    sets. Works for AI-vs-human OR human-vs-human (a second coder, for
    inter-rater reliability): both inputs are read the same way, so any two
    coding CSVs can be compared.
    Only compares windows present in BOTH csv_a and csv_b.

    output_filename lets callers name each comparison distinctly
    (e.g. kappa_ai_vs_sofia.csv). label_a/label_b customize the printed header.
    Returns the per-code kappa DataFrame, with a trailing MEAN row averaging the
    numeric kappas only. The 'human_positive_windows' / 'ai_positive_windows'
    columns count positives for csv_a / csv_b respectively.
    """
    os.makedirs(output_dir, exist_ok=True)

    a_windows = load_human_codes(csv_a)
    b_windows = load_human_codes(csv_b)

    a_indices = set(w["window_index"] for w in a_windows)
    b_indices = set(w["window_index"] for w in b_windows)
    print(f"{label_a} windows: {len(a_indices)}")
    print(f"{label_b} windows: {len(b_indices)}")
    print(f"Shared windows for comparison: {sorted(a_indices & b_indices)}")

    kappa_results = kappa_by_code(a_windows, b_windows)
    kappa_results.append({
        "code": MEAN_ROW_LABEL,
        "kappa": mean_kappa([r["kappa"] for r in kappa_results]),
        "human_positive_windows": "",
        "ai_positive_windows": "",
    })

    print(f"\n{'Code':<8} {'Kappa':>8} {label_a + '+':>8} {label_b + '+':>8}")
    print("-" * 38)
    for r in kappa_results:
        if r["code"] == MEAN_ROW_LABEL:
            print("-" * 38)
        print(f"{r['code']:<8} {str(r['kappa']):>8} "
              f"{str(r['human_positive_windows']):>8} "
              f"{str(r['ai_positive_windows']):>8}")
    print("(N/A = kappa undefined: neither rater ever used that code)")

    # Save results
    results_df = pd.DataFrame(kappa_results)
    out_path = os.path.join(output_dir, output_filename)
    results_df.to_csv(out_path, index=False)
    print(f"\nKappa results saved -> {out_path}")

    return results_df


ARM_ORDER = ["multimodal", "vision_only", "audio_only", "transcript_only"]


def _normalize_arm_csvs(arm_csvs):
    """
    Accepts a single {arm: csv_path} dict (one lecture) or a list of such dicts
    (multiple lectures). Returns a list of per-lecture dicts.
    """
    if isinstance(arm_csvs, dict):
        return [arm_csvs]
    return list(arm_csvs)


def compute_comparison_table(arm_csvs, human_csv, output_csv, per_lecture=False):
    """
    Builds the cross-arm Cohen's Kappa comparison table.

    arm_csvs:
      - {arm_name: results_csv_path} for a single lecture, OR
      - [ {arm_name: results_csv_path}, ... ] one dict per lecture (pooled).
    human_csv:
      - path to the single lecture's human CSV, OR
      - list of human CSV paths aligned with the arm_csvs list.
    output_csv: path for the pooled wide table.
    per_lecture: also write a per-lecture breakdown next to output_csv.

    Raw kappa is computed for every (code x arm) --- no support-matrix masking.
    A trailing MEAN row averages each arm's numeric kappas.
    """
    lecture_arm_csvs = _normalize_arm_csvs(arm_csvs)
    human_csvs = human_csv if isinstance(human_csv, (list, tuple)) else [human_csv]
    if len(human_csvs) != len(lecture_arm_csvs):
        raise ValueError(
            f"human_csv count ({len(human_csvs)}) must match lecture count "
            f"({len(lecture_arm_csvs)})"
        )

    arms = [a for a in ARM_ORDER if any(a in d for d in lecture_arm_csvs)]

    def build_table(lecture_dicts, human_paths):
        """Pool the given lectures and return {code: {arm: kappa}}, all_codes."""
        pooled_human = []
        for hp in human_paths:
            pooled_human.extend(load_human_codes(hp))
        table = {}
        for arm in arms:
            pooled_ai = []
            for d, hp in zip(lecture_dicts, human_paths):
                if arm not in d:
                    continue
                pooled_ai.extend(load_ai_codes(d[arm]))
            # Re-pool human to align lectures that actually have this arm.
            arm_human = []
            for d, hp in zip(lecture_dicts, human_paths):
                if arm in d:
                    arm_human.extend(load_human_codes(hp))
            for r in kappa_by_code(arm_human, pooled_ai):
                table.setdefault(r["code"], {})[arm] = r["kappa"]
        return table

    def write_wide(table, path, extra=None):
        codes = sorted(table)
        rows = []
        for code in codes:
            row = {"code": code}
            if extra:
                row.update(extra)
            for arm in arms:
                row[f"{arm}_kappa"] = table[code].get(arm, "N/A")
            rows.append(row)
        # MEAN row, per arm, over that arm's numeric kappas only.
        mean_row = {"code": MEAN_ROW_LABEL}
        if extra:
            mean_row.update(extra)
        for arm in arms:
            mean_row[f"{arm}_kappa"] = mean_kappa(
                [table[c].get(arm, "N/A") for c in codes]
            )
        rows.append(mean_row)
        cols = (["code"] + (list(extra.keys()) if extra else [])
                + [f"{arm}_kappa" for arm in arms])
        df = pd.DataFrame(rows, columns=cols)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        df.to_csv(path, index=False)
        return df

    # Pooled table (main output)
    pooled = build_table(lecture_arm_csvs, human_csvs)
    pooled_df = write_wide(pooled, output_csv)

    print(f"\n=== Comparison table (pooled, kappa per code x arm) ===")
    header = f"{'Code':<8}" + "".join(f"{a:>18}" for a in arms)
    print(header)
    print("-" * len(header))
    for code in sorted(pooled):
        line = f"{code:<8}"
        for arm in arms:
            line += f"{str(pooled[code].get(arm, 'N/A')):>18}"
        print(line)
    print("-" * len(header))
    mean_line = f"{MEAN_ROW_LABEL:<8}"
    for arm in arms:
        mean_line += f"{str(mean_kappa([pooled[c].get(arm, 'N/A') for c in sorted(pooled)])):>18}"
    print(mean_line)
    print("(N/A = kappa undefined: neither rater used that code; excluded from MEAN)")
    print(f"\nComparison table saved -> {output_csv}")

    # Per-lecture breakdown
    if per_lecture and len(lecture_arm_csvs) > 1:
        by_lecture_rows = []
        for d, hp in zip(lecture_arm_csvs, human_csvs):
            lid = _lecture_id_from_human_csv(hp)
            table = build_table([d], [hp])
            codes = sorted(table)
            for code in codes:
                row = {"code": code, "lecture_id": lid}
                for arm in arms:
                    row[f"{arm}_kappa"] = table[code].get(arm, "N/A")
                by_lecture_rows.append(row)
            mean_row = {"code": MEAN_ROW_LABEL, "lecture_id": lid}
            for arm in arms:
                mean_row[f"{arm}_kappa"] = mean_kappa(
                    [table[c].get(arm, "N/A") for c in codes]
                )
            by_lecture_rows.append(mean_row)
        cols = ["lecture_id", "code"] + [f"{arm}_kappa" for arm in arms]
        by_df = pd.DataFrame(by_lecture_rows, columns=cols)
        by_path = os.path.join(
            os.path.dirname(output_csv) or ".", "comparison_table_by_lecture.csv"
        )
        by_df.to_csv(by_path, index=False)
        print(f"Per-lecture breakdown saved -> {by_path}")

    return pooled_df


def _lecture_id_from_human_csv(human_csv):
    """Best-effort lecture id from a human_coding_{id}.csv path (else stem)."""
    base = os.path.basename(human_csv)
    stem = os.path.splitext(base)[0]
    prefix = "human_coding_"
    return stem[len(prefix):] if stem.startswith(prefix) else stem
