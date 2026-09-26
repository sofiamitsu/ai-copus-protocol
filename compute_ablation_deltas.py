"""
Phase 6b — ablation deltas (thesis Table 6.2).

Reads a `comparison_table.csv` written by validate.validator.compute_comparison_table
and reports, per COPUS code, how much agreement CHANGES when the multimodal arm is
ablated down to a single modality:

    delta_kappa_<arm> = kappa_<arm> - kappa_multimodal
    delta_ac1_<arm>   = ac1_<arm>   - ac1_multimodal

SIGN CONVENTION -- read it as "what ablating does to agreement":

    negative  ablating HURT: the removed modality carried signal for that code
    positive  ablating HELPED: that modality was noise for that code
    0         the modality made no difference

The brief specified "delta = kappa_multimodal - kappa_arm" alongside "negative =
ablating hurt", which are not compatible: with multimodal first, an arm that does
worse yields a POSITIVE number. The interpretation was kept (it is what the thesis
prose says) and the subtraction order flipped to match it. Pass
--sign multimodal-minus-arm for the other order.

Deltas are only defined where BOTH arms have a numeric coefficient. A code where
either side is N/A (neither rater used it, or no variation) is written as N/A
rather than being silently treated as 0 -- an unobserved code is not evidence that
a modality did not matter.

Codes the HUMAN never marked are N/A too, matching the threshold-count rule in the
comparison table. Their kappa is 0.00 on both arms (built from false positives
alone), which would otherwise print as a delta of exactly 0.0 and read as "this
modality made no difference" for a code that was never observed at all.

Each row carries the prevalence context needed to read the delta: n_total_windows,
n_human_marked, and the per-arm n_ai_marked counts. A large delta on a code with
n_human_marked = 1 is not a modality effect, it is one window.

Usage:
    uv run python compute_ablation_deltas.py \
      --comparison-table output/ablation_study/comparison_table.csv
    uv run python compute_ablation_deltas.py \
      --comparison-table .../comparison_table.csv --output .../ablation_deltas.csv
"""
import argparse
import os

import pandas as pd

from validate.convert_copus_sheet import INSTRUCTOR_CODES
from validate.validator import ARM_SHORT, MEAN_ROW_LABEL, SUMMARY_LABELS

REFERENCE_ARM = "multimodal"
ABLATED_ARMS = ["audio_only", "vision_only", "transcript_only"]

# Rows that are not COPUS codes and carry no delta.
NON_CODE_ROWS = set(SUMMARY_LABELS) | {MEAN_ROW_LABEL}


def _num(value):
    """A float, or None for N/A / blank / ERR cells."""
    v = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(v) else float(v)


def delta(arm_value, reference_value, sign="arm-minus-multimodal"):
    """
    One delta cell, or "N/A" when either side is undefined.
    """
    a, r = _num(arm_value), _num(reference_value)
    if a is None or r is None:
        return "N/A"
    return round(a - r if sign == "arm-minus-multimodal" else r - a, 3)


def compute_deltas(comparison_csv, output_csv=None, sign="arm-minus-multimodal"):
    """
    Build ablation_deltas.csv from a comparison table. Returns the DataFrame.
    """
    table = pd.read_csv(comparison_csv, dtype=str, keep_default_na=False)
    missing = [c for c in (f"{REFERENCE_ARM}_kappa", f"{REFERENCE_ARM}_ac1")
               if c not in table.columns]
    if missing:
        raise SystemExit(
            f"{comparison_csv} has no {', '.join(missing)} column. Table 6.2 is "
            f"defined against the multimodal arm, so the comparison table must "
            f"come from a 4-arm run (run_ablation.py, or run.py --arm all). If "
            f"this table predates the AC1 change, re-run the comparison step.")

    arms = [a for a in ABLATED_ARMS if f"{a}_kappa" in table.columns]
    skipped = [a for a in ABLATED_ARMS if a not in arms]
    if skipped:
        print(f"[warn] no columns for {skipped}; those deltas are omitted.")

    output_csv = output_csv or os.path.join(
        os.path.dirname(comparison_csv) or ".", "ablation_deltas.csv")

    rows = []
    for _, r in table.iterrows():
        code = r["code"]
        if code in NON_CODE_ROWS:
            continue
        row = {
            "professor_id": r.get("professor_id", ""),
            "code": code,
            "n_total_windows": r.get("n_total_windows", ""),
            "n_human_marked": r.get("n_human_marked", ""),
            f"n_ai_marked_{ARM_SHORT[REFERENCE_ARM]}":
                r.get(f"n_ai_marked_{ARM_SHORT[REFERENCE_ARM]}", ""),
        }
        for arm in arms:
            row[f"n_ai_marked_{ARM_SHORT[arm]}"] = r.get(f"n_ai_marked_{ARM_SHORT[arm]}", "")
        human_marked = _num(r.get("n_human_marked")) or 0
        for metric in ("kappa", "ac1"):
            for arm in arms:
                row[f"delta_{metric}_{ARM_SHORT[arm]}"] = (
                    delta(r.get(f"{arm}_{metric}"),
                          r.get(f"{REFERENCE_ARM}_{metric}"), sign)
                    if human_marked > 0 else "N/A")
        rows.append(row)

    cols = (["professor_id", "code", "n_total_windows", "n_human_marked",
             f"n_ai_marked_{ARM_SHORT[REFERENCE_ARM]}"]
            + [f"n_ai_marked_{ARM_SHORT[a]}" for a in arms]
            + [f"delta_{m}_{ARM_SHORT[a]}" for m in ("kappa", "ac1") for a in arms])
    df = pd.DataFrame(rows, columns=cols)
    # Protocol order, so the table reads like every other one in the thesis.
    order = {c: i for i, c in enumerate(INSTRUCTOR_CODES)}
    df = df.sort_values("code", key=lambda s: s.map(lambda c: order.get(c, len(order))))
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)

    _print_table(df, arms, sign)
    print(f"\nAblation deltas saved -> {output_csv}")
    return df


def _print_table(df, arms, sign):
    order = "arm - multimodal" if sign == "arm-minus-multimodal" else "multimodal - arm"
    print(f"\n=== Ablation deltas ({order}; negative = ablating hurt) ===")
    header = (f"{'Code':<8}{'Human+':>7}" +
              "".join(f"{'d' + ARM_SHORT[a][:4] + ' k/AC1':>18}" for a in arms))
    print(header)
    print("-" * len(header))
    for _, r in df.iterrows():
        line = f"{r['code']:<8}{str(r['n_human_marked']):>7}"
        for a in arms:
            cell = (f"{r[f'delta_kappa_{ARM_SHORT[a]}']} / "
                    f"{r[f'delta_ac1_{ARM_SHORT[a]}']}")
            line += f"{cell:>18}"
        print(line)
    print("-" * len(header))
    for a in arms:
        ks = [_num(v) for v in df[f"delta_kappa_{ARM_SHORT[a]}"]]
        ks = [k for k in ks if k is not None]
        if not ks:
            continue
        hurt = sum(1 for k in ks if k < 0)
        helped = sum(1 for k in ks if k > 0)
        print(f"  {a:16} kappa defined on {len(ks)} code(s): "
              f"{hurt} worse than multimodal, {helped} better")
    print("(N/A = undefined for one of the two arms, or never marked by the human; "
          "not a zero delta)")


def main():
    parser = argparse.ArgumentParser(
        description="Ablation deltas vs the multimodal arm (thesis Table 6.2)")
    parser.add_argument("--comparison-table", required=True,
                        help="comparison_table.csv from a 4-arm run")
    parser.add_argument("--output", default=None,
                        help="Output CSV (default: ablation_deltas.csv beside the input)")
    parser.add_argument("--sign", default="arm-minus-multimodal",
                        choices=["arm-minus-multimodal", "multimodal-minus-arm"],
                        help="Subtraction order. Default makes negative = ablating hurt.")
    args = parser.parse_args()
    compute_deltas(args.comparison_table, args.output, args.sign)


if __name__ == "__main__":
    main()
