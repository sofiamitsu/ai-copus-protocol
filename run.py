"""
Phase 7 — run.py CLI entry point.

Processes one professor's full data bundle in a single command:
  - 3 lectures (.mp4)
  - 2 sets of manual COPUS coding (Sofia + Dr. Kaw), as filled Excel sheets
  - 1 optional student survey (.csv)

For every lecture it chunks the video, classifies each 2-min window with the
selected arm(s), aggregates to a results CSV, converts both human coders'
Excel sheets to long-format CSVs, computes three Cohen's kappa comparisons
(AI-vs-Sofia, AI-vs-Kaw, Sofia-vs-Kaw), and renders a behavioral timeline.
After all lectures it pools kappa across lectures, builds a professor-level
dashboard, and (if a survey was given) writes a survey summary.

Usage:
    uv run python run.py \
      --professor "Dr. Smith" \
      --lectures lecture_A.mp4 lecture_B.mp4 lecture_C.mp4 \
      --lecture-ids lecture_A lecture_B lecture_C \
      --sofia-coding sofia_A.xlsx sofia_B.xlsx sofia_C.xlsx \
      --kaw-coding kaw_A.xlsx kaw_B.xlsx kaw_C.xlsx \
      --survey survey_results.csv \
      --output-dir output/dr_smith \
      --arm multimodal

--arm defaults to "multimodal" but accepts "all" to run the full ablation
study (all 4 modality arms) alongside the primary comparison.
"""
import argparse
import os

import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from chunk.chunker import chunk_video
from test_pipeline import ARM_CONFIG, run_arm
from validate.convert_copus_sheet import convert
from validate.validator import (
    compute_kappa,
    compute_comparison_table,
    load_human_codes,
    load_ai_codes,
    kappa_by_code,
)
from report.dashboard import generate_dashboard, generate_comparison_dashboard

WINDOW_SECONDS = 120


def chunks_exist(chunks_dir):
    """True if at least the first chunk's three variants are present."""
    return all(
        os.path.exists(os.path.join(chunks_dir, f"chunk_000_{suffix}.{ext}"))
        for suffix, ext in (("full", "mp4"), ("muted", "mp4"), ("audio", "mp3"))
    )


def count_result_jsons(results_dir):
    """How many per-chunk result JSONs a results dir actually contains."""
    if not os.path.isdir(results_dir):
        return 0
    return len([f for f in os.listdir(results_dir) if f.endswith("_result.json")])


def process_lecture(lecture_path, lecture_id, sofia_xlsx, kaw_xlsx,
                    lecture_dir, arms_to_run, primary_arm,
                    max_chunks, window_seconds):
    """
    Run the full per-lecture pipeline. Returns a dict describing this lecture's
    outputs (arm CSVs, human CSVs, chunk/error counts) for professor-level pooling.

    kaw_xlsx may be None (Dr. Kaw coding is optional): when absent, the
    AI-vs-Kaw and Sofia-vs-Kaw comparisons are skipped and human_kaw is None.
    """
    chunks_dir = os.path.join(lecture_dir, "chunks")

    print(f"\n########## Lecture: {lecture_id} ##########")

    # 1. Chunk (idempotent — skip if already chunked)
    if chunks_exist(chunks_dir):
        print(f"Chunks already exist in {chunks_dir}, skipping chunking")
    else:
        print("=== Chunking ===")
        chunk_video(
            input_path=lecture_path,
            output_dir=chunks_dir,
            window_seconds=window_seconds,
        )

    # How many windows to process.
    if max_chunks is None:
        n = len([f for f in os.listdir(chunks_dir) if f.endswith("_full.mp4")])
    else:
        n = max_chunks

    # 2 + 3. Classify + aggregate each requested arm.
    arm_csvs = {}
    for arm_name in arms_to_run:
        results_dir = os.path.join(lecture_dir, f"results_{arm_name}")
        arm_csvs[arm_name] = run_arm(
            arm_name,
            chunks_dir=chunks_dir,
            results_dir=results_dir,
            output_csv=os.path.join(lecture_dir, f"results_{arm_name}.csv"),
            lecture_id=lecture_id,
            max_chunks=n,
            window_seconds=window_seconds,
        )

    primary_csv = arm_csvs[primary_arm]
    primary_results_dir = os.path.join(lecture_dir, f"results_{primary_arm}")
    processed = count_result_jsons(primary_results_dir)
    errors = max(n - processed, 0)
    if errors:
        print(f"[warn] {lecture_id}: {errors}/{n} windows have no {primary_arm} "
              f"result (failed/skipped chunks)")

    # 4. Convert human coders' Excel sheets to long-format CSVs (Kaw optional).
    human_sofia = os.path.join(lecture_dir, "human_sofia.csv")
    print("\n=== Converting human coding: Sofia ===")
    convert(sofia_xlsx, lecture_id, human_sofia)
    human_kaw = None
    if kaw_xlsx:
        human_kaw = os.path.join(lecture_dir, "human_kaw.csv")
        print("=== Converting human coding: Dr. Kaw ===")
        convert(kaw_xlsx, lecture_id, human_kaw)

    _warn_window_mismatch(lecture_id, primary_csv, human_sofia, human_kaw)

    # 5. Kappa comparisons for this lecture (Kaw comparisons only if present).
    print("\n=== Validation: AI vs Sofia ===")
    compute_kappa(primary_csv, human_sofia, lecture_dir,
                  output_filename="kappa_ai_vs_sofia.csv",
                  label_a="AI", label_b="Sofia")
    if human_kaw:
        print("\n=== Validation: AI vs Dr. Kaw ===")
        compute_kappa(primary_csv, human_kaw, lecture_dir,
                      output_filename="kappa_ai_vs_kaw.csv",
                      label_a="AI", label_b="Kaw")
        print("\n=== Validation: Sofia vs Dr. Kaw (inter-rater) ===")
        compute_kappa(human_sofia, human_kaw, lecture_dir,
                      output_filename="kappa_sofia_vs_kaw.csv",
                      label_a="Sofia", label_b="Kaw")

    # 6. Per-lecture behavioral timeline (primary arm).
    print("\n=== Dashboard ===")
    generate_dashboard(primary_csv, os.path.join(lecture_dir, "dashboard.html"))

    # If running the full ablation, also emit the cross-arm comparison for this
    # lecture (Sofia is the reference human here).
    if len(arms_to_run) > 1:
        print("\n=== Ablation comparison (this lecture) ===")
        comparison_csv = os.path.join(lecture_dir, "comparison_table.csv")
        compute_comparison_table(
            arm_csvs=arm_csvs,
            human_csv=human_sofia,
            output_csv=comparison_csv,
        )
        generate_comparison_dashboard(
            arm_csvs, comparison_csv,
            os.path.join(lecture_dir, "comparison_dashboard.html"),
        )

    return {
        "lecture_id": lecture_id,
        "primary_csv": primary_csv,
        "human_sofia": human_sofia,
        "human_kaw": human_kaw,
        "processed": processed,
        "attempted": n,
        "errors": errors,
    }


def _warn_window_mismatch(lecture_id, ai_csv, sofia_csv, kaw_csv):
    """
    Warn (don't fail) if a human sheet doesn't cover the same windows as the AI
    run. kappa comparisons already restrict to the shared window overlap.
    """
    ai_n = len(load_ai_codes(ai_csv))
    humans = [("Sofia", sofia_csv)]
    if kaw_csv:
        humans.append(("Dr. Kaw", kaw_csv))
    for name, path in humans:
        human_n = len(load_human_codes(path))
        if human_n < ai_n:
            print(f"[warn] {lecture_id}: {name} coded {human_n} windows but AI "
                  f"produced {ai_n}; comparing the overlapping windows only.")


def build_combined_kappa(lecture_infos, output_csv):
    """
    Pool every lecture's windows and compute one kappa per code for each of the
    three comparisons, giving a professor-level reliability table.
    """
    ai, sofia, kaw = [], [], []
    have_kaw = all(info.get("human_kaw") for info in lecture_infos)
    for info in lecture_infos:
        ai.extend(load_ai_codes(info["primary_csv"]))
        sofia.extend(load_human_codes(info["human_sofia"]))
        if have_kaw:
            kaw.extend(load_human_codes(info["human_kaw"]))

    comparisons = {"ai_vs_sofia": kappa_by_code(sofia, ai)}
    if have_kaw:
        comparisons["ai_vs_kaw"] = kappa_by_code(kaw, ai)
        comparisons["sofia_vs_kaw"] = kappa_by_code(sofia, kaw)
    # code -> {comparison: kappa}
    kappa_by_comp = {
        comp: {r["code"]: r["kappa"] for r in rows}
        for comp, rows in comparisons.items()
    }
    all_codes = sorted({c for rows in comparisons.values() for c in
                        (r["code"] for r in rows)})

    rows = []
    for code in all_codes:
        rows.append({
            "code": code,
            "ai_vs_sofia_kappa": kappa_by_comp.get("ai_vs_sofia", {}).get(code, "N/A"),
            "ai_vs_kaw_kappa": kappa_by_comp.get("ai_vs_kaw", {}).get(code, "N/A"),
            "sofia_vs_kaw_kappa": kappa_by_comp.get("sofia_vs_kaw", {}).get(code, "N/A"),
        })
    df = pd.DataFrame(rows, columns=["code", "ai_vs_sofia_kappa",
                                     "ai_vs_kaw_kappa", "sofia_vs_kaw_kappa"])
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)

    print("\n=== Combined kappa (pooled across lectures) ===")
    header = f"{'Code':<8}{'AI-Sofia':>12}{'AI-Kaw':>12}{'Sofia-Kaw':>12}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['code']:<8}{str(r['ai_vs_sofia_kappa']):>12}"
              f"{str(r['ai_vs_kaw_kappa']):>12}{str(r['sofia_vs_kaw_kappa']):>12}")
    print(f"\nCombined kappa saved → {output_csv}")
    return df


def build_professor_dashboard(lecture_infos, output_html, window_seconds):
    """
    Lay every lecture's primary-arm timeline end-to-end on one time axis to give
    a combined behavioral profile for the professor.
    """
    frames = []
    offset = 0
    for info in lecture_infos:
        df = pd.read_csv(info["primary_csv"])
        if df.empty:
            continue
        df = df.copy()
        df["window_start"] = df["window_start"] + offset
        df["window_end"] = df["window_end"] + offset
        frames.append(df)
        # Next lecture starts one window past this lecture's last window.
        offset = int(df["window_end"].max())

    if not frames:
        print("[warn] no results to build professor dashboard")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined_csv = os.path.splitext(output_html)[0] + "_results.csv"
    combined.to_csv(combined_csv, index=False)
    generate_dashboard(combined_csv, output_html)


def analyze_survey(survey_csv, output_csv):
    """
    Summarize a student survey CSV: numeric columns get count/mean/median/
    min/max; categorical columns get their top response and unique count.
    Best-effort — a malformed survey warns instead of crashing the run.
    """
    try:
        df = pd.read_csv(survey_csv)
    except Exception as e:
        print(f"[warn] could not read survey {survey_csv}: {e}")
        return

    rows = []
    for col in df.columns:
        series = df[col]
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() >= max(1, len(series) // 2):
            rows.append({
                "question": col,
                "type": "numeric",
                "n": int(numeric.notna().sum()),
                "mean": round(float(numeric.mean()), 3) if numeric.notna().any() else "N/A",
                "median": round(float(numeric.median()), 3) if numeric.notna().any() else "N/A",
                "min": numeric.min() if numeric.notna().any() else "N/A",
                "max": numeric.max() if numeric.notna().any() else "N/A",
                "top_response": "",
            })
        else:
            counts = series.dropna().astype(str).value_counts()
            top = counts.index[0] if not counts.empty else "N/A"
            rows.append({
                "question": col,
                "type": "categorical",
                "n": int(series.notna().sum()),
                "mean": "", "median": "", "min": "", "max": "",
                "top_response": f"{top} ({int(counts.iloc[0])})" if not counts.empty else "N/A",
            })

    out_df = pd.DataFrame(rows, columns=["question", "type", "n", "mean",
                                         "median", "min", "max", "top_response"])
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"\nSurvey analysis saved → {output_csv} ({len(rows)} questions)")


def main():
    parser = argparse.ArgumentParser(description="Phase 7 professor-bundle runner")
    parser.add_argument("--professor", required=True, help="Professor name (label only)")
    parser.add_argument("--lectures", nargs="+", required=True, help="Lecture .mp4 paths")
    parser.add_argument("--lecture-ids", nargs="+", required=True,
                        help="Lecture ids, aligned with --lectures")
    parser.add_argument("--sofia-coding", nargs="+", required=True,
                        help="Sofia's filled COPUS .xlsx per lecture")
    parser.add_argument("--kaw-coding", nargs="+", required=True,
                        help="Dr. Kaw's filled COPUS .xlsx per lecture")
    parser.add_argument("--survey", default=None, help="Optional student survey .csv")
    parser.add_argument("--output-dir", required=True, help="Output dir, e.g. output/dr_smith")
    parser.add_argument("--arm", choices=list(ARM_CONFIG) + ["all"], default="multimodal")
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="Cap windows per lecture (default: all)")
    parser.add_argument("--window-seconds", type=int, default=WINDOW_SECONDS)
    args = parser.parse_args()

    # All per-lecture list args must be the same length.
    lists = {
        "--lectures": args.lectures,
        "--lecture-ids": args.lecture_ids,
        "--sofia-coding": args.sofia_coding,
        "--kaw-coding": args.kaw_coding,
    }
    n_lectures = len(args.lectures)
    for flag, values in lists.items():
        if len(values) != n_lectures:
            parser.error(
                f"{flag} has {len(values)} values but --lectures has "
                f"{n_lectures}; all per-lecture args must align."
            )

    arms_to_run = list(ARM_CONFIG) if args.arm == "all" else [args.arm]
    primary_arm = "multimodal" if "multimodal" in arms_to_run else arms_to_run[0]

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Professor: {args.professor}")
    print(f"Arms: {arms_to_run}  (primary for kappa: {primary_arm})")
    print(f"Output: {args.output_dir}")

    lecture_infos = []
    for lecture_path, lecture_id, sofia_xlsx, kaw_xlsx in zip(
        args.lectures, args.lecture_ids, args.sofia_coding, args.kaw_coding
    ):
        info = process_lecture(
            lecture_path=lecture_path,
            lecture_id=lecture_id,
            sofia_xlsx=sofia_xlsx,
            kaw_xlsx=kaw_xlsx,
            lecture_dir=os.path.join(args.output_dir, lecture_id),
            arms_to_run=arms_to_run,
            primary_arm=primary_arm,
            max_chunks=args.max_chunks,
            window_seconds=args.window_seconds,
        )
        lecture_infos.append(info)

    # 7. Combined kappa across all lectures.
    print("\n========== Professor-level outputs ==========")
    build_combined_kappa(
        lecture_infos, os.path.join(args.output_dir, "combined_kappa.csv")
    )

    # 8. Professor-level behavioral dashboard.
    build_professor_dashboard(
        lecture_infos,
        os.path.join(args.output_dir, "professor_dashboard.html"),
        args.window_seconds,
    )

    # Optional survey.
    if args.survey:
        analyze_survey(
            args.survey, os.path.join(args.output_dir, "survey_analysis.csv")
        )

    # Final progress summary.
    total_processed = sum(i["processed"] for i in lecture_infos)
    total_attempted = sum(i["attempted"] for i in lecture_infos)
    total_errors = sum(i["errors"] for i in lecture_infos)
    print("\n========== Run summary ==========")
    print(f"Professor: {args.professor}")
    print(f"Lectures: {len(lecture_infos)}")
    print(f"Windows processed: {total_processed}/{total_attempted} "
          f"(primary arm: {primary_arm})")
    print(f"Failed/skipped windows: {total_errors}")
    for info in lecture_infos:
        print(f"  {info['lecture_id']}: {info['processed']}/{info['attempted']} "
              f"windows, {info['errors']} errors")
    print(f"All outputs under: {args.output_dir}")


if __name__ == "__main__":
    main()
