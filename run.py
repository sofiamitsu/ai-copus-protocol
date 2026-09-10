"""
Phase 7 — run.py CLI entry point.

Processes one professor's full data bundle in a single command:
  - 3 lectures (.mp4)
  - 1 set of manual COPUS coding (Sofia), as filled Excel sheets
  - 1 optional student survey (.csv)

For every lecture it chunks the video, classifies each 2-min window with the
selected arm(s), aggregates to a results CSV, converts both human coders'
Excel sheet to a long-format CSV, computes the AI-vs-Sofia Cohen's kappa, and
renders a behavioral timeline.
After all lectures it pools kappa across lectures, builds a professor-level
dashboard, and (if a survey was given) writes a survey summary.

Usage:
    uv run python run.py \
      --professor "Dr. Smith" \
      --lectures lecture_A.mp4 lecture_B.mp4 lecture_C.mp4 \
      --lecture-ids lecture_A lecture_B lecture_C \
      --sofia-coding sofia_A.xlsx sofia_B.xlsx sofia_C.xlsx \
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
from classify.classifier import DEFAULT_MODEL, VALID_MODELS
from test_pipeline import ARM_CONFIG, run_arm
from validate.convert_copus_sheet import convert
from validate.validator import (
    MEAN_ROW_LABEL,
    compute_kappa,
    compute_comparison_table,
    load_human_codes,
    load_ai_codes,
    kappa_by_code,
    mean_kappa,
)
from report.dashboard import generate_dashboard, generate_comparison_dashboard

WINDOW_SECONDS = 120


def chunks_missing(chunks_dir, window_indices=None):
    """
    Which of the required chunks are absent from chunks_dir.

    Checking every required index (not just chunk_000) matters once sparse
    sampling exists: a directory chunked for one window set would otherwise look
    "already chunked" to a run that needs a different set.
    """
    if not os.path.isdir(chunks_dir):
        return list(window_indices) if window_indices is not None else [0]
    required = window_indices if window_indices is not None else [0]

    def usable(path):
        # A 0-byte file is what a failed ffmpeg cut leaves behind; treating it as
        # present would skip re-cutting it and fail later at classification.
        return os.path.exists(path) and os.path.getsize(path) > 0

    return [
        i for i in required
        if not all(
            usable(os.path.join(chunks_dir, f"chunk_{i:03d}_{suffix}.{ext}"))
            for suffix, ext in (("full", "mp4"), ("muted", "mp4"), ("audio", "mp3"))
        )
    ]


def process_lecture(lecture_path, lecture_id, sofia_xlsx,
                    lecture_dir, arms_to_run, primary_arm,
                    max_chunks, window_seconds,
                    window_indices=None, model=DEFAULT_MODEL):
    """
    Run the full per-lecture pipeline. Returns a dict describing this lecture's
    outputs (arm CSV, human CSV, chunk/error counts) for professor-level pooling.
    """
    chunks_dir = os.path.join(lecture_dir, "chunks")

    print(f"\n########## Lecture: {lecture_id} ##########")

    # Which windows this run covers: explicit sparse list, a cap, or all of them.
    if window_indices is not None:
        indices = sorted(set(window_indices))
    elif max_chunks is not None:
        indices = list(range(max_chunks))
    else:
        indices = None  # run_arm derives it from what is on disk

    # 1. Chunk (idempotent — cut only what is missing for THIS window set)
    missing = chunks_missing(chunks_dir, indices)
    if not missing:
        print(f"Chunks already exist in {chunks_dir}, skipping chunking")
    else:
        print(f"=== Chunking ({len(missing)} window(s) to cut) ===")
        chunk_video(
            input_path=lecture_path,
            output_dir=chunks_dir,
            window_seconds=window_seconds,
            chunk_indices=indices,
        )

    # 2 + 3. Classify + aggregate each requested arm.
    arm_csvs = {}
    arm_results = {}
    for arm_name in arms_to_run:
        results_dir = os.path.join(lecture_dir, f"results_{arm_name}")
        arm_results[arm_name] = run_arm(
            arm_name,
            chunks_dir=chunks_dir,
            results_dir=results_dir,
            output_csv=os.path.join(lecture_dir, f"results_{arm_name}.csv"),
            lecture_id=lecture_id,
            window_indices=indices,
            window_seconds=window_seconds,
            model=model,
        )
        arm_csvs[arm_name] = arm_results[arm_name]["csv"]

    primary = arm_results[primary_arm]
    primary_csv = primary["csv"]
    # The windows the primary arm actually covered — this is what the human
    # sheet must be coded against.
    indices = primary["indices"]
    n = len(indices)
    processed = primary["processed"]
    errors = len(primary["failed"]) + len(primary["missing"])
    if errors:
        print(f"[warn] {lecture_id}: {processed}/{n} windows have a {primary_arm} "
              f"result — {len(primary['failed'])} failed, "
              f"{len(primary['missing'])} had no media on disk")

    # 4. Convert the human coding Excel sheet to a long-format CSV.
    # The same window list the classifier used, so AI and human windows line up
    # by construction rather than by coincidence.
    human_sofia = os.path.join(lecture_dir, "human_sofia.csv")
    print("\n=== Converting human coding: Sofia ===")
    convert(sofia_xlsx, lecture_id, human_sofia,
            windows=indices, window_seconds=window_seconds)
    _warn_window_mismatch(lecture_id, primary_csv, human_sofia)

    # 5. Kappa for this lecture.
    # The HUMAN csv must be csv_a: compute_kappa names its count columns
    # human_positive_windows / ai_positive_windows after csv_a / csv_b, so
    # passing the AI csv first silently swaps them (see compute_baseline_kappa.py).
    # Kappa itself is symmetric; only the two count columns depend on the order.
    print("\n=== Validation: AI vs Sofia ===")
    compute_kappa(human_sofia, primary_csv, lecture_dir,
                  output_filename="kappa_ai_vs_sofia.csv",
                  label_a="Sofia", label_b="AI")

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
        "processed": processed,
        "attempted": n,
        "errors": errors,
        "failed": primary["failed"],
        "indices": indices,
        "model": model,
    }


def _warn_window_mismatch(lecture_id, ai_csv, sofia_csv):
    """
    Warn (don't fail) if a human sheet doesn't cover the same windows as the AI
    run. kappa comparisons already restrict to the shared window overlap.
    """
    ai_n = len(load_ai_codes(ai_csv))
    for name, path in [("Sofia", sofia_csv)]:
        human_n = len(load_human_codes(path))
        if human_n < ai_n:
            print(f"[warn] {lecture_id}: {name} coded {human_n} windows but AI "
                  f"produced {ai_n}; comparing the overlapping windows only.")


def build_combined_kappa(lecture_infos, output_csv):
    """
    Pool every lecture's windows and compute one AI-vs-Sofia kappa per code,
    giving a professor-level reliability table.
    """
    ai, sofia = [], []
    for info in lecture_infos:
        ai.extend(load_ai_codes(info["primary_csv"]))
        sofia.extend(load_human_codes(info["human_sofia"]))

    comparisons = {"ai_vs_sofia": kappa_by_code(sofia, ai)}
    # code -> {comparison: kappa}
    kappa_by_comp = {
        comp: {r["code"]: r["kappa"] for r in rows}
        for comp, rows in comparisons.items()
    }
    all_codes = sorted({c for rows in comparisons.values() for c in
                        (r["code"] for r in rows)})

    kappa_cols = ["ai_vs_sofia_kappa"]
    rows = []
    for code in all_codes:
        rows.append({
            "code": code,
            "ai_vs_sofia_kappa": kappa_by_comp.get("ai_vs_sofia", {}).get(code, "N/A"),
        })
    # MEAN over each comparison's numeric kappas; N/A codes are excluded, not
    # counted as 0.00.
    mean_row = {"code": MEAN_ROW_LABEL}
    for col in kappa_cols:
        mean_row[col] = mean_kappa([r[col] for r in rows])
    rows.append(mean_row)
    df = pd.DataFrame(rows, columns=["code"] + kappa_cols)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)

    print("\n=== Combined kappa (pooled across lectures) ===")
    header = f"{'Code':<8}{'AI-Sofia':>12}"
    print(header)
    print("-" * len(header))
    for r in rows:
        if r["code"] == MEAN_ROW_LABEL:
            print("-" * len(header))
        print(f"{r['code']:<8}{str(r['ai_vs_sofia_kappa']):>12}")
    print("(N/A = kappa undefined: a rater never used that code; excluded from MEAN)")
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
    parser.add_argument("--survey", default=None, help="Optional student survey .csv")
    parser.add_argument("--output-dir", required=True, help="Output dir, e.g. output/dr_smith")
    parser.add_argument("--arm", choices=list(ARM_CONFIG) + ["all"], default="multimodal")
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="Cap windows per lecture (default: all)")
    parser.add_argument("--windows", default=None,
                        help="Comma-separated chunk indices to process, e.g. "
                             "0,1,2,3. Get them from "
                             "`python -m utils.sparse_windows --duration N`. "
                             "Applies to every lecture in the run.")
    parser.add_argument("--model", choices=VALID_MODELS, default=DEFAULT_MODEL,
                        help=f"Gemini model for classification (default: {DEFAULT_MODEL})")
    parser.add_argument("--window-seconds", type=int, default=WINDOW_SECONDS)
    args = parser.parse_args()

    if args.windows is not None and args.max_chunks is not None:
        parser.error("--windows and --max-chunks are mutually exclusive; "
                     "--windows already names exactly which chunks to process.")
    window_indices = None
    if args.windows is not None:
        try:
            window_indices = sorted({int(w) for w in args.windows.split(",") if w.strip()})
        except ValueError:
            parser.error(f"--windows must be comma-separated integers, got: {args.windows}")
        if not window_indices:
            parser.error("--windows was empty")
        if window_indices[0] < 0:
            parser.error(f"--windows must be non-negative, got {window_indices[0]}")

    # All per-lecture list args must be the same length.
    n_lectures = len(args.lectures)
    lists = {
        "--lectures": args.lectures,
        "--lecture-ids": args.lecture_ids,
        "--sofia-coding": args.sofia_coding,
    }
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
    print(f"Model: {args.model}")
    if window_indices is not None:
        print(f"Windows: {len(window_indices)} sparse chunks "
              f"({window_indices[0]}..{window_indices[-1]})")
    print(f"Output: {args.output_dir}")

    lecture_infos = []
    for lecture_path, lecture_id, sofia_xlsx in zip(
        args.lectures, args.lecture_ids, args.sofia_coding
    ):
        info = process_lecture(
            lecture_path=lecture_path,
            lecture_id=lecture_id,
            sofia_xlsx=sofia_xlsx,
            lecture_dir=os.path.join(args.output_dir, lecture_id),
            arms_to_run=arms_to_run,
            primary_arm=primary_arm,
            max_chunks=args.max_chunks,
            window_seconds=args.window_seconds,
            window_indices=window_indices,
            model=args.model,
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
    print(f"Model: {args.model}")
    print(f"Lectures: {len(lecture_infos)}")
    print(f"Windows processed: {total_processed}/{total_attempted} "
          f"(primary arm: {primary_arm})")
    print(f"Failed/skipped windows: {total_errors}")
    for info in lecture_infos:
        line = (f"  {info['lecture_id']}: {info['processed']}/{info['attempted']} "
                f"windows, {info['errors']} errors")
        if info["failed"]:
            line += f" (failed chunks: {info['failed']})"
        print(line)
    print(f"All outputs under: {args.output_dir}")


if __name__ == "__main__":
    main()
