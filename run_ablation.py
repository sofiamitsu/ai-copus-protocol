"""
Phase 6 — Full Ablation Validation batch runner.

Runs all 4 classification arms across one or more ground-truth lectures, then
produces the cross-arm Cohen's κ comparison table and comparison dashboards.

Every arm sees the SAME prompt (classify.classifier.PROMPT_COPUS_UNIFIED), the
SAME chunk indices (--windows), and the SAME model (--model). The modality is
the independent variable and lives purely in the input each arm is handed:
video+audio, silent video, audio, or a scrubbed transcript. That is what makes
the per-code × per-arm κ table comparable across arms.

Usage:
    uv run python run_ablation.py \
      --lectures lecture_001.mp4 lecture_002.mp4 \
      --lecture-ids lecture_001 lecture_002 \
      --output-dir output/ablation_study \
      [--windows 0,1,2,3] [--model gemini-2.5-pro] \
      [--max-chunks N] [--window-seconds 120]

Human ground truth is expected at human_coding_{lecture_id}.csv (repo root by
default; override with --human-dir). Validation is skipped per-lecture if its
human CSV is missing.
"""
import argparse
import os
from dotenv import load_dotenv
load_dotenv()

from chunk.chunker import chunk_video
from classify.classifier import DEFAULT_MODEL, VALID_MODELS
from run import chunks_missing
from test_pipeline import ARM_CONFIG, run_arm
from validate.validator import compute_comparison_table, load_human_codes
from report.dashboard import generate_comparison_dashboard


def process_lecture(lecture_path, lecture_id, output_dir, max_chunks, window_seconds,
                    window_indices=None, model=DEFAULT_MODEL):
    """
    Chunk + run all 4 arms for one lecture, on identical chunk indices.
    Returns ({arm: results_csv}, indices_processed).
    """
    lecture_dir = os.path.join(output_dir, lecture_id)
    chunks_dir = os.path.join(lecture_dir, "chunks")

    print(f"\n########## Lecture: {lecture_id} ##########")

    if window_indices is not None:
        indices = sorted(set(window_indices))
    elif max_chunks is not None:
        indices = list(range(max_chunks))
    else:
        indices = None  # run_arm derives it from what is on disk

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

    arm_csvs = {}
    processed_indices = None
    for arm_name in ARM_CONFIG:
        result = run_arm(
            arm_name,
            chunks_dir=chunks_dir,
            results_dir=os.path.join(lecture_dir, f"results_{arm_name}"),
            output_csv=os.path.join(lecture_dir, f"results_{arm_name}.csv"),
            lecture_id=lecture_id,
            window_indices=indices,
            window_seconds=window_seconds,
            model=model,
        )
        arm_csvs[arm_name] = result["csv"]
        # Every arm must cover the same windows or the κ columns aren't comparable.
        if processed_indices is None:
            processed_indices = result["indices"]
        elif result["indices"] != processed_indices:
            print(f"[warn] arm {arm_name} processed a different window set than "
                  f"the first arm — κ columns will not be comparable.")
        if result["failed"]:
            print(f"[warn] arm {arm_name}: failed chunks {result['failed']}")

    return arm_csvs, (processed_indices or [])


def main():
    parser = argparse.ArgumentParser(description="Phase 6 ablation batch runner")
    parser.add_argument("--lectures", nargs="+", required=True,
                        help="Lecture .mp4 paths")
    parser.add_argument("--lecture-ids", nargs="+", required=True,
                        help="Lecture ids, aligned with --lectures")
    parser.add_argument("--output-dir", default="output/ablation_study")
    parser.add_argument("--human-dir", default=".",
                        help="Dir holding human_coding_{id}.csv files")
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="Cap chunks per lecture (default: all)")
    parser.add_argument("--windows", default=None,
                        help="Comma-separated chunk indices, applied to ALL arms "
                             "and all lectures. Get them from "
                             "`python -m utils.sparse_windows --duration N`.")
    parser.add_argument("--model", choices=VALID_MODELS, default=DEFAULT_MODEL,
                        help=f"Gemini model for classification (default: {DEFAULT_MODEL})")
    parser.add_argument("--window-seconds", type=int, default=120)
    args = parser.parse_args()

    if len(args.lectures) != len(args.lecture_ids):
        parser.error(
            f"--lectures ({len(args.lectures)}) and --lecture-ids "
            f"({len(args.lecture_ids)}) must have the same length"
        )
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

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Arms: {list(ARM_CONFIG)}")
    print(f"Model: {args.model}")
    if window_indices is not None:
        print(f"Windows: {len(window_indices)} sparse chunks "
              f"({window_indices[0]}..{window_indices[-1]})")
    print(f"Output: {args.output_dir}")

    per_lecture_arm_csvs = {}
    per_lecture_indices = {}
    for lecture_path, lecture_id in zip(args.lectures, args.lecture_ids):
        arm_csvs, indices = process_lecture(
            lecture_path, lecture_id, args.output_dir,
            args.max_chunks, args.window_seconds,
            window_indices=window_indices, model=args.model,
        )
        per_lecture_arm_csvs[lecture_id] = arm_csvs
        per_lecture_indices[lecture_id] = indices

    # Validation — only for lectures whose human coding exists.
    validatable = []
    human_paths = []
    for lecture_id, arm_csvs in per_lecture_arm_csvs.items():
        human_csv = os.path.join(args.human_dir, f"human_coding_{lecture_id}.csv")
        if os.path.exists(human_csv):
            # The human CSV must cover the windows the arms actually ran on.
            human_rows = load_human_codes(human_csv)
            human_idx = {w["window_index"] for w in human_rows}
            processed = set(per_lecture_indices[lecture_id])
            uncovered = sorted(processed - human_idx)
            if uncovered:
                print(f"[warn] {lecture_id}: {len(uncovered)} classified window(s) "
                      f"have no human coding and are excluded from κ: {uncovered}")
            # A human row that is PRESENT but empty is far more dangerous than a
            # missing one: κ scores it as "the human saw nothing here", so every
            # code the AI emits becomes a false positive. On lecture_001 a single
            # such window (the 19.9s trailing partial, which the 30-row template
            # padded blank) cost 0.35 κ on Lec.
            blank = sorted(w["window_index"] for w in human_rows
                           if w["window_index"] in processed and not w["codes"])
            if blank:
                print(f"[warn] {lecture_id}: {len(blank)} human window(s) are coded "
                      f"EMPTY and will score every AI code there as a false "
                      f"positive: {blank}. Confirm these were genuinely observed "
                      f"and not just uncoded rows.")
            validatable.append((lecture_id, arm_csvs))
            human_paths.append(human_csv)
        else:
            print(f"\n[skip validation] no human coding at {human_csv}")

    if not validatable:
        print("\nNo human-coded lectures found — skipping comparison table/dashboard.")
        return

    lecture_dicts = [ac for _, ac in validatable]
    print("\n========== Comparison table ==========")
    compute_comparison_table(
        arm_csvs=lecture_dicts,
        human_csv=human_paths,
        output_csv=os.path.join(args.output_dir, "comparison_table.csv"),
        per_lecture=True,
    )

    # One comparison dashboard per lecture (timelines are per-lecture);
    # the κ bar chart reuses the pooled comparison table.
    comparison_csv = os.path.join(args.output_dir, "comparison_table.csv")
    for lecture_id, arm_csvs in validatable:
        out_html = os.path.join(
            args.output_dir, lecture_id, "comparison_dashboard.html"
        )
        generate_comparison_dashboard(arm_csvs, comparison_csv, out_html)

    print(f"\nAblation study complete. Model: {args.model}")


if __name__ == "__main__":
    main()
