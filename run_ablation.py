"""
Phase 6 — Full Ablation Validation batch runner.

Runs all 4 classification arms across one or more ground-truth lectures, then
produces the cross-arm Cohen's κ comparison table and comparison dashboards.

Usage:
    uv run python run_ablation.py \
      --lectures lecture_001.mp4 lecture_002.mp4 \
      --lecture-ids lecture_001 lecture_002 \
      --output-dir output/ablation_study \
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
from test_pipeline import ARM_CONFIG, run_arm
from validate.validator import compute_comparison_table
from report.dashboard import generate_comparison_dashboard


def chunks_exist(chunks_dir):
    """True if at least the first chunk's three variants are present."""
    return all(
        os.path.exists(os.path.join(chunks_dir, f"chunk_000_{suffix}.{ext}"))
        for suffix, ext in (("full", "mp4"), ("muted", "mp4"), ("audio", "mp3"))
    )


def process_lecture(lecture_path, lecture_id, output_dir, max_chunks, window_seconds):
    """Chunk + run all 4 arms for one lecture. Returns {arm: results_csv}."""
    lecture_dir = os.path.join(output_dir, lecture_id)
    chunks_dir = os.path.join(lecture_dir, "chunks")

    print(f"\n########## Lecture: {lecture_id} ##########")
    if chunks_exist(chunks_dir):
        print(f"Chunks already exist in {chunks_dir}, skipping chunking")
    else:
        print("=== Chunking ===")
        chunk_video(
            input_path=lecture_path,
            output_dir=chunks_dir,
            window_seconds=window_seconds,
        )

    # Determine how many chunks to process if not capped.
    if max_chunks is None:
        n = len([f for f in os.listdir(chunks_dir) if f.endswith("_full.mp4")])
    else:
        n = max_chunks

    arm_csvs = {}
    for arm_name in ARM_CONFIG:
        arm_csvs[arm_name] = run_arm(
            arm_name,
            chunks_dir=chunks_dir,
            results_dir=os.path.join(lecture_dir, f"results_{arm_name}"),
            output_csv=os.path.join(lecture_dir, f"results_{arm_name}.csv"),
            lecture_id=lecture_id,
            max_chunks=n,
            window_seconds=window_seconds,
        )
    return arm_csvs


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
    parser.add_argument("--window-seconds", type=int, default=120)
    args = parser.parse_args()

    if len(args.lectures) != len(args.lecture_ids):
        parser.error(
            f"--lectures ({len(args.lectures)}) and --lecture-ids "
            f"({len(args.lecture_ids)}) must have the same length"
        )

    os.makedirs(args.output_dir, exist_ok=True)

    per_lecture_arm_csvs = {}
    for lecture_path, lecture_id in zip(args.lectures, args.lecture_ids):
        per_lecture_arm_csvs[lecture_id] = process_lecture(
            lecture_path, lecture_id, args.output_dir,
            args.max_chunks, args.window_seconds,
        )

    # Validation — only for lectures whose human coding exists.
    validatable = []
    human_paths = []
    for lecture_id, arm_csvs in per_lecture_arm_csvs.items():
        human_csv = os.path.join(args.human_dir, f"human_coding_{lecture_id}.csv")
        if os.path.exists(human_csv):
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

    print("\nAblation study complete.")


if __name__ == "__main__":
    main()
