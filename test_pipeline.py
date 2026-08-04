import argparse
import os
from dotenv import load_dotenv
load_dotenv()

from chunk.chunker import chunk_video
from classify.classifier import (
    classify_chunk_multimodal,
    classify_chunk_vision_only,
    classify_chunk_audio_only,
    classify_chunk_transcript_only,
)
from aggregate.aggregator import aggregate_results

LECTURE_PATH = "videoplayback.mp4"  # replace with your filename
CHUNKS_DIR = "output/pipeline_test/chunks"
LECTURE_ID = "lecture_001"
WINDOW_SECONDS = 120
MAX_CHUNKS = 24

ARM_CONFIG = {
    "multimodal":     {"suffix": "full", "ext": "mp4", "fn": classify_chunk_multimodal},
    "vision_only":    {"suffix": "muted", "ext": "mp4", "fn": classify_chunk_vision_only},
    "audio_only":     {"suffix": "audio", "ext": "mp3", "fn": classify_chunk_audio_only},
    "transcript_only": {"suffix": "audio", "ext": "mp3", "fn": classify_chunk_transcript_only},
}


def run_arm(arm_name, chunks_dir, results_dir, output_csv, lecture_id,
            max_chunks, window_seconds=WINDOW_SECONDS):
    """
    Classify every chunk for one arm, then aggregate to a CSV.
    Skips a chunk if its per-chunk result JSON already exists (idempotent),
    so re-runs make no new Gemini calls.
    """
    cfg = ARM_CONFIG[arm_name]

    print(f"\n=== Classifying arm: {arm_name} ===")
    for i in range(max_chunks):
        chunk_path = os.path.join(chunks_dir, f"chunk_{i:03d}_{cfg['suffix']}.{cfg['ext']}")
        if not os.path.exists(chunk_path):
            print(f"Chunk {i} not found, skipping")
            continue
        result_json = os.path.join(results_dir, f"chunk_{i:03d}_result.json")
        if os.path.exists(result_json):
            print(f"Chunk {i} already classified, skipping")
            continue
        try:
            cfg["fn"](
                chunk_path,
                chunk_index=i,
                output_dir=results_dir,
                window_start=i * window_seconds,
                window_end=(i + 1) * window_seconds,
            )
        except Exception as e:
            # Don't crash the whole run on one bad chunk — log and skip it.
            print(f"  [ERROR] chunk {i} arm {arm_name} failed: {e}")
            continue

    print(f"\n=== Aggregating arm: {arm_name} ===")
    aggregate_results(
        chunks_dir=results_dir,
        output_csv=output_csv,
        lecture_id=lecture_id,
        arm=arm_name,
    )
    return output_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=list(ARM_CONFIG) + ["all"], default="multimodal")
    parser.add_argument("--max-chunks", type=int, default=MAX_CHUNKS)
    args = parser.parse_args()

    print("=== STEP 1: Chunking ===")
    chunk_video(
        input_path=LECTURE_PATH,
        output_dir=CHUNKS_DIR,
        window_seconds=WINDOW_SECONDS,
    )

    arms_to_run = list(ARM_CONFIG) if args.arm == "all" else [args.arm]
    output_csvs = {}
    for arm_name in arms_to_run:
        output_csvs[arm_name] = run_arm(
            arm_name,
            chunks_dir=CHUNKS_DIR,
            results_dir=f"output/pipeline_test/results_{arm_name}",
            output_csv=f"output/pipeline_test/results_{arm_name}.csv",
            lecture_id=LECTURE_ID,
            max_chunks=args.max_chunks,
        )

    print(f"\nDone! Results: {output_csvs}")

    # Dashboard only makes sense for a single arm's results
    if len(arms_to_run) == 1:
        from report.dashboard import generate_dashboard
        print("\n=== Dashboard ===")
        generate_dashboard(
            results_csv=output_csvs[arms_to_run[0]],
            output_html=f"output/pipeline_test/dashboard_{arms_to_run[0]}.html",
        )
