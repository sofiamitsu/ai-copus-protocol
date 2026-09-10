import argparse
import os
from dotenv import load_dotenv
load_dotenv()

from chunk.chunker import chunk_video
from classify.classifier import (
    DEFAULT_MODEL,
    classify_chunk_multimodal,
    classify_chunk_vision_only,
    classify_chunk_audio_only,
    classify_chunk_transcript_only,
)
from aggregate.aggregator import aggregate_results

# __main__ smoke-run defaults only; run.py drives real runs through run_arm().
DEFAULT_LECTURE_PATH = "videoplayback.mp4"
DEFAULT_OUTPUT_DIR = "output/pipeline_test"
WINDOW_SECONDS = 120
MAX_CHUNKS = 24  # __main__ smoke-run default only; real runs use --windows


def lecture_id_from_path(lecture_path):
    """
    Derive a lecture identifier from the input filename stem: mehran.mp4 ->
    "mehran". Hardcoding one id (it used to be "lecture_001" for every run) made
    every smoke run write over the last one and stamped the wrong lecture_id into
    every CSV row.
    """
    return os.path.splitext(os.path.basename(lecture_path))[0]


ARM_CONFIG = {
    "multimodal":     {"suffix": "full", "ext": "mp4", "fn": classify_chunk_multimodal},
    "vision_only":    {"suffix": "muted", "ext": "mp4", "fn": classify_chunk_vision_only},
    "audio_only":     {"suffix": "audio", "ext": "mp3", "fn": classify_chunk_audio_only},
    "transcript_only": {"suffix": "audio", "ext": "mp3", "fn": classify_chunk_transcript_only},
}


def resolve_window_indices(window_indices=None, max_chunks=None, chunks_dir=None):
    """
    The set of chunk indices an arm should process.

    Exactly one of window_indices (explicit sparse list) or max_chunks
    (range(max_chunks)) drives this; with neither, every chunk on disk is used.
    """
    if window_indices is not None:
        return sorted(set(window_indices))
    if max_chunks is not None:
        return list(range(max_chunks))
    n = len([f for f in os.listdir(chunks_dir) if f.endswith("_full.mp4")])
    return list(range(n))


def run_arm(arm_name, chunks_dir, results_dir, output_csv, lecture_id,
            max_chunks=None, window_seconds=WINDOW_SECONDS,
            window_indices=None, model=DEFAULT_MODEL):
    """
    Classify the requested chunks for one arm, then aggregate to a CSV.
    Skips a chunk if its per-chunk result JSON already exists (idempotent),
    so re-runs make no new Gemini calls.

    A chunk that raises is retried once and then recorded as failed -- it no
    longer disappears silently. Returns a dict:
      {"csv", "indices", "processed", "missing", "failed"}
    """
    cfg = ARM_CONFIG[arm_name]
    indices = resolve_window_indices(window_indices, max_chunks, chunks_dir)

    print(f"\n=== Classifying arm: {arm_name} "
          f"({len(indices)} windows, model={model}) ===")
    failed, missing = [], []
    for i in indices:
        chunk_path = os.path.join(chunks_dir, f"chunk_{i:03d}_{cfg['suffix']}.{cfg['ext']}")
        if not os.path.exists(chunk_path):
            print(f"Chunk {i} not found, skipping")
            missing.append(i)
            continue
        result_json = os.path.join(results_dir, f"chunk_{i:03d}_result.json")
        if os.path.exists(result_json):
            print(f"Chunk {i} already classified, skipping")
            continue
        # One retry, then record the failure rather than swallowing it.
        for attempt in (1, 2):
            try:
                cfg["fn"](
                    chunk_path,
                    chunk_index=i,
                    output_dir=results_dir,
                    window_start=i * window_seconds,
                    window_end=(i + 1) * window_seconds,
                    model=model,
                )
                break
            except Exception as e:
                if attempt == 1:
                    print(f"  [retry] chunk {i} arm {arm_name} failed: {e}")
                    continue
                print(f"  [ERROR] chunk {i} arm {arm_name} failed twice: {e}")
                failed.append(i)

    if failed:
        print(f"\n[ERROR] arm {arm_name}: {len(failed)} chunk(s) failed after retry: "
              f"{failed}")
    if missing:
        print(f"[warn] arm {arm_name}: {len(missing)} chunk(s) had no media on disk: "
              f"{missing}")

    print(f"\n=== Aggregating arm: {arm_name} ===")
    aggregate_results(
        results_dir=results_dir,
        output_csv=output_csv,
        lecture_id=lecture_id,
        arm=arm_name,
    )
    processed = len([i for i in indices
                     if os.path.exists(os.path.join(results_dir,
                                                    f"chunk_{i:03d}_result.json"))])
    print(f"Arm {arm_name}: {processed}/{len(indices)} windows have results")
    return {
        "csv": output_csv,
        "indices": indices,
        "processed": processed,
        "missing": missing,
        "failed": failed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=list(ARM_CONFIG) + ["all"], default="multimodal")
    parser.add_argument("--max-chunks", type=int, default=MAX_CHUNKS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--lecture", default=DEFAULT_LECTURE_PATH,
                        help="Lecture .mp4 to smoke-run")
    parser.add_argument("--lecture-id", default=None,
                        help="Lecture identifier; defaults to the --lecture filename stem")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    lecture_id = args.lecture_id or lecture_id_from_path(args.lecture)
    # Every output is scoped under the lecture id, so smoke-running a second
    # lecture no longer overwrites the first one's chunks, results, or CSVs.
    lecture_dir = os.path.join(args.output_dir, lecture_id)
    chunks_dir = os.path.join(lecture_dir, "chunks")
    print(f"Lecture: {args.lecture}  (lecture_id={lecture_id})")
    print(f"Output:  {lecture_dir}")

    print("\n=== STEP 1: Chunking ===")
    chunk_video(
        input_path=args.lecture,
        output_dir=chunks_dir,
        window_seconds=WINDOW_SECONDS,
    )

    arms_to_run = list(ARM_CONFIG) if args.arm == "all" else [args.arm]
    output_csvs = {}
    for arm_name in arms_to_run:
        output_csvs[arm_name] = run_arm(
            arm_name,
            chunks_dir=chunks_dir,
            results_dir=os.path.join(lecture_dir, f"results_{arm_name}"),
            output_csv=os.path.join(lecture_dir, f"results_{arm_name}.csv"),
            lecture_id=lecture_id,
            max_chunks=args.max_chunks,
            model=args.model,
        )["csv"]

    print(f"\nDone! Results: {output_csvs}")

    # Dashboard only makes sense for a single arm's results
    if len(arms_to_run) == 1:
        from report.dashboard import generate_dashboard
        print("\n=== Dashboard ===")
        generate_dashboard(
            results_csv=output_csvs[arms_to_run[0]],
            output_html=os.path.join(lecture_dir,
                                     f"dashboard_{arms_to_run[0]}.html"),
        )
