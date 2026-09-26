import argparse
import json
import os
from collections import Counter
from dotenv import load_dotenv
load_dotenv()

from chunk.chunker import chunk_video
from classify.classifier import (
    DEFAULT_MODEL,
    GENERATION_SEED,
    GENERATION_TEMPERATURE,
    classify_chunk_multimodal,
    classify_chunk_vision_only,
    classify_chunk_audio_only,
    classify_chunk_transcript_only,
)
from aggregate.aggregator import aggregate_results
from utils.professor_ids import resolve_professor

# __main__ smoke-run defaults only; run.py drives real runs through run_arm().
DEFAULT_LECTURE_PATH = "videoplayback.mp4"
DEFAULT_OUTPUT_DIR = "output/pipeline_test"
WINDOW_SECONDS = 120
MAX_CHUNKS = 24  # __main__ smoke-run default only; real runs use --windows

# Attempts per window per arm before it is recorded as failed. A window that
# stays failed is dropped from every arm's comparison, so it is worth spending
# two extra calls here rather than losing the window for all four arms.
MAX_ATTEMPTS = 3


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


def _stale_settings(result_json, model):
    """
    Ways a cached per-chunk result disagrees with what this run would produce.

    run_arm reuses any chunk that already has a JSON, which is what makes reruns
    free -- but a JSON written before decoding was pinned (sampled at temperature
    1.0, or seeded differently, or from another model) is not comparable with a
    fresh one, and nothing about the CSV would show it.
    """
    try:
        with open(result_json) as f:
            data = json.load(f)
    except Exception:
        return []
    reasons = []
    cached_model = data.get("model")
    if cached_model and cached_model != model:
        reasons.append(f"model={cached_model} (this run uses {model})")
    # Pre-pinning JSONs have no temperature key at all: they were sampled.
    if "temperature" not in data:
        reasons.append("no recorded temperature (sampled, before decoding was pinned)")
    elif float(data["temperature"]) != float(GENERATION_TEMPERATURE):
        reasons.append(f"temperature={data['temperature']} "
                       f"(this run uses {GENERATION_TEMPERATURE})")
    elif int(data.get("seed", -1)) != int(GENERATION_SEED):
        reasons.append(f"seed={data.get('seed')} (this run uses {GENERATION_SEED})")
    return reasons


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
            window_indices=None, model=DEFAULT_MODEL, professor_id="",
            max_attempts=MAX_ATTEMPTS):
    """
    Classify the requested chunks for one arm, then aggregate to a CSV.
    Skips a chunk if its per-chunk result JSON already exists (idempotent),
    so re-runs make no new Gemini calls.

    A chunk that raises is retried up to max_attempts times (transient Vertex
    errors and timeouts are the common case) and only then recorded as failed --
    it no longer disappears silently. A window that no arm can fill is dropped
    from EVERY arm at comparison time, so retries are what keep the paired
    per-arm window set intact. Returns a dict:
      {"csv", "indices", "processed", "missing", "failed", "retries"}
    """
    cfg = ARM_CONFIG[arm_name]
    indices = resolve_window_indices(window_indices, max_chunks, chunks_dir)
    # {window: attempts spent before it finally failed}, {window: winning attempt}
    attempts_used, recovered = {}, {}
    stale = []

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
            stale.extend(_stale_settings(result_json, model))
            print(f"Chunk {i} already classified, skipping")
            continue
        # One retry, then record the failure rather than swallowing it.
        for attempt in range(1, max_attempts + 1):
            try:
                cfg["fn"](
                    chunk_path,
                    chunk_index=i,
                    output_dir=results_dir,
                    window_start=i * window_seconds,
                    window_end=(i + 1) * window_seconds,
                    model=model,
                )
                if attempt > 1:
                    recovered[i] = attempt
                    print(f"  [ok] chunk {i} arm {arm_name} succeeded on attempt "
                          f"{attempt}/{max_attempts}")
                break
            except Exception as e:
                attempts_used[i] = attempt
                if attempt < max_attempts:
                    print(f"  [retry] chunk {i} arm {arm_name} attempt "
                          f"{attempt}/{max_attempts} failed: {e}")
                    continue
                print(f"  [ERROR] chunk {i} arm {arm_name} failed "
                      f"{max_attempts} times: {e}")
                failed.append(i)

    if failed:
        print(f"\n[ERROR] arm {arm_name}: {len(failed)} chunk(s) failed after "
              f"{max_attempts} attempts: {failed}. Every arm is scored on the "
              f"windows ALL arms have, so these windows will be dropped from the "
              f"comparison for every arm -- re-run to fill them.")
    if missing:
        print(f"[warn] arm {arm_name}: {len(missing)} chunk(s) had no media on disk: "
              f"{missing}")
    if stale:
        for reason, count in sorted(Counter(stale).items()):
            print(f"[warn] arm {arm_name}: {count} cached chunk(s) were produced "
                  f"with {reason}. Cached results are REUSED as-is, so this run "
                  f"mixes settings. Delete {results_dir} and re-run if these "
                  f"numbers are going in the thesis.")

    print(f"\n=== Aggregating arm: {arm_name} ===")
    aggregate_results(
        results_dir=results_dir,
        output_csv=output_csv,
        lecture_id=lecture_id,
        arm=arm_name,
        professor_id=professor_id,
    )
    processed = len([i for i in indices
                     if os.path.exists(os.path.join(results_dir,
                                                    f"chunk_{i:03d}_result.json"))])
    retries = {
        "max_attempts": max_attempts,
        # Windows that needed more than one attempt but did succeed.
        "recovered": dict(sorted(recovered.items())),
        # Extra attempts spent beyond the first, across all windows.
        "retry_attempts": sum(a - 1 for a in recovered.values())
                          + sum(max(a - 1, 0) for i, a in attempts_used.items()
                                if i in failed),
        "failed": list(failed),
    }
    if recovered:
        print(f"Arm {arm_name}: {len(recovered)} window(s) needed a retry "
              f"(succeeded on attempt {sorted(set(recovered.values()))})")
    print(f"Arm {arm_name}: {processed}/{len(indices)} windows have results")
    return {
        "csv": output_csv,
        "indices": indices,
        "processed": processed,
        "missing": missing,
        "failed": failed,
        "retries": retries,
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
    professor_id, _ = resolve_professor(args.lecture)
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
            professor_id=professor_id,
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
