"""
Arm runner: classify one lecture's windows for one ablation arm, then aggregate.

Windows are classified concurrently on a thread pool. Almost all of a window's
time is spent waiting on Vertex, so running them side by side is where a
multi-lecture, multi-arm run gets its speed. The number of requests actually in
flight is capped globally in classify.classifier (set_max_concurrency), so arms
and lectures can also run in parallel without exceeding quota.

Each window is independent -- its own chunk, its own JSON, pinned decoding -- so
the order windows finish in has no effect on the results.
"""
import json
import os
import random
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from aggregate.aggregator import aggregate_results
from classify.classifier import (
    DEFAULT_MODEL,
    GENERATION_SEED,
    GENERATION_TEMPERATURE,
    classify_chunk_audio_only,
    classify_chunk_multimodal,
    classify_chunk_transcript_only,
    classify_chunk_vision_only,
)

WINDOW_SECONDS = 120

# Attempts per window per arm before it is recorded as failed. A window that
# stays failed is dropped from every arm's comparison, so it is worth spending
# two extra calls here rather than losing the window for all four arms.
MAX_ATTEMPTS = 3

# Base delay before a retry, doubled on each further attempt (plus jitter).
# Under parallel load most failures are rate limits (429), and an instant retry
# just hits the same limit again.
RETRY_BASE_SECONDS = float(os.environ.get("GEMINI_RETRY_BASE_SECONDS", "4"))

# Default number of windows classified at once within one arm.
DEFAULT_WORKERS = 8

class Cancelled(Exception):
    """Raised by callers once a run's cancel_event has stopped classification."""


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


def _is_rate_limit(error):
    text = str(error)
    return "429" in text or "RESOURCE_EXHAUSTED" in text


def _retry_delay(attempt, error):
    """Exponential backoff with jitter; rate limits back off twice as long."""
    delay = RETRY_BASE_SECONDS * (2 ** (attempt - 1))
    if _is_rate_limit(error):
        delay *= 2
    return delay * (0.75 + random.random() / 2)


def _classify_window(fn, chunk_path, i, results_dir, window_seconds, model,
                     max_attempts, tag, cancel_event):
    """
    Classify one window with retries. Returns (status, attempts) where status is
    "ok", "failed" or "cancelled".
    """
    for attempt in range(1, max_attempts + 1):
        if cancel_event is not None and cancel_event.is_set():
            return "cancelled", attempt - 1
        try:
            result = fn(
                chunk_path,
                chunk_index=i,
                output_dir=results_dir,
                window_start=i * window_seconds,
                window_end=(i + 1) * window_seconds,
                model=model,
            )
            codes = (result or {}).get("codes_present")
            note = f" (attempt {attempt}/{max_attempts})" if attempt > 1 else ""
            print(f"  {tag} chunk {i}: {codes}{note}")
            return "ok", attempt
        except Exception as e:
            if attempt < max_attempts:
                delay = _retry_delay(attempt, e)
                print(f"  {tag} [retry] chunk {i} attempt {attempt}/{max_attempts} "
                      f"failed: {e} -- retrying in {delay:.0f}s")
                # Wait, but wake immediately if the run is cancelled.
                if cancel_event is not None:
                    cancel_event.wait(delay)
                else:
                    time.sleep(delay)
                continue
            print(f"  {tag} [ERROR] chunk {i} failed {max_attempts} times: {e}")
            return "failed", attempt


def run_arm(arm_name, chunks_dir, results_dir, output_csv, lecture_id,
            max_chunks=None, window_seconds=WINDOW_SECONDS,
            window_indices=None, model=DEFAULT_MODEL, professor_id="",
            max_attempts=MAX_ATTEMPTS, workers=DEFAULT_WORKERS,
            cancel_event=None, on_window_done=None):
    """
    Classify the requested chunks for one arm, then aggregate to a CSV.
    Skips a chunk if its per-chunk result JSON already exists (idempotent),
    so re-runs make no new Gemini calls.

    Up to `workers` windows are classified at once. A chunk that raises is
    retried up to max_attempts times with backoff (transient Vertex errors, rate
    limits and timeouts are the common case) and only then recorded as failed --
    it never disappears silently. A window that no arm can fill is dropped from
    EVERY arm at comparison time, so retries are what keep the paired per-arm
    window set intact.

    cancel_event (threading.Event): once set, windows not yet started are
    skipped. on_window_done(): called once per window that is resolved (done,
    cached, failed or missing) -- used for progress bars.

    Returns a dict: {"csv", "indices", "processed", "missing", "failed", "retries"}
    """
    cfg = ARM_CONFIG[arm_name]
    tag = f"[{lecture_id}/{arm_name}]"
    indices = resolve_window_indices(window_indices, max_chunks, chunks_dir)
    stale, missing, todo = [], [], []

    def done():
        if on_window_done is not None:
            on_window_done()

    print(f"\n=== {tag} classifying {len(indices)} windows "
          f"(model={model}, workers={workers}) ===")
    for i in indices:
        chunk_path = os.path.join(chunks_dir, f"chunk_{i:03d}_{cfg['suffix']}.{cfg['ext']}")
        if not os.path.exists(chunk_path):
            print(f"  {tag} chunk {i} not found, skipping")
            missing.append(i)
            done()
            continue
        result_json = os.path.join(results_dir, f"chunk_{i:03d}_result.json")
        if os.path.exists(result_json):
            stale.extend(_stale_settings(result_json, model))
            done()
            continue
        todo.append((i, chunk_path))
    cached = len(indices) - len(missing) - len(todo)
    if cached:
        print(f"  {tag} {cached} window(s) already classified, reusing")

    outcomes = {}  # window -> (status, attempts)
    lock = threading.Lock()

    def work(item):
        i, chunk_path = item
        outcome = _classify_window(cfg["fn"], chunk_path, i, results_dir,
                                   window_seconds, model, max_attempts, tag,
                                   cancel_event)
        with lock:
            outcomes[i] = outcome
        done()

    if todo:
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(todo))),
                                thread_name_prefix=f"{arm_name}") as pool:
            # list() re-raises anything unexpected from a worker.
            list(pool.map(work, todo))

    failed = sorted(i for i, (s, _) in outcomes.items() if s == "failed")
    cancelled = sorted(i for i, (s, _) in outcomes.items() if s == "cancelled")
    # {window: winning attempt} for windows that needed more than one.
    recovered = {i: a for i, (s, a) in sorted(outcomes.items()) if s == "ok" and a > 1}

    if failed:
        print(f"\n[ERROR] {tag} {len(failed)} chunk(s) failed after "
              f"{max_attempts} attempts: {failed}. Every arm is scored on the "
              f"windows ALL arms have, so these windows will be dropped from the "
              f"comparison for every arm -- re-run to fill them.")
    if cancelled:
        print(f"[warn] {tag} cancelled before {len(cancelled)} window(s) ran")
    if missing:
        print(f"[warn] {tag} {len(missing)} chunk(s) had no media on disk: {missing}")
    if stale:
        for reason, count in sorted(Counter(stale).items()):
            print(f"[warn] {tag} {count} cached chunk(s) were produced "
                  f"with {reason}. Cached results are REUSED as-is, so this run "
                  f"mixes settings. Delete {results_dir} and re-run if these "
                  f"numbers are going in the thesis.")

    print(f"\n=== {tag} aggregating ===")
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
        "recovered": recovered,
        # Extra attempts spent beyond the first, across all windows.
        "retry_attempts": sum(max(a - 1, 0) for s, a in outcomes.values()
                              if s in ("ok", "failed")),
        "failed": list(failed),
    }
    if recovered:
        print(f"{tag} {len(recovered)} window(s) needed a retry "
              f"(succeeded on attempt {sorted(set(recovered.values()))})")
    print(f"{tag} {processed}/{len(indices)} windows have results")
    return {
        "csv": output_csv,
        "indices": indices,
        "processed": processed,
        "missing": missing,
        "failed": failed,
        "retries": retries,
    }


def run_arms(arms, workers=DEFAULT_WORKERS, **kwargs):
    """
    Run several arms of one lecture side by side. `kwargs` are shared run_arm
    arguments except the per-arm results_dir/output_csv, which are derived from
    `lecture_dir`. Returns {arm: run_arm result} in the order given.
    """
    lecture_dir = kwargs.pop("lecture_dir")

    def one(arm_name):
        return run_arm(
            arm_name,
            results_dir=os.path.join(lecture_dir, f"results_{arm_name}"),
            output_csv=os.path.join(lecture_dir, f"results_{arm_name}.csv"),
            workers=workers,
            **kwargs,
        )

    with ThreadPoolExecutor(max_workers=max(1, len(arms)),
                            thread_name_prefix="arm") as pool:
        results = list(pool.map(one, arms))
    return dict(zip(arms, results))
