"""
Acceptance tests for per-window retries in run_arm (no API calls, no cost).

Fault injection: ARM_CONFIG's classify function is swapped for one that fails a
set number of times per chunk, so the retry policy is exercised without Vertex.
Also checks that parallel workers give the same results as a sequential run,
actually overlap, and stop on cancel.

    uv run python -m tests.test_retry
"""
import json
import os
import tempfile
import threading
import time

import classify.runner as runner
from classify.runner import MAX_ATTEMPTS, run_arm

assert MAX_ATTEMPTS == 3, MAX_ATTEMPTS
runner.RETRY_BASE_SECONDS = 0  # no backoff sleeps in tests


def make_fake_arm(failures_per_chunk, delay=0.0):
    """A classify fn that fails `failures_per_chunk[i]` times for chunk i."""
    calls = {}
    lock = threading.Lock()

    def fn(path, chunk_index, output_dir, window_start, window_end, model):
        with lock:
            calls[chunk_index] = calls.get(chunk_index, 0) + 1
            n = calls[chunk_index]
        time.sleep(delay)
        if n <= failures_per_chunk.get(chunk_index, 0):
            raise RuntimeError(f"injected failure {n}")
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"chunk_{chunk_index:03d}_result.json"), "w") as f:
            json.dump({"chunk_index": chunk_index, "window_start": window_start,
                       "window_end": window_end, "model": model,
                       "codes_present": ["Lec"], "reasoning": "fake"}, f)
        return {}

    return fn, calls


def setup(failures, n_chunks=4, delay=0.0):
    tmp = tempfile.mkdtemp()
    chunks = os.path.join(tmp, "chunks")
    os.makedirs(chunks)
    for i in range(n_chunks):
        for suffix, ext in (("full", "mp4"), ("muted", "mp4"), ("audio", "mp3")):
            with open(os.path.join(chunks, f"chunk_{i:03d}_{suffix}.{ext}"), "w") as f:
                f.write("x")  # non-empty: chunks_missing treats 0-byte as absent
    fn, calls = make_fake_arm(failures, delay)
    runner.ARM_CONFIG["multimodal"] = {"suffix": "full", "ext": "mp4", "fn": fn}
    return tmp, chunks, calls


def run(tmp, chunks, windows=(0, 1, 2, 3), **kwargs):
    return run_arm("multimodal", chunks_dir=chunks,
                   results_dir=os.path.join(tmp, "results_multimodal"),
                   output_csv=os.path.join(tmp, "results_multimodal.csv"),
                   lecture_id="L1", window_indices=list(windows),
                   professor_id="professor_9", **kwargs)


# --- Chunk 1 fails twice then succeeds; chunk 3 fails every attempt ---
tmp, chunks, calls = setup({1: 2, 3: 99})
res = run(tmp, chunks)

assert calls[0] == 1, calls          # clean chunk: one call
assert calls[1] == 3, calls          # two failures, succeeds on attempt 3
assert calls[3] == MAX_ATTEMPTS, calls   # never succeeds: stops at 3, no infinite retry
assert res["failed"] == [3], res["failed"]
assert res["processed"] == 3, res["processed"]
print(f"ok  attempts per chunk {dict(sorted(calls.items()))}: recovers on 3, "
      f"gives up after {MAX_ATTEMPTS}")

r = res["retries"]
assert r["max_attempts"] == MAX_ATTEMPTS
assert r["recovered"] == {1: 3}, r["recovered"]
assert r["failed"] == [3], r["failed"]
# 2 extra attempts to recover chunk 1 + 2 wasted extra attempts on chunk 3
assert r["retry_attempts"] == 4, r["retry_attempts"]
print(f"ok  retry stats: {r['retry_attempts']} extra attempts, "
      f"recovered {r['recovered']}, still failed {r['failed']}")

# --- A clean run reports no retries at all ---
tmp, chunks, calls = setup({})
res = run(tmp, chunks)
assert res["retries"]["recovered"] == {} and res["retries"]["retry_attempts"] == 0
assert res["failed"] == [] and res["processed"] == 4
print("ok  clean run: 0 extra attempts, 0 failures")

# --- Retry accounting is identical sequentially and in parallel ---
for workers in (1, 4):
    tmp, chunks, calls = setup({1: 2, 3: 99})
    res = run(tmp, chunks, workers=workers)
    assert res["failed"] == [3] and res["retries"]["recovered"] == {1: 3}, res
    assert res["retries"]["retry_attempts"] == 4, res["retries"]
print("ok  retry stats identical with workers=1 and workers=4")

# --- Parallel windows overlap and produce the same CSV as a sequential run ---
N, DELAY = 8, 0.3
csvs, elapsed = {}, {}
for workers in (1, 4):
    tmp, chunks, calls = setup({}, n_chunks=N, delay=DELAY)
    t0 = time.monotonic()
    res = run(tmp, chunks, windows=range(N), workers=workers)
    elapsed[workers] = time.monotonic() - t0
    assert res["processed"] == N and res["failed"] == [], res
    with open(res["csv"]) as f:
        csvs[workers] = f.read()
assert csvs[1] == csvs[4], "parallel run changed the aggregated CSV"
assert elapsed[1] >= N * DELAY * 0.9, elapsed
assert elapsed[4] < elapsed[1] / 2, elapsed
print(f"ok  {N} windows: {elapsed[1]:.2f}s sequential vs {elapsed[4]:.2f}s with "
      f"4 workers, identical CSV")

# --- Progress callback fires once per window ---
ticks = []
tmp, chunks, calls = setup({})
run(tmp, chunks, workers=2, on_window_done=lambda: ticks.append(1))
assert len(ticks) == 4, ticks
print("ok  on_window_done called once per window")

# --- Cancelling stops windows that have not started ---
cancel = threading.Event()
tmp, chunks, calls = setup({}, n_chunks=N, delay=0.2)
threading.Timer(0.1, cancel.set).start()
res = run(tmp, chunks, windows=range(N), workers=2, cancel_event=cancel)
assert res["processed"] == 2, res   # the two in flight finish, the rest skip
assert sum(calls.values()) == 2, calls
print(f"ok  cancel: {res['processed']}/{N} windows ran, the rest were skipped")

print("\nAll retry tests passed.")
