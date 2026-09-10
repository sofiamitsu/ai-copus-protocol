"""
Acceptance tests for utils.sparse_windows (no API calls, no cost).

    uv run python test_sparse_windows.py
"""
import math

from utils.sparse_windows import compute_sparse_windows, plan_sparse_windows


def blocks_of(indices):
    """Split a sorted index list into runs of consecutive integers."""
    runs = []
    for i in indices:
        if runs and i == runs[-1][-1] + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    return runs


def check_wellformed(indices, duration_min, window_size_min):
    assert indices == sorted(set(indices)), f"not sorted/unique: {indices}"
    total = math.ceil(duration_min / window_size_min)
    assert indices[-1] < total, f"index {indices[-1]} past end ({total} chunks)"
    assert indices[0] >= 0


# --- 4-block scheme, D=120 (real-9 typical) ---
idx = compute_sparse_windows(120, 2)
check_wellformed(idx, 120, 2)
assert len(idx) == 32, len(idx)
runs = blocks_of(idx)
assert len(runs) == 4, runs
assert all(len(r) == 8 for r in runs), runs
assert idx == (list(range(0, 8)) + list(range(17, 25))
               + list(range(35, 43)) + list(range(52, 60))), idx
print("ok  120 min -> 4-block, 32 chunks, matches worked example")

# --- 3-block scheme, D=60 (golden typical) ---
idx = compute_sparse_windows(60, 2)
check_wellformed(idx, 60, 2)
assert len(idx) == 24, len(idx)
runs = blocks_of(idx)
assert len(runs) == 3, runs
assert all(len(r) == 8 for r in runs), runs
assert idx == (list(range(0, 8)) + list(range(11, 19))
               + list(range(22, 30))), idx
print("ok   60 min -> 3-block, 24 chunks, matches worked example")

# --- Fallback, D=45 ---
idx = compute_sparse_windows(45, 2)
check_wellformed(idx, 45, 2)
assert idx == list(range(23)), idx  # ceil(45/2) = 23, all chunks
_, scheme, _ = plan_sparse_windows(45, 2)
assert scheme == "full", scheme
print("ok   45 min -> fallback, all 23 chunks (warning printed above)")

# --- Boundary: 75 uses 4-block, 74.99 uses 3-block ---
idx75, scheme75, _ = plan_sparse_windows(75, 2)
assert scheme75 == "4-block", scheme75
check_wellformed(idx75, 75, 2)
assert len(idx75) == 32, len(idx75)
# ceil(75/2) = 38 chunks; chunk 37 is the partial 74-75 min window.
assert idx75 == (list(range(0, 8)) + list(range(10, 18))
                 + list(range(20, 28)) + list(range(30, 38))), idx75
print("ok   75.00 min -> 4-block (boundary), matches worked example")

idx74, scheme74, _ = plan_sparse_windows(74.99, 2)
assert scheme74 == "3-block", scheme74
check_wellformed(idx74, 74.99, 2)
assert len(idx74) == 24, len(idx74)
print("ok   74.99 min -> 3-block (boundary)")

# --- Boundary: 48 is the 3-block floor, 47.99 falls back ---
_, scheme48, _ = plan_sparse_windows(48, 2)
assert scheme48 == "3-block", scheme48
_, scheme47, _ = plan_sparse_windows(47.99, 2)
assert scheme47 == "full", scheme47
print("ok   48.00 / 47.99 min -> 3-block / fallback (boundary)")

# --- No overlap or duplication anywhere in the operating range ---
d = 48.0
while d <= 130.0:
    indices, scheme, blocks = plan_sparse_windows(d, 2)
    check_wellformed(indices, d, 2)
    expected = 32 if scheme == "4-block" else 24
    assert len(indices) == expected, (d, scheme, len(indices))
    for (_, _, a_end), (_, b_start, _) in zip(blocks, blocks[1:]):
        assert b_start > a_end, f"blocks overlap at {d} min: {blocks}"
    d = round(d + 0.01, 2)
print("ok   48-130 min swept in 0.01 steps: no overlaps, no duplicates")

# --- lecture_001, the real golden lecture (2899.94s = 48.33 min) ---
idx, scheme, _ = plan_sparse_windows(2899.940136 / 60, 2)
assert scheme == "3-block", scheme
print(f"ok   lecture_001 (48.33 min) -> {scheme}, {len(idx)} chunks: "
      f"{','.join(str(i) for i in idx)}")

print("\nAll sparse-window acceptance tests passed.")
