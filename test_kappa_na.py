"""
Acceptance tests for the kappa N/A fix and the MEAN row (no API calls, no cost).

    uv run python test_kappa_na.py
"""
from sklearn.metrics import cohen_kappa_score

from validate.validator import kappa_by_code, mean_kappa, MEAN_ROW_LABEL


def windows(code_lists):
    return [{"window_index": i, "codes": c} for i, c in enumerate(code_lists)]


# --- One-sided use is a REAL zero, not N/A ---
# 6 windows. Both rate Lec with real variation; only the AI ever says FUp.
# The AI firing a code the human never used is a genuine over-fire: kappa is
# defined (0.0) and must be reported, not blanked out as N/A.
human = windows([["Lec"], ["Lec"], [], ["Lec"], [], ["Lec"]])
ai     = windows([["Lec"], [], ["Lec"], ["Lec"], [], ["Lec", "FUp"]])

rows = {r["code"]: r for r in kappa_by_code(human, ai)}

# Sanity: sklearn really does define kappa here, and it is 0.0.
raw = cohen_kappa_score([0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1])
assert raw == 0.0, raw
print(f"ok  raw cohen_kappa_score with one constant rater = {raw} (defined, not undefined)")

assert rows["FUp"]["kappa"] == 0.0, rows["FUp"]
assert rows["FUp"]["human_positive_windows"] == 0
assert rows["FUp"]["ai_positive_windows"] == 1
print("ok  FUp -> 0.0 (human never used it, AI did) -- over-fire is visible")

assert isinstance(rows["Lec"]["kappa"], float), rows["Lec"]
print(f"ok  Lec -> {rows['Lec']['kappa']} (real variation both sides, still numeric)")

# --- Mirror case: human used a code the AI never emitted ---
human2 = windows([["Lec", "O"], ["Lec"], [], ["Lec"]])
ai2 = windows([["Lec"], ["Lec"], [], []])
rows2 = {r["code"]: r for r in kappa_by_code(human2, ai2)}
assert rows2["O"]["kappa"] == 0.0, rows2["O"]
print("ok  O -> 0.0 (AI never used it) -- symmetric, under-fire also visible")

# --- N/A is reserved for codes NEITHER rater used ---
# "MG" appears nowhere, so it is not even in all_codes; construct the case that
# does reach the guard: a code present in one window set but only as an empty
# marker is impossible, so verify via a code both raters leave at zero across
# the shared window overlap.
human3 = windows([["Lec"], ["Lec"], ["MG"]])
ai3 = [{"window_index": 0, "codes": ["Lec"]},
       {"window_index": 1, "codes": ["Lec"]}]
rows3 = {r["code"]: r for r in kappa_by_code(human3, ai3)}
# MG is in all_codes (window 2 of the human set) but window 2 is not shared,
# so within the overlap neither rater used it -> genuinely undefined.
assert rows3["MG"]["kappa"] == "N/A", rows3["MG"]
assert rows3["MG"]["human_positive_windows"] == 0
assert rows3["MG"]["ai_positive_windows"] == 0
print("ok  MG -> N/A (neither rater used it in the shared overlap)")

# --- Both raters mark every window: no variation, still N/A ---
human4 = windows([["Lec"], ["Lec"], ["Lec"]])
ai4 = windows([["Lec"], ["Lec"], ["Lec"]])
rows4 = {r["code"]: r for r in kappa_by_code(human4, ai4)}
assert rows4["Lec"]["kappa"] == "N/A", rows4["Lec"]
print("ok  Lec -> N/A when both raters mark all windows (no variation)")

# --- MEAN skips N/A entirely ---
assert mean_kappa([0.74, "N/A", 0.30]) == 0.52, mean_kappa([0.74, "N/A", 0.30])
assert mean_kappa(["N/A", "N/A"]) == "N/A"
assert mean_kappa([0.5, "ERR: boom"]) == 0.5
print("ok  mean_kappa skips N/A and ERR rows")

# The bug this fixes: averaging the old 0.00s drags the mean down.
old_way = round((0.74 + 0.0 + 0.30) / 3, 3)
new_way = mean_kappa([0.74, "N/A", 0.30])
assert new_way > old_way
print(f"ok  MEAN with N/A excluded = {new_way} vs {old_way} counting N/A as 0.00")

# --- No overlapping windows at all ---
assert all(r["kappa"] == "N/A"
           for r in kappa_by_code(windows([["Lec"]]),
                                  [{"window_index": 99, "codes": ["Lec"]}]))
print("ok  no shared windows -> N/A, not a crash")

print(f"\nAll kappa N/A + {MEAN_ROW_LABEL} tests passed.")
