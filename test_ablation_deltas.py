"""
Acceptance tests for compute_ablation_deltas.py (no API calls, no cost).

    uv run python test_ablation_deltas.py
"""
import os
import tempfile

import pandas as pd

from compute_ablation_deltas import compute_deltas, delta
from validate.convert_copus_sheet import INSTRUCTOR_CODES

# --- Sign convention: negative = ablating hurt ---
assert delta(0.4, 0.9) == -0.5      # arm worse than multimodal
assert delta(0.9, 0.4) == 0.5       # arm better: that modality was noise
assert delta("N/A", 0.9) == "N/A"   # undefined on the ablated arm
assert delta(0.4, "N/A") == "N/A"   # undefined on multimodal
assert delta(0.4, 0.9, sign="multimodal-minus-arm") == 0.5
print("ok  delta signs: arm worse -> negative, arm better -> positive, N/A propagates")

# --- End-to-end from a comparison table ---
tmp = tempfile.mkdtemp()
rows = []
for code in INSTRUCTOR_CODES:
    rows.append({
        "professor_id": "professor_9", "code": code,
        "n_total_windows": 24, "n_human_marked": 0, "n_codes_human_observed": "",
        "n_ai_marked_multimodal": 0, "n_ai_marked_vision": 0,
        "n_ai_marked_audio": 0, "n_ai_marked_transcript": 0,
        "multimodal_kappa": "N/A", "vision_only_kappa": "N/A",
        "audio_only_kappa": "N/A", "transcript_only_kappa": "N/A",
        "multimodal_ac1": "N/A", "vision_only_ac1": "N/A",
        "audio_only_ac1": "N/A", "transcript_only_ac1": "N/A",
    })
by_code = {r["code"]: r for r in rows}
# Lec: vision much worse (video alone loses the spoken cue), audio identical.
by_code["Lec"].update(n_human_marked=23, multimodal_kappa=1.0, vision_only_kappa=-0.043,
                      audio_only_kappa=1.0, transcript_only_kappa=0.647,
                      multimodal_ac1=1.0, vision_only_ac1=0.909,
                      audio_only_ac1=1.0, transcript_only_ac1=0.953)
# D/V: audio BETTER than multimodal -> positive delta (video was noise here).
by_code["D/V"].update(n_human_marked=5, multimodal_kappa=0.56, audio_only_kappa=0.864,
                      vision_only_kappa=0.417, transcript_only_kappa=0.647,
                      multimodal_ac1=0.733, audio_only_ac1=0.94,
                      vision_only_ac1=0.462, transcript_only_ac1=0.807)
# O: human never marked it, both arms kappa 0.0 off false positives only.
by_code["O"].update(n_human_marked=0, multimodal_kappa=0.0, audio_only_kappa=0.0,
                    multimodal_ac1=0.957, audio_only_ac1=0.957)
# Summary rows must be skipped, not treated as codes.
rows.append({"professor_id": "professor_9", "code": "codes_clearing_kappa_0.7",
             "multimodal_kappa": 2, "audio_only_kappa": 3})
rows.append({"professor_id": "professor_9", "code": "codes_clearing_ac1_0.7",
             "multimodal_ac1": 4, "audio_only_ac1": 4})

table = os.path.join(tmp, "comparison_table.csv")
pd.DataFrame(rows).to_csv(table, index=False)
df = compute_deltas(table)

assert list(df["code"]) == INSTRUCTOR_CODES, list(df["code"])
print("ok  all 12 codes in protocol order; summary rows dropped")

d = df.set_index("code")
assert d.loc["Lec", "delta_kappa_vision"] == round(-0.043 - 1.0, 3) == -1.043
assert d.loc["Lec", "delta_kappa_audio"] == 0.0
assert d.loc["Lec", "delta_ac1_transcript"] == round(0.953 - 1.0, 3) == -0.047
print(f"ok  Lec: vision {d.loc['Lec', 'delta_kappa_vision']} (ablating hurt), "
      f"audio {d.loc['Lec', 'delta_kappa_audio']} (no change)")

assert d.loc["D/V", "delta_kappa_audio"] == round(0.864 - 0.56, 3) == 0.304
print(f"ok  D/V: audio {d.loc['D/V', 'delta_kappa_audio']} > 0 — video was noise for this code")

assert d.loc["CQ", "delta_kappa_audio"] == "N/A"
# A code the human never marked is N/A even though both arms have numeric kappa 0.0.
assert d.loc["O", "delta_kappa_audio"] == "N/A", d.loc["O"]
print("ok  unobserved codes are N/A, not 0.0 — including O (kappa 0.0 from false positives)")

for col in ["professor_id", "n_total_windows", "n_human_marked",
            "n_ai_marked_multimodal", "delta_kappa_audio", "delta_kappa_vision",
            "delta_kappa_transcript", "delta_ac1_audio", "delta_ac1_vision",
            "delta_ac1_transcript"]:
    assert col in df.columns, col
assert os.path.exists(os.path.join(tmp, "ablation_deltas.csv"))
print("ok  ablation_deltas.csv written with N per row and all 6 delta columns")

# --- A pre-AC1 / single-arm table must fail loudly, not emit a silent blank table ---
old = os.path.join(tmp, "old_comparison_table.csv")
pd.DataFrame([{"code": "Lec", "multimodal_kappa": 0.5}]).to_csv(old, index=False)
try:
    compute_deltas(old)
    raise AssertionError("expected SystemExit for a table with no AC1 columns")
except SystemExit as e:
    assert "multimodal_ac1" in str(e), e
print("ok  a table with no AC1 columns exits with an explanatory error")

print("\nAll ablation delta tests passed.")
