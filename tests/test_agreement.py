"""
Acceptance tests for per-code agreement: prevalence, % agreement, Gwet's AC1,
precision / recall, threshold-count summary rows, aggregate agreement rows, and
per_code_errors.csv (no API calls, no cost).

    uv run python -m tests.test_agreement

Optional cross-check against Gwet's own implementation, if irrCAC is importable:
    PYTHONPATH=/path/to/irrCAC uv run python -m tests.test_agreement
"""
import os
import tempfile

import pandas as pd

from validate.convert_copus_sheet import INSTRUCTOR_CODES
from validate.validator import (
    AC1_CLEAR_LABEL, AGGREGATE_LABELS, KAPPA_CLEAR_LABEL, POOLED_KAPPA_LABEL,
    RAW_AGREEMENT_LABEL, WEIGHTED_KAPPA_LABEL, aggregate_agreement, agreement_by_code,
    compute_comparison_table, count_clearing_observed, gwet_ac1, precision_recall,
)


def windows(code_lists, start=0):
    return [{"window_index": start + i, "codes": c} for i, c in enumerate(code_lists)]


# --- The prevalence paradox that motivated AC1: Lec on 23/24 vs 24/24 ---
human = windows([["Lec"]] * 23 + [["Adm"]])
ai = windows([["Lec"]] * 24)
lec = {r["code"]: r for r in agreement_by_code(human, ai)}["Lec"]
assert (lec["tp"], lec["fp"], lec["fn"], lec["tn"]) == (23, 1, 0, 0), lec
assert lec["pct_agreement"] == 95.8, lec
assert lec["kappa"] == 0.0, lec
# pa = 23/24; pi = 47/48; pe = 2*(47/48)*(1/48); AC1 = (pa - pe) / (1 - pe)
pa, pi = 23 / 24, 47 / 48
pe = 2 * pi * (1 - pi)
assert lec["ac1"] == round((pa - pe) / (1 - pe), 3) == 0.957, lec
print(f"ok  Lec 23/24 vs 24/24: kappa={lec['kappa']}, AC1={lec['ac1']}, "
      f"agreement={lec['pct_agreement']}%")

# --- All 12 codes always present, unobserved ones N/A ---
rows = agreement_by_code(human, ai)
assert [r["code"] for r in rows] == INSTRUCTOR_CODES
cq = rows[INSTRUCTOR_CODES.index("CQ")]
assert cq["kappa"] == "N/A" and cq["ac1"] == "N/A", cq
assert (cq["tp"], cq["fp"], cq["fn"], cq["tn"]) == (0, 0, 0, 24), cq
assert cq["pct_agreement"] == 100.0
print("ok  all 12 codes emitted; never-observed CQ -> kappa N/A, AC1 N/A, tn=24")

# --- One-sided code: AI over-fires, human never marks ---
h = windows([[], [], [], []])
a = windows([["PQ"], [], [], []])
pq = {r["code"]: r for r in agreement_by_code(h, a)}["PQ"]
assert pq["kappa"] == 0.0 and (pq["fp"], pq["tn"]) == (1, 3), pq
assert pq["ac1"] == gwet_ac1(0, 1, 0, 3)
print(f"ok  one-sided PQ over-fire: kappa=0.0, AC1={pq['ac1']} (defined, not N/A)")

# --- Both raters mark every window: kappa undefined, AC1 = 1.0 ---
assert gwet_ac1(10, 0, 0, 0) == 1.0
print("ok  both raters mark all windows -> AC1 1.0")

# --- Cross-check AC1 against irrCAC (Gwet's reference package), if available ---
try:
    from irrCAC.raw import CAC
    import random
    random.seed(7)
    for _ in range(200):
        n = random.randint(5, 40)
        hl = [random.random() < random.random() for _ in range(n)]
        al = [x if random.random() < 0.8 else not x for x in hl]
        if not any(hl + al):
            continue
        tp = sum(1 for x, y in zip(hl, al) if x and y)
        fp = sum(1 for x, y in zip(hl, al) if y and not x)
        fn = sum(1 for x, y in zip(hl, al) if x and not y)
        tn = n - tp - fp - fn
        ref = CAC(pd.DataFrame({"h": [int(x) for x in hl], "a": [int(y) for y in al]}),
                  categories=[0, 1]).gwet()["est"]["coefficient_value"]
        assert abs(gwet_ac1(tp, fp, fn, tn) - ref) < 0.0015, (tp, fp, fn, tn, ref)
    print("ok  AC1 matches irrCAC.gwet() on 200 random 2x2 tables")
except ImportError:
    print("--  irrCAC not importable; skipped reference cross-check")

# --- End-to-end table: columns, summary rows, paired windows, errors file ---
tmp = tempfile.mkdtemp()


def write(name, code_lists, indices, arm=None):
    df = pd.DataFrame({
        "lecture_id": "L1", "professor_id": "professor_9",
        "window_index": indices, "window_start": 0, "window_end": 120,
        "copus_codes": ["|".join(c) for c in code_lists],
    })
    if arm:
        df["arm"] = arm
    path = os.path.join(tmp, name)
    df.to_csv(path, index=False)
    return path


idx = list(range(6))
human_csv = write("human_coding_L1.csv",
                  [["Lec"], ["Lec", "PQ"], ["Lec"], ["Lec", "AnQ"], ["Lec"], ["Adm"]], idx)
arm_csvs = {
    "multimodal": write("mm.csv", [["Lec"], ["Lec", "PQ"], ["Lec"], ["Lec", "AnQ"], ["Lec"], ["Adm"]], idx),
    "vision_only": write("v.csv", [["Lec"]] * 6, idx),
    "audio_only": write("a.csv", [["Lec"], ["Lec", "PQ"], ["Lec"], ["Lec"], ["Lec"], ["Adm"]], idx),
    # transcript arm failed window 0 -> window 0 must drop from EVERY arm
    "transcript_only": write("t.csv", [["Lec", "PQ"], ["Lec"], ["Lec", "AnQ"], ["Lec"], ["Adm"]], idx[1:]),
}
out = os.path.join(tmp, "comparison_table.csv")
df = compute_comparison_table(arm_csvs, human_csv, out, professor_ids="professor_9")

assert list(df["code"]) == (INSTRUCTOR_CODES + [KAPPA_CLEAR_LABEL, AC1_CLEAR_LABEL]
                            + list(AGGREGATE_LABELS))
assert "MEAN" not in set(df["code"])
for col in ["professor_id", "n_total_windows", "n_human_marked",
            "n_codes_human_observed",
            "n_ai_marked_multimodal", "n_ai_marked_audio", "n_ai_marked_vision",
            "n_ai_marked_transcript", "multimodal_kappa", "vision_only_kappa",
            "audio_only_kappa", "transcript_only_kappa", "multimodal_ac1",
            "vision_only_ac1", "audio_only_ac1", "transcript_only_ac1",
            "pct_agreement_multimodal", "pct_agreement_audio",
            "pct_agreement_vision", "pct_agreement_transcript",
            "multimodal_precision", "vision_only_precision", "audio_only_precision",
            "transcript_only_precision", "multimodal_recall", "vision_only_recall",
            "audio_only_recall", "transcript_only_recall"]:
    assert col in df.columns, col
# Precision / recall come after every pre-existing column.
assert list(df.columns)[-8:] == (
    [f"{a}_precision" for a in ("multimodal", "vision_only", "audio_only", "transcript_only")]
    + [f"{a}_recall" for a in ("multimodal", "vision_only", "audio_only", "transcript_only")])
print("ok  comparison_table has all 12 codes, 2 threshold + 3 aggregate rows, no MEAN, "
      "all new columns")

codes = df.set_index("code")
assert codes.loc["Lec", "n_total_windows"] == 5, "window 0 should be dropped for every arm"
assert codes.loc["Lec", "n_human_marked"] == 4
print("ok  failed transcript window dropped from every arm (n_total_windows=5)")

# multimodal is perfect on Lec/PQ/AnQ/Adm (4 observed codes)
assert codes.loc[KAPPA_CLEAR_LABEL, "multimodal_kappa"] == "4 of 4"
assert codes.loc[AC1_CLEAR_LABEL, "multimodal_ac1"] == "4 of 4"
# Denominator is the codes the HUMAN marked (4 here), not 12.
assert int(codes.loc[KAPPA_CLEAR_LABEL, "n_codes_human_observed"]) == 4
assert int(codes.loc[AC1_CLEAR_LABEL, "n_codes_human_observed"]) == 4
print(f"ok  summary rows: multimodal clears kappa on "
      f"{codes.loc[KAPPA_CLEAR_LABEL, 'multimodal_kappa']} human-observed codes")

# --- A code the human never marked must NOT count toward the AC1 clears ---
# 20 windows: the human never marks W; the AI fires it once. kappa 0.0, AC1 high.
h_w = windows([["Lec"]] * 20)
a_w = windows([["Lec", "W"]] + [["Lec"]] * 19)
stats = agreement_by_code(h_w, a_w)
w_row = {r["code"]: r for r in stats}["W"]
assert w_row["n_human_marked"] == 0 and w_row["tp"] == 0
assert w_row["kappa"] == 0.0 and w_row["ac1"] >= 0.9, w_row
cleared, n_obs = count_clearing_observed(stats, "ac1")
assert n_obs == 1 and cleared == 1, (cleared, n_obs)  # only Lec is human-observed
print(f"ok  AC1 {w_row['ac1']} on a code with tp=0 is excluded: counts are "
      f"{cleared}/{n_obs} human-observed codes, not .../12")

errs = pd.read_csv(os.path.join(tmp, "per_code_errors.csv"))
assert len(errs) == 48, len(errs)
assert list(errs.columns) == ["professor_id", "code", "arm", "tp", "fp", "fn", "tn"]
assert ((errs.tp + errs.fp + errs.fn + errs.tn) == 5).all()
v_pq = errs[(errs.arm == "vision_only") & (errs.code == "PQ")].iloc[0]
assert (v_pq.tp, v_pq.fp, v_pq.fn, v_pq.tn) == (0, 0, 1, 4), v_pq
assert (errs[errs.code == "MG"][["tp", "fp", "fn"]] == 0).all().all()
print("ok  per_code_errors.csv: 48 rows, zeros for unobserved codes, counts sum to n")



# --- 2026-10-08 additions ---------------------------------------------------- #

def hand_rolled_ac1(tp, fp, fn, tn):
    """The Gwet (2008) closed form gwet_ac1 used before irrCAC (pre-3f976ea)."""
    n = tp + fp + fn + tn
    if n == 0 or tp + fp + fn == 0:
        return "N/A"
    pa = (tp + tn) / n
    pi = (2 * tp + fn + fp) / (2 * n)
    pe = 2 * pi * (1 - pi)
    return round((pa - pe) / (1 - pe), 3)


def test_irrcac_matches_hand_rolled_on_lec_case():
    # Lec: human 23/24, AI 24/24 -> tp=23, fp=1, fn=0, tn=0.
    table = (23, 1, 0, 0)
    new, old = gwet_ac1(*table), hand_rolled_ac1(*table)
    assert old == 0.957, old
    assert abs(new - old) <= 0.001, f"irrCAC {new} vs hand-rolled {old} on {table}"
    # Edge cases are unchanged: no windows, and a code nobody used, are N/A.
    assert gwet_ac1(0, 0, 0, 0) == hand_rolled_ac1(0, 0, 0, 0) == "N/A"
    assert gwet_ac1(0, 0, 0, 24) == hand_rolled_ac1(0, 0, 0, 24) == "N/A"
    print(f"ok  irrCAC AC1 {new} == hand-rolled {old} on Lec (tp, fp, fn, tn) = {table}")


def test_precision_recall_computation():
    assert precision_recall(3, 1, 2) == (0.75, 0.6)
    assert precision_recall(2, 1, 0) == (0.667, 1.0)          # 3 decimals
    assert precision_recall(0, 0, 4) == ("N/A", 0.0)          # AI never marked it
    assert precision_recall(0, 2, 0) == (0.0, "N/A")          # human never marked it
    assert precision_recall(0, 0, 0) == ("N/A", "N/A")
    # Same values come through agreement_by_code. Lec 23/24 vs 24/24:
    lec_row = {r["code"]: r for r in agreement_by_code(human, ai)}["Lec"]
    assert (lec_row["precision"], lec_row["recall"]) == (0.958, 1.0), lec_row
    # ...and into comparison_table.csv (vision_only never marks PQ: tp=0, fn=1).
    assert codes.loc["PQ", "vision_only_precision"] == "N/A"
    assert float(codes.loc["PQ", "vision_only_recall"]) == 0.0
    assert float(codes.loc["Lec", "multimodal_precision"]) == 1.0
    # Precision / recall are per-code only: blank on every summary row.
    for label in [KAPPA_CLEAR_LABEL, AC1_CLEAR_LABEL, *AGGREGATE_LABELS]:
        assert pd.isna(codes.loc[label, "multimodal_precision"]), label
        assert pd.isna(codes.loc[label, "multimodal_recall"]), label
    print("ok  precision = tp/(tp+fp), recall = tp/(tp+fn), N/A on a zero denominator")


# Worked example for the aggregate rows. 10 windows:
#   Lec: human 9, AI 10 -> tp 9, fp 1, fn 0, tn 0   kappa 0.0   (pe = 0.9)
#   PQ:  human 3, AI 2  -> tp 2, fp 0, fn 1, tn 7   kappa 0.737 (0.28 / 0.38)
#   W:   human 0, AI 1  -> not human-observed, excluded from every aggregate
AGG_HUMAN = windows([["Lec", "PQ"]] * 3 + [["Lec"]] * 6 + [[]])
AGG_AI = windows([["Lec", "PQ"]] * 2 + [["Lec"]] * 3 + [["Lec", "W"]] + [["Lec"]] * 4)
AGG_ROWS = agreement_by_code(AGG_HUMAN, AGG_AI)


def test_overall_raw_agreement_pct():
    by_code = {r["code"]: r for r in AGG_ROWS}
    assert [by_code[c][k] for c in ("Lec", "PQ") for k in ("tp", "fp", "fn", "tn")] == \
        [9, 1, 0, 0, 2, 0, 1, 7]
    # (9 + 0 + 2 + 7) matches / (10 + 10) decisions = 90.0%. W's 9 true
    # negatives would push this to 93.3% if it were counted.
    assert aggregate_agreement(AGG_ROWS)[RAW_AGREEMENT_LABEL] == 90.0
    assert aggregate_agreement([])[RAW_AGREEMENT_LABEL] == "N/A"
    print("ok  overall_raw_agreement_pct = 18 / 20 decisions = 90.0%")


def test_pooled_kappa():
    # Concatenated vectors: 20 decisions, human 12 ones, AI 12 ones, 18 agree.
    # po = 0.9, pe = 0.6^2 + 0.4^2 = 0.52, kappa = 0.38 / 0.48 = 0.792.
    assert aggregate_agreement(AGG_ROWS)[POOLED_KAPPA_LABEL] == 0.792
    assert aggregate_agreement([])[POOLED_KAPPA_LABEL] == "N/A"
    print("ok  pooled_kappa = (0.9 - 0.52) / (1 - 0.52) = 0.792")


def test_prevalence_weighted_mean_kappa():
    by_code = {r["code"]: r for r in AGG_ROWS}
    assert (by_code["Lec"]["kappa"], by_code["PQ"]["kappa"]) == (0.0, 0.737)
    # (9 * 0.0 + 3 * 0.737) / (9 + 3) = 0.184
    assert aggregate_agreement(AGG_ROWS)[WEIGHTED_KAPPA_LABEL] == 0.184
    # A human-observed code with N/A kappa is skipped entirely (weight too):
    # both raters mark Lec on every window -> no variation -> N/A.
    h = windows([["Lec", "PQ"]] * 3 + [["Lec"]] * 7)
    a = windows([["Lec", "PQ"]] * 2 + [["Lec"]] * 8)
    rows = agreement_by_code(h, a)
    assert {r["code"]: r for r in rows}["Lec"]["kappa"] == "N/A"
    assert aggregate_agreement(rows)[WEIGHTED_KAPPA_LABEL] == 0.737
    # No code with a numeric kappa -> N/A.
    only_na = agreement_by_code(windows([["Lec"]] * 4), windows([["Lec"]] * 4))
    assert aggregate_agreement(only_na)[WEIGHTED_KAPPA_LABEL] == "N/A"
    assert aggregate_agreement([])[WEIGHTED_KAPPA_LABEL] == "N/A"
    print("ok  prevalence_weighted_mean_kappa = (9*0.0 + 3*0.737) / 12 = 0.184; "
          "N/A codes skipped")


def test_summary_rows_formatted_as_string():
    raw = pd.read_csv(out, dtype=str, keep_default_na=False).set_index("code")
    assert raw.loc[KAPPA_CLEAR_LABEL, "multimodal_kappa"] == "4 of 4"
    assert raw.loc[AC1_CLEAR_LABEL, "multimodal_ac1"] == "4 of 4"
    # Every arm shares the denominator (4 human-observed codes), whatever it clears.
    for arm in ("vision_only", "audio_only", "transcript_only"):
        for label, metric in ((KAPPA_CLEAR_LABEL, "kappa"), (AC1_CLEAR_LABEL, "ac1")):
            cell = raw.loc[label, f"{arm}_{metric}"]
            count, of, total = cell.split(" ")
            assert count.isdigit() and of == "of" and total == "4", cell
    # Every other value cell on the threshold rows is blank.
    blank = [c for c in raw.columns if c not in ("professor_id", "n_codes_human_observed")
             and not c.endswith(("_kappa", "_ac1"))]
    assert (raw.loc[[KAPPA_CLEAR_LABEL, AC1_CLEAR_LABEL], blank] == "").all().all()
    # Aggregate rows sit at the very bottom, in their own columns.
    assert list(raw.index[-3:]) == list(AGGREGATE_LABELS)
    assert raw.loc[RAW_AGREEMENT_LABEL, "pct_agreement_multimodal"] == "100.0"
    assert raw.loc[POOLED_KAPPA_LABEL, "multimodal_kappa"] == "1.0"
    assert raw.loc[WEIGHTED_KAPPA_LABEL, "multimodal_kappa"] == "1.0"
    print(f"ok  summary rows are paste-ready strings: "
          f"{raw.loc[KAPPA_CLEAR_LABEL, 'multimodal_kappa']!r}")


test_irrcac_matches_hand_rolled_on_lec_case()
test_precision_recall_computation()
test_overall_raw_agreement_pct()
test_pooled_kappa()
test_prevalence_weighted_mean_kappa()
test_summary_rows_formatted_as_string()

print("\nAll agreement tests passed.")
