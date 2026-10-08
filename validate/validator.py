import math
import pandas as pd
import os
from irrCAC.raw import CAC
from sklearn.metrics import cohen_kappa_score

from validate.convert_copus_sheet import INSTRUCTOR_CODES

# Label for the summary row appended to every kappa table.
MEAN_ROW_LABEL = "MEAN"

# COPUS "acceptable agreement" threshold (Smith et al., 2013).
AGREEMENT_THRESHOLD = 0.7
KAPPA_CLEAR_LABEL = "codes_clearing_kappa_0.7"
AC1_CLEAR_LABEL = "codes_clearing_ac1_0.7"
THRESHOLD_LABELS = (KAPPA_CLEAR_LABEL, AC1_CLEAR_LABEL)

# Aggregate (all-codes) agreement rows, appended below the threshold rows.
# Supplementary only: per-code reliability is the primary view, because pooling
# lets true negatives from low-prevalence codes inflate the aggregate.
RAW_AGREEMENT_LABEL = "overall_raw_agreement_pct"
POOLED_KAPPA_LABEL = "pooled_kappa"
WEIGHTED_KAPPA_LABEL = "prevalence_weighted_mean_kappa"
AGGREGATE_LABELS = (RAW_AGREEMENT_LABEL, POOLED_KAPPA_LABEL, WEIGHTED_KAPPA_LABEL)

# Every non-code row a kappa table can end with.
SUMMARY_LABELS = THRESHOLD_LABELS + AGGREGATE_LABELS

# Short arm names used in the prevalence / % agreement column names.
ARM_SHORT = {
    "multimodal": "multimodal",
    "vision_only": "vision",
    "audio_only": "audio",
    "transcript_only": "transcript",
}


def load_human_codes(csv_path):
    """
    Loads long-format human coding CSV.
    Expected columns: lecture_id, window_index, window_start, window_end, copus_codes
    """
    df = pd.read_csv(csv_path)
    rows = []
    for _, row in df.iterrows():
        codes = row["copus_codes"].split("|") if pd.notna(row["copus_codes"]) and row["copus_codes"] != "" else []
        rows.append({
            "window_index": int(row["window_index"]),
            "codes": codes
        })
    return rows


def load_ai_codes(csv_path):
    """
    Loads results.csv from the pipeline.
    Expected columns: lecture_id, window_index, window_start, window_end, arm,
    model, copus_codes
    """
    df = pd.read_csv(csv_path)
    rows = []
    for _, row in df.iterrows():
        codes = row["copus_codes"].split("|") if pd.notna(row["copus_codes"]) and row["copus_codes"] != "" else []
        rows.append({
            "window_index": int(row["window_index"]),
            "codes": codes
        })
    return rows


def mean_kappa(values):
    """
    Mean of the numeric kappa values, skipping "N/A" and "ERR:" entries.

    Codes that neither rater used contribute no information and must not drag
    the average toward zero -- that is the whole point of reporting them as N/A
    rather than 0.00.
    """
    nums = []
    for v in values:
        f = pd.to_numeric(v, errors="coerce")
        if pd.notna(f):
            nums.append(float(f))
    if not nums:
        return "N/A"
    return round(sum(nums) / len(nums), 3)


def binary_kappa(code, human_labels, ai_labels):
    """
    Cohen's kappa for one code's present/absent labels, or "N/A" when undefined.
    """
    n = len(human_labels)
    human_pos = sum(human_labels)
    ai_pos = sum(ai_labels)
    if n == 0:
        return "N/A"
    if human_pos == 0 and ai_pos == 0:
        # NEITHER rater used this code -- there are no observations at all,
        # so kappa is genuinely undefined. When only ONE rater is at zero the
        # code IS defined: sklearn returns 0.0, and that 0.0 is a real
        # finding (one rater over- or under-fired the code on every window),
        # not an artifact. Suppressing it hid an 11-window AnQ over-fire.
        return "N/A"
    if len(set(human_labels + ai_labels)) < 2:
        # No variation in either rater.
        return "N/A"
    if human_pos == n or ai_pos == n:
        # Mirror image of the case above: a rater who marked EVERY
        # window is also constant, so kappa is depressed by prevalence
        # rather than by disagreement. Reported, but flagged.
        print(f"[warn] {code}: a rater marked all {n} windows "
              f"(human={human_pos}, ai={ai_pos}); kappa is prevalence-"
              f"depressed, not a disagreement measure.")
    try:
        kappa = cohen_kappa_score(human_labels, ai_labels)
        # sklearn returns nan when the (1 - pe) denominator is 0.
        return "N/A" if math.isnan(kappa) else round(kappa, 3)
    except Exception as e:
        return f"ERR: {e}"


def gwet_ac1(tp, fp, fn, tn):
    """
    Gwet's AC1 for two raters and two categories (code present / absent), via
    irrCAC (Gwet's reference implementation).

    Gwet, K. L. (2008). Computing inter-rater reliability and its variance in
    the presence of high agreement. BJMSP, 61(1), 29-48.

    For two categories this is the closed form
        pa  = (tp + tn) / n
        pi  = ((tp + fn) + (tp + fp)) / (2n)    # mean share of "present"
        pe  = 2 * pi * (1 - pi)
        AC1 = (pa - pe) / (1 - pe)
    which irrCAC reproduces exactly (checked on every 2x2 table up to n = 20).

    Unlike kappa's chance term, pe shrinks toward 0 as a code approaches 0% or
    100% prevalence, so near-perfect agreement on Lec (marked in 23 of 24 windows
    by both raters) is no longer scored as ~0.

    Returns "N/A" when there are no windows or NEITHER rater used the code. The
    formula gives 1.0 there, but that is agreement on an absence nobody
    observed; counting it would let CQ/MG/1o1 "clear" the 0.7 threshold without
    ever appearing. That guard is applied here, not by irrCAC (which returns 1.0).
    Also "N/A" for a single window: irrCAC's variance step divides by n - 1.
    """
    n = tp + fp + fn + tn
    if n == 0 or tp + fp + fn == 0:
        return "N/A"
    ratings = pd.DataFrame({
        "human": [1] * tp + [0] * fp + [1] * fn + [0] * tn,
        "ai":    [1] * tp + [1] * fp + [0] * fn + [0] * tn,
    })
    try:
        ac1 = CAC(ratings, categories=[0, 1]).gwet()["est"]["coefficient_value"]
    except ZeroDivisionError:
        return "N/A"
    return "N/A" if math.isnan(ac1) else round(ac1, 3)


def precision_recall(tp, fp, fn):
    """
    (precision, recall) of the AI against the human for one code, each rounded
    to 3 decimals, or "N/A" when its denominator is zero:
        precision = tp / (tp + fp)    # of the windows the AI marked
        recall    = tp / (tp + fn)    # of the windows the human marked
    """
    precision = round(tp / (tp + fp), 3) if tp + fp else "N/A"
    recall = round(tp / (tp + fn), 3) if tp + fn else "N/A"
    return precision, recall


def aggregate_agreement(rows):
    """
    Three all-codes agreement measures over the codes the human marked at least
    once (see observed_rows), from agreement_by_code rows:

      overall_raw_agreement_pct       sum(tp + tn) / sum(tp + tn + fp + fn) * 100,
                                      1 decimal
      pooled_kappa                    Cohen's kappa on the per-code human / AI
                                      binary vectors concatenated across codes,
                                      3 decimals
      prevalence_weighted_mean_kappa  per-code kappa weighted by n_human_marked,
                                      N/A codes skipped, 3 decimals

    Each is "N/A" when undefined (no observed codes / no numeric kappa). These
    are supplementary: true negatives from low-prevalence codes inflate them.
    """
    obs = observed_rows(rows)

    matches = sum(r["tp"] + r["tn"] for r in obs)
    decisions = sum(r["tp"] + r["tn"] + r["fp"] + r["fn"] for r in obs)
    raw = round(100.0 * matches / decisions, 1) if decisions else "N/A"

    # The concatenated vectors are fully determined by each code's 2x2 counts,
    # and kappa does not depend on window order.
    human, ai = [], []
    for r in obs:
        for h, a, k in ((1, 1, r["tp"]), (0, 1, r["fp"]), (1, 0, r["fn"]), (0, 0, r["tn"])):
            human += [h] * k
            ai += [a] * k
    pooled = "N/A"
    if human:
        try:
            k = cohen_kappa_score(human, ai)
            pooled = "N/A" if math.isnan(k) else round(k, 3)
        except Exception as e:
            pooled = f"ERR: {e}"

    weighted = [(r["n_human_marked"], float(k)) for r in obs
                for k in [pd.to_numeric(r["kappa"], errors="coerce")] if pd.notna(k)]
    total_w = sum(w for w, _ in weighted)
    weighted_kappa = (round(sum(w * k for w, k in weighted) / total_w, 3)
                      if total_w else "N/A")

    return {
        RAW_AGREEMENT_LABEL: raw,
        POOLED_KAPPA_LABEL: pooled,
        WEIGHTED_KAPPA_LABEL: weighted_kappa,
    }


def clearing_cell(cleared, n_observed):
    """Threshold-count cell, pasted verbatim into thesis Table 6.1: "3 of 5"."""
    return f"{cleared} of {n_observed}"


def tag_lecture(windows, lecture_key):
    """
    Key each window by (lecture_key, window_index) before pooling lectures.

    Pooling on bare window_index pairs the k-th human row at an index with the
    k-th AI row at that index -- so if lecture A's AI failed window 5 and lecture
    B's did not, A's human coding was scored against B's AI output.
    """
    return [{**w, "window_index": (lecture_key, w["window_index"])} for w in windows]


def _pair_windows(human_windows, ai_windows):
    """
    Aligned (human, ai) window pairs over shared window_index values, preserving
    human order. Pooled lists may repeat an index across lectures; the k-th human
    row with an index pairs with the k-th AI row with that index.
    """
    ai_by_index = {}
    for w in ai_windows:
        ai_by_index.setdefault(w["window_index"], []).append(w)
    pairs, ai_cursor = [], {}
    for w in human_windows:
        idx = w["window_index"]
        bucket = ai_by_index.get(idx)
        pos = ai_cursor.get(idx, 0)
        if not bucket or pos >= len(bucket):
            continue
        pairs.append((w, bucket[pos]))
        ai_cursor[idx] = pos + 1
    return pairs


def agreement_by_code(human_windows, ai_windows, codes=INSTRUCTOR_CODES):
    """
    Per-code agreement for multi-label output, treating each code as its own
    present/absent binary rating. One row per code in `codes` -- ALL of them,
    including codes neither rater used -- with:

      n_total_windows, n_human_marked, n_ai_marked,
      tp (both marked), fp (AI only), fn (human only), tn (neither),
      pct_agreement = (tp + tn) / n_total_windows, as a percentage,
      kappa (Cohen), ac1 (Gwet),
      precision = tp / (tp + fp), recall = tp / (tp + fn) (AI against human)

    Only windows present in BOTH lists are scored. Confusion matrices across
    codes are not defined for multi-label output; these per-code 2x2 tables are.
    """
    pairs = _pair_windows(human_windows, ai_windows)
    n = len(pairs)

    known = set(codes)
    stray = sorted({c for h, a in pairs for c in h["codes"] + a["codes"]
                    if c and c not in known})
    if stray:
        print(f"[warn] codes outside the COPUS instructor set were ignored: {stray}")

    rows = []
    for code in codes:
        human_labels = [1 if code in h["codes"] else 0 for h, _ in pairs]
        ai_labels = [1 if code in a["codes"] else 0 for _, a in pairs]
        tp = sum(1 for h, a in zip(human_labels, ai_labels) if h and a)
        fp = sum(1 for h, a in zip(human_labels, ai_labels) if a and not h)
        fn = sum(1 for h, a in zip(human_labels, ai_labels) if h and not a)
        tn = n - tp - fp - fn
        precision, recall = precision_recall(tp, fp, fn)
        rows.append({
            "code": code,
            "n_total_windows": n,
            "n_human_marked": tp + fn,
            "n_ai_marked": tp + fp,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "pct_agreement": round(100.0 * (tp + tn) / n, 1) if n else "N/A",
            "kappa": binary_kappa(code, human_labels, ai_labels),
            "ac1": gwet_ac1(tp, fp, fn, tn),
            "precision": precision,
            "recall": recall,
        })
    return rows


def count_clearing(values, threshold=AGREEMENT_THRESHOLD):
    """How many values are numeric and >= threshold (N/A never clears)."""
    nums = pd.to_numeric(pd.Series(list(values), dtype=object), errors="coerce")
    return int((nums >= threshold).sum())


def observed_rows(rows):
    """
    The agreement rows for codes the HUMAN marked at least once.

    The threshold counts are scored over these only, so the denominator is
    "codes observed by the human coder", not a flat 12. Two reasons:
      - a code neither rater used is N/A and could never clear anyway;
      - a code the human never marked but the AI fired once gets kappa 0.0 and
        AC1 ~0.95 -- high agreement on a code with no true positives. Counting
        those made audio_only look like it cleared 7-8 of 12 when 3 of the
        clears (FUp, W, O) had tp = 0.
    """
    return [r for r in rows if r["n_human_marked"] > 0]


def count_clearing_observed(rows, metric, threshold=AGREEMENT_THRESHOLD):
    """(codes clearing threshold, codes the human marked) for one arm."""
    obs = observed_rows(rows)
    return count_clearing((r[metric] for r in obs), threshold), len(obs)


def kappa_by_code(human_windows, ai_windows):
    """
    Computes Cohen's Kappa per COPUS code given already-loaded window lists
    (each a list of {"window_index", "codes"} dicts). Only compares windows
    present in BOTH sets. Returns a list of dicts:
      {"code", "kappa", "human_positive_windows", "ai_positive_windows"}
    sorted by code.

    Kappa is "N/A" only when it is mathematically undefined rather than low:
      - NEITHER rater ever marked the code (no observations at all), or
      - neither rater varies on the code (e.g. both marked every window).

    A code used by exactly one rater is NOT N/A. cohen_kappa_score returns
    exactly 0.0 there, and that zero is the correct reading: the two raters
    agreed on nothing for that code. Reporting it as N/A hides one-sided
    over-fires (Sofia n=0 vs AI n=11 on AnQ) behind a blank cell.

    Window lists may pool multiple lectures; window_index collisions across
    lectures are handled by pairing on position within the shared-index order
    per (human, ai) pair --- callers that pool should pass matching-length,
    aligned lists (see compute_comparison_table).
    """
    all_codes = set()
    for w in human_windows:
        all_codes.update(w["codes"])
    for w in ai_windows:
        all_codes.update(w["codes"])
    all_codes.discard("")

    pairs = _pair_windows(human_windows, ai_windows)

    n = len(pairs)
    results = []
    for code in sorted(all_codes):
        human_labels = [1 if code in h["codes"] else 0 for h, _ in pairs]
        ai_labels = [1 if code in a["codes"] else 0 for _, a in pairs]
        human_pos = sum(human_labels)
        ai_pos = sum(ai_labels)

        kappa = binary_kappa(code, human_labels, ai_labels)

        results.append({
            "code": code,
            "kappa": kappa,
            "human_positive_windows": human_pos,
            "ai_positive_windows": ai_pos,
        })
    return results


def compute_kappa(csv_a, csv_b, output_dir, output_filename="kappa_results.csv",
                  label_a="A", label_b="B", professor_id=""):
    """
    Computes Cohen's Kappa for each COPUS code comparing two long-format code
    sets. Works for AI-vs-human OR human-vs-human (a second coder, for
    inter-rater reliability): both inputs are read the same way, so any two
    coding CSVs can be compared.
    Only compares windows present in BOTH csv_a and csv_b.

    output_filename lets callers name each comparison distinctly
    (e.g. kappa_ai_vs_sofia.csv). label_a/label_b customize the printed header.
    Returns the per-code kappa DataFrame, with a trailing MEAN row averaging the
    numeric kappas only. The 'human_positive_windows' / 'ai_positive_windows'
    columns count positives for csv_a / csv_b respectively. Every row, MEAN
    included, is stamped with professor_id.
    """
    os.makedirs(output_dir, exist_ok=True)

    a_windows = load_human_codes(csv_a)
    b_windows = load_human_codes(csv_b)

    a_indices = set(w["window_index"] for w in a_windows)
    b_indices = set(w["window_index"] for w in b_windows)
    print(f"{label_a} windows: {len(a_indices)}")
    print(f"{label_b} windows: {len(b_indices)}")
    print(f"Shared windows for comparison: {sorted(a_indices & b_indices)}")

    kappa_results = kappa_by_code(a_windows, b_windows)
    kappa_results.append({
        "code": MEAN_ROW_LABEL,
        "kappa": mean_kappa([r["kappa"] for r in kappa_results]),
        "human_positive_windows": "",
        "ai_positive_windows": "",
    })

    print(f"\n{'Code':<8} {'Kappa':>8} {label_a + '+':>8} {label_b + '+':>8}")
    print("-" * 38)
    for r in kappa_results:
        if r["code"] == MEAN_ROW_LABEL:
            print("-" * 38)
        print(f"{r['code']:<8} {str(r['kappa']):>8} "
              f"{str(r['human_positive_windows']):>8} "
              f"{str(r['ai_positive_windows']):>8}")
    print("(N/A = kappa undefined: neither rater ever used that code)")

    # Save results
    results_df = pd.DataFrame(kappa_results)
    results_df.insert(0, "professor_id", professor_id)
    out_path = os.path.join(output_dir, output_filename)
    results_df.to_csv(out_path, index=False)
    print(f"\nKappa results saved -> {out_path}")

    return results_df


ARM_ORDER = ["multimodal", "vision_only", "audio_only", "transcript_only"]


def _normalize_arm_csvs(arm_csvs):
    """
    Accepts a single {arm: csv_path} dict (one lecture) or a list of such dicts
    (multiple lectures). Returns a list of per-lecture dicts.
    """
    if isinstance(arm_csvs, dict):
        return [arm_csvs]
    return list(arm_csvs)


def compute_comparison_table(arm_csvs, human_csv, output_csv, per_lecture=False,
                             professor_ids="", errors_csv=None):
    """
    Builds the cross-arm agreement table: one row per COPUS instructor code
    (all 12, in protocol order, observed or not) with, per arm, Cohen's kappa,
    Gwet's AC1, and raw % agreement, plus prevalence counts for context.

    arm_csvs:
      - {arm_name: results_csv_path} for a single lecture, OR
      - [ {arm_name: results_csv_path}, ... ] one dict per lecture (pooled).
    human_csv:
      - path to the single lecture's human CSV, OR
      - list of human CSV paths aligned with the arm_csvs list.
    output_csv: path for the pooled wide table.
    per_lecture: also write a per-lecture breakdown next to output_csv.
    professor_ids: one id for all lectures, or a list aligned with arm_csvs.
      A pooled row covering several professors is stamped with every id,
      pipe-joined (e.g. "professor_1|professor_2"), so it is never blank.
    errors_csv: long-format per-(code, arm) tp/fp/fn/tn table for Figure 6.1.
      Defaults to per_code_errors.csv next to output_csv.

    Every arm is scored on the SAME windows: within each lecture, a window that
    any arm lacks a result for (a failed chunk) is dropped from all arms, with a
    warning. Otherwise one arm's failure changes its denominator and the arms are
    no longer a paired comparison.

    The table ends with two summary rows instead of a mean: how many of the
    human-observed codes clear 0.7 on kappa, and on AC1, per arm, written as
    "3 of 5". An unweighted mean of kappa is not reported -- it mixes N/A codes
    and wildly different prevalences. Below those come the three supplementary
    aggregate rows from aggregate_agreement (overall raw % agreement in the
    pct_agreement columns; pooled and prevalence-weighted kappa in the kappa
    columns).
    """
    lecture_arm_csvs = _normalize_arm_csvs(arm_csvs)
    human_csvs = human_csv if isinstance(human_csv, (list, tuple)) else [human_csv]
    if len(human_csvs) != len(lecture_arm_csvs):
        raise ValueError(
            f"human_csv count ({len(human_csvs)}) must match lecture count "
            f"({len(lecture_arm_csvs)})"
        )

    arms = [a for a in ARM_ORDER if any(a in d for d in lecture_arm_csvs)]
    if isinstance(professor_ids, (list, tuple)):
        lecture_prof_ids = list(professor_ids)
        if len(lecture_prof_ids) != len(lecture_arm_csvs):
            raise ValueError(
                f"professor_ids count ({len(lecture_prof_ids)}) must match lecture "
                f"count ({len(lecture_arm_csvs)})")
    else:
        lecture_prof_ids = [professor_ids] * len(lecture_arm_csvs)

    # Load each lecture once, restricted to the windows every arm (and the human)
    # has.
    lectures = []
    for d, hp, pid in zip(lecture_arm_csvs, human_csvs, lecture_prof_ids):
        lid = _lecture_id_from_human_csv(hp)
        human = load_human_codes(hp)
        ai = {arm: load_ai_codes(d[arm]) for arm in arms if arm in d}
        human_idx = {w["window_index"] for w in human}
        common = set(human_idx)
        for arm, rows in ai.items():
            arm_idx = {w["window_index"] for w in rows}
            missing = sorted(human_idx - arm_idx)
            if missing:
                print(f"[warn] {lid}: arm {arm} has no result for human-coded "
                      f"window(s) {missing}; dropped from EVERY arm so the arms stay "
                      f"paired. Re-run the arm to fill them.")
            common &= arm_idx
        key = len(lectures)
        keep = lambda rows: tag_lecture(
            [w for w in rows if w["window_index"] in common], key)
        lectures.append({
            "lecture_id": lid, "professor_id": pid,
            "human": keep(human), "ai": {arm: keep(r) for arm, r in ai.items()},
        })

    def build_table(lecs):
        """{arm: [agreement row per code]} pooled over the given lectures."""
        table = {}
        for arm in arms:
            with_arm = [l for l in lecs if arm in l["ai"]]
            human = [w for l in with_arm for w in l["human"]]
            ai = [w for l in with_arm for w in l["ai"][arm]]
            table[arm] = agreement_by_code(human, ai)
        return table

    def wide_rows(table, prefix):
        """One row per code plus the two threshold-count rows."""
        first = table[arms[0]]
        rows = []
        for i, code in enumerate(INSTRUCTOR_CODES):
            row = dict(prefix)
            row["code"] = code
            row["n_total_windows"] = first[i]["n_total_windows"]
            row["n_human_marked"] = first[i]["n_human_marked"]
            for arm in arms:
                row[f"n_ai_marked_{ARM_SHORT[arm]}"] = table[arm][i]["n_ai_marked"]
            for arm in arms:
                row[f"{arm}_kappa"] = table[arm][i]["kappa"]
            for arm in arms:
                row[f"{arm}_ac1"] = table[arm][i]["ac1"]
            for arm in arms:
                row[f"pct_agreement_{ARM_SHORT[arm]}"] = table[arm][i]["pct_agreement"]
            for arm in arms:
                row[f"{arm}_precision"] = table[arm][i]["precision"]
            for arm in arms:
                row[f"{arm}_recall"] = table[arm][i]["recall"]
            rows.append(row)
        for label, metric in ((KAPPA_CLEAR_LABEL, "kappa"), (AC1_CLEAR_LABEL, "ac1")):
            row = dict(prefix)
            row["code"] = label
            for arm in arms:
                cleared, n_obs = count_clearing_observed(table[arm], metric)
                row[f"{arm}_{metric}"] = clearing_cell(cleared, n_obs)
                # Denominator: codes the human marked at least once. Same for
                # every arm (the human coding does not change between arms).
                row["n_codes_human_observed"] = n_obs
            rows.append(row)
        aggregates = {arm: aggregate_agreement(table[arm]) for arm in arms}
        for label, column in ((RAW_AGREEMENT_LABEL, "pct_agreement_{short}"),
                              (POOLED_KAPPA_LABEL, "{arm}_kappa"),
                              (WEIGHTED_KAPPA_LABEL, "{arm}_kappa")):
            row = dict(prefix)
            row["code"] = label
            for arm in arms:
                row[column.format(arm=arm, short=ARM_SHORT[arm])] = aggregates[arm][label]
            rows.append(row)
        return rows

    def columns(prefix_cols):
        return (prefix_cols + ["code", "n_total_windows", "n_human_marked",
                               "n_codes_human_observed"]
                + [f"n_ai_marked_{ARM_SHORT[a]}" for a in arms]
                + [f"{a}_kappa" for a in arms]
                + [f"{a}_ac1" for a in arms]
                + [f"pct_agreement_{ARM_SHORT[a]}" for a in arms]
                + [f"{a}_precision" for a in arms]
                + [f"{a}_recall" for a in arms])

    # Pooled table (main output)
    pooled_prof_id = "|".join(sorted({p for p in lecture_prof_ids if p}))
    pooled = build_table(lectures)
    pooled_df = pd.DataFrame(wide_rows(pooled, {"professor_id": pooled_prof_id}),
                             columns=columns(["professor_id"]))
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    pooled_df.to_csv(output_csv, index=False)

    # Per-code error counts (Figure 6.1): 12 codes x arms, zeros included.
    errors_csv = errors_csv or os.path.join(
        os.path.dirname(output_csv) or ".", "per_code_errors.csv")
    err_rows = [
        {"professor_id": pooled_prof_id, "code": r["code"], "arm": arm,
         "tp": r["tp"], "fp": r["fp"], "fn": r["fn"], "tn": r["tn"]}
        for arm in arms for r in pooled[arm]
    ]
    pd.DataFrame(err_rows, columns=["professor_id", "code", "arm", "tp", "fp", "fn", "tn"]
                 ).to_csv(errors_csv, index=False)

    n = pooled[arms[0]][0]["n_total_windows"] if arms else 0
    print(f"\n=== Comparison table (pooled, {n} windows; kappa / AC1 per code x arm) ===")
    header = f"{'Code':<8}{'Human+':>7}" + "".join(f"{a:>22}" for a in arms)
    print(header)
    print("-" * len(header))
    for i, code in enumerate(INSTRUCTOR_CODES):
        line = f"{code:<8}{pooled[arms[0]][i]['n_human_marked']:>7}"
        for arm in arms:
            r = pooled[arm][i]
            cell = f"{r['kappa']} / {r['ac1']}"
            line += f"{cell:>22}"
        print(line)
    print("-" * len(header))
    n_observed = len(observed_rows(pooled[arms[0]])) if arms else 0
    for label, metric in (("k>=0.7", "kappa"), ("AC1>=0.7", "ac1")):
        cells = ""
        for a in arms:
            cleared, n_obs = count_clearing_observed(pooled[a], metric)
            cells += f"{f'{cleared}/{n_obs}':>22}"
        print(f"{label:<15}{cells}")
    print(f"(counts are over the {n_observed} code(s) the human marked at least "
          f"once, not all 12: a code the human never used scores AC1 ~0.95 on "
          f"true negatives alone)")
    print("(N/A = undefined: neither rater used the code, or no variation; never clears 0.7)")
    print(f"\nComparison table saved -> {output_csv}")
    print(f"Per-code TP/FP/FN/TN saved -> {errors_csv}")

    # Per-lecture breakdown
    if per_lecture and len(lectures) > 1:
        by_rows = []
        for lec in lectures:
            by_rows.extend(wide_rows(build_table([lec]), {
                "professor_id": lec["professor_id"], "lecture_id": lec["lecture_id"]}))
        by_df = pd.DataFrame(by_rows, columns=columns(["professor_id", "lecture_id"]))
        by_path = os.path.join(
            os.path.dirname(output_csv) or ".", "comparison_table_by_lecture.csv"
        )
        by_df.to_csv(by_path, index=False)
        print(f"Per-lecture breakdown saved -> {by_path}")

    return pooled_df


def _lecture_id_from_human_csv(human_csv):
    """Best-effort lecture id from a human_coding_{id}.csv path (else stem)."""
    base = os.path.basename(human_csv)
    stem = os.path.splitext(base)[0]
    prefix = "human_coding_"
    return stem[len(prefix):] if stem.startswith(prefix) else stem
