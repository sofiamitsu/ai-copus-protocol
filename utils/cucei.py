"""
CUCEI scores: loading and sanity-checking the study's CUCEI results.

Two sources, either or both:
  data/cucei/*.xlsm   one filled CUCEI_Scoring_Tool per professor (preferred --
                      the workbook IS the working file, no CSV to keep in sync)
  data/cucei_scores.csv
The Streamlit app also accepts one workbook per run as an upload;
stage_data_dir() stages just that professor's scores for the run.

A report only ever shows its own professor's scores, read against the scale
midpoint -- never a cross-professor average.

The College and University Classroom Environment Inventory (CUCEI) scores are
computed OUTSIDE this pipeline and supplied as a finished CSV, one row per
(professor_id, dimension):

    professor_id, dimension, mean, sd, n_respondents

  - dimension is one of the 7 CUCEI subscales in DIMENSIONS
  - mean / sd are on the 1.00-4.00 response scale
  - extra columns are allowed and ignored

The pipeline never scores raw survey answers. (A scorer built from
CUCEI_Scoring_Tool.xlsm was removed on 2026-09-19: that workbook's Mean formula
divides only the last item by 7, its SD formula errors, and its Scoring Key
reverses 19 of 49 items the opposite way to the item wording and to the Qualtrics
survey's own R_/N_ labels. The final scores are produced and checked by hand.)
"""
import glob
import os
import re
import statistics

import pandas as pd

DIMENSIONS = [
    "Personalization", "Involvement", "Student Cohesiveness", "Satisfaction",
    "Task Orientation", "Innovation", "Individualization",
]

# What each subscale measures (paraphrasing Fraser, Treagust & Dennis, 1986),
# used for the one-line interpretations in the Faculty Feedback Report.
DIMENSION_MEANINGS = {
    "Personalization": "opportunities to interact with you, and your concern for "
                       "students' welfare",
    "Involvement": "how actively and attentively students take part in class",
    "Student Cohesiveness": "how well students know, help and support one another",
    "Satisfaction": "how much students enjoy the class",
    "Task Orientation": "how clear and well organized class activities are",
    "Innovation": "how often new or unusual activities and techniques are used",
    "Individualization": "how far students can make decisions and work at their "
                         "own pace or interest",
}

SCALE_MIN, SCALE_MAX, SCALE_MIDPOINT = 1.0, 4.0, 2.5
SCORING_KEY_SHEET = "Scoring Key"
REQUIRED_COLUMNS = ["professor_id", "dimension", "mean", "sd", "n_respondents"]

# Group Summary sheet layout (dimension rows are found by name, not by number).
GROUP_SHEET = "Group Summary"
GROUP_COLS = {"dimension": 1, "n_respondents": 2, "mean": 3, "sd": 4}
# A workbook read and a straight recompute from the raw answers must agree to
# this much; the 2026-09 Mean-formula bug showed up as a difference of ~14.
RECOMPUTE_TOLERANCE = 0.001
POSITIVE_SCORE = {"SA": 4, "A": 3, "D": 2, "SD": 1}
N_ITEMS = 49

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DEFAULT_PATH = os.path.join(_DATA_DIR, "cucei_scores.csv")
DEFAULT_WORKBOOK_DIR = os.path.join(_DATA_DIR, "cucei")


def professor_id_from_filename(path):
    """'CUCEI PROFESSOR 1 (1).xlsm' -> 'professor_1'."""
    m = re.search(r"prof(?:essor)?[\s_-]*(\d+)", os.path.basename(path), re.IGNORECASE)
    return f"professor_{int(m.group(1))}" if m else None


def _recompute_from_answers(wb):
    """
    {dimension: (mean, sd, n_complete)} computed from the Data Entry sheet using
    the workbook's own Scoring Key -- an independent check on its formulas.
    """
    key = {}
    for row in wb[SCORING_KEY_SHEET].iter_rows(min_row=2, values_only=True):
        if row[0]:
            key[str(row[0]).strip()] = (
                [int(x) for x in str(row[1]).split(",")],
                [int(x) for x in str(row[2]).split(",")])
    ws = wb["Data Entry"]
    header = [c.value for c in ws[1]]
    cols = {f"Q{i}": header.index(f"Q{i}") for i in range(1, N_ITEMS + 1)}
    answers = []
    for values in ws.iter_rows(min_row=2, values_only=True):
        row = {q: (str(values[j]).strip().upper() if values[j] else None)
               for q, j in cols.items()}
        if any(row.values()):
            answers.append(row)

    out = {}
    for dim, (pos, neg) in key.items():
        means = []
        for row in answers:
            items = []
            for q in pos + neg:
                a = row[f"Q{q}"]
                if a not in POSITIVE_SCORE:
                    items = None
                    break
                score = POSITIVE_SCORE[a]
                items.append(5 - score if q in neg else score)
            if items:  # a subscale counts only when all 7 items are answered
                means.append(sum(items) / len(items))
        out[dim] = (
            statistics.mean(means) if means else None,
            statistics.stdev(means) if len(means) > 1 else None,
            len(means),
        )
    return out


def load_workbook_scores(path, professor_id=None):
    """
    One professor's scores from a filled CUCEI_Scoring_Tool workbook.

    Reads the Group Summary sheet (the professor-facing numbers) and verifies
    them against a recompute from the raw SA/A/D/SD answers using the workbook's
    own Scoring Key. A disagreement means the sheet's formulas are wrong -- as
    they were before 2026-09-26, when every subscale Mean divided only the last
    item by 7 -- so it raises instead of publishing the number.
    """
    import openpyxl

    professor_id = professor_id or professor_id_from_filename(path)
    if not professor_id:
        raise ValueError(
            f"{os.path.basename(path)}: cannot tell which professor this is. Name "
            f"the file with the professor number, e.g. 'CUCEI PROFESSOR 1.xlsm'.")

    values = openpyxl.load_workbook(path, data_only=True)
    if GROUP_SHEET not in values.sheetnames:
        raise ValueError(f"{os.path.basename(path)}: no '{GROUP_SHEET}' sheet.")
    recomputed = _recompute_from_answers(values)

    ws = values[GROUP_SHEET]
    found = {}
    for row in ws.iter_rows(min_row=1, values_only=True):
        name = str(row[GROUP_COLS["dimension"] - 1] or "").strip()
        if name in DIMENSIONS:
            found[name] = {k: row[c - 1] for k, c in GROUP_COLS.items() if k != "dimension"}

    missing = [d for d in DIMENSIONS if d not in found]
    if missing:
        raise ValueError(f"{os.path.basename(path)}: {GROUP_SHEET} has no row for "
                         f"{missing}.")

    rows, unsaved = [], []
    for dim in DIMENSIONS:
        cell = found[dim]
        mean, sd, n = cell["mean"], cell["sd"], cell["n_respondents"]
        calc_mean, calc_sd, calc_n = recomputed.get(dim, (None, None, 0))
        if mean is None:
            # Saved by something that does not evaluate formulas: fall back to the
            # recompute rather than failing.
            unsaved.append(dim)
            mean, sd, n = calc_mean, calc_sd, calc_n
        elif calc_mean is not None and abs(float(mean) - calc_mean) > RECOMPUTE_TOLERANCE:
            raise ValueError(
                f"{os.path.basename(path)}: {GROUP_SHEET} reports {dim} = "
                f"{float(mean):.4f}, but scoring the raw answers with this "
                f"workbook's own Scoring Key gives {calc_mean:.4f}. The sheet's "
                f"formulas disagree with its data -- fix the workbook before using "
                f"it (the Mean column must be the subscale Sum / 7).")
        rows.append({"professor_id": professor_id, "dimension": dim,
                     "mean": mean, "sd": sd, "n_respondents": n})
    if unsaved:
        print(f"[warn] {os.path.basename(path)}: {GROUP_SHEET} has no saved values "
              f"for {unsaved}; scores were recomputed from the raw answers. Open the "
              f"workbook in Excel and save it to store the computed values.")
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def _workbook_paths(workbook_dir):
    return sorted(p for p in glob.glob(os.path.join(workbook_dir, "*.xls[xm]"))
                  if not os.path.basename(p).startswith("~$"))


def load_workbook_dir(workbook_dir=DEFAULT_WORKBOOK_DIR):
    """Every CUCEI workbook in a folder, concatenated. None if there are none."""
    frames = [load_workbook_scores(p) for p in _workbook_paths(workbook_dir)]
    return pd.concat(frames, ignore_index=True) if frames else None


def stage_data_dir(scores, dest_dir):
    """
    Build a report data dir at dest_dir for one run: this professor's CUCEI
    scores (a DataFrame from load_workbook_scores, e.g. the Streamlit upload)
    and nothing else from CUCEI -- no other professor's workbook or CSV row is
    copied, so the run cannot show or depend on anyone else's scores. Returns
    dest_dir.

    Scores are written as CSV rows, so the uploaded file's name does not have to
    carry the professor number.
    """
    os.makedirs(os.path.join(dest_dir, "cucei"), exist_ok=True)
    scores[REQUIRED_COLUMNS].to_csv(
        os.path.join(dest_dir, "cucei_scores.csv"), index=False)
    return dest_dir


def load_all(path=DEFAULT_PATH, workbook_dir=DEFAULT_WORKBOOK_DIR):
    """
    Every professor's scores, validated, or None if no source exists yet.

    Reads the filled workbooks in data/cucei/ and data/cucei_scores.csv, whichever
    are present. Raises ValueError on a malformed source rather than letting a
    wrong number reach a professor's report.
    """
    frames = []
    from_workbooks = load_workbook_dir(workbook_dir) if workbook_dir else None
    if from_workbooks is not None:
        frames.append(from_workbooks)
    if os.path.exists(path):
        csv_df = pd.read_csv(path)
        missing = [c for c in REQUIRED_COLUMNS if c not in csv_df.columns]
        if missing:
            raise ValueError(f"{path} is missing column(s) {missing}; expected "
                             f"{REQUIRED_COLUMNS}")
        frames.append(csv_df)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df["professor_id"] = df["professor_id"].astype(str).str.strip()
    df["dimension"] = df["dimension"].astype(str).str.strip()

    unknown = sorted(set(df["dimension"]) - set(DIMENSIONS))
    if unknown:
        raise ValueError(f"unknown CUCEI dimension(s) {unknown}; expected {DIMENSIONS}")
    dupes = df[df.duplicated(["professor_id", "dimension"], keep=False)]
    if not dupes.empty:
        raise ValueError(
            f"duplicate CUCEI rows for "
            f"{sorted(set(zip(dupes.professor_id, dupes.dimension)))} -- a professor "
            f"appears in more than one workbook, or in both a workbook and {path}.")

    for col in ("mean", "sd", "n_respondents"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    out_of_scale = df[(df["mean"] < SCALE_MIN) | (df["mean"] > SCALE_MAX)]
    if not out_of_scale.empty:
        raise ValueError(
            f"CUCEI means must be on the {SCALE_MIN:g}-{SCALE_MAX:g} scale; "
            f"got {out_of_scale[['professor_id', 'dimension', 'mean']].values.tolist()}. "
            f"(Values like 24.57 are the scoring workbook's broken Mean column.)")
    return df


def load_scores(professor_id, path=DEFAULT_PATH, workbook_dir=DEFAULT_WORKBOOK_DIR):
    """
    This professor's rows in DIMENSIONS order, or None if the file or the
    professor is not there yet.
    """
    df = load_all(path, workbook_dir)
    if df is None:
        return None
    mine = df[df["professor_id"] == str(professor_id)]
    if mine.empty:
        return None
    order = {d: i for i, d in enumerate(DIMENSIONS)}
    return mine.sort_values("dimension", key=lambda s: s.map(order)).reset_index(drop=True)
