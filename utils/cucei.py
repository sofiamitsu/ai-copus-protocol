"""
CUCEI scores: loading and sanity-checking data/cucei_scores.csv.

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
import os

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
REQUIRED_COLUMNS = ["professor_id", "dimension", "mean", "sd", "n_respondents"]

DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "cucei_scores.csv")


def load_all(path=DEFAULT_PATH):
    """
    Every professor's scores, validated, or None if the file does not exist yet.

    Raises ValueError on a malformed file rather than letting a wrong number
    reach a professor's report.
    """
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing column(s) {missing}; expected "
                         f"{REQUIRED_COLUMNS}")
    df["professor_id"] = df["professor_id"].astype(str).str.strip()
    df["dimension"] = df["dimension"].astype(str).str.strip()

    unknown = sorted(set(df["dimension"]) - set(DIMENSIONS))
    if unknown:
        raise ValueError(f"{path}: unknown CUCEI dimension(s) {unknown}; "
                         f"expected {DIMENSIONS}")
    dupes = df[df.duplicated(["professor_id", "dimension"], keep=False)]
    if not dupes.empty:
        raise ValueError(f"{path}: duplicate rows for "
                         f"{sorted(set(zip(dupes.professor_id, dupes.dimension)))}")

    for col in ("mean", "sd", "n_respondents"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    out_of_scale = df[(df["mean"] < SCALE_MIN) | (df["mean"] > SCALE_MAX)]
    if not out_of_scale.empty:
        raise ValueError(
            f"{path}: CUCEI means must be on the {SCALE_MIN:g}-{SCALE_MAX:g} scale; "
            f"got {out_of_scale[['professor_id', 'dimension', 'mean']].values.tolist()}. "
            f"(Values like 24.57 are the scoring workbook's broken Mean column.)")
    return df


def load_scores(professor_id, path=DEFAULT_PATH):
    """
    This professor's rows in DIMENSIONS order, or None if the file or the
    professor is not there yet.
    """
    df = load_all(path)
    if df is None:
        return None
    mine = df[df["professor_id"] == str(professor_id)]
    if mine.empty:
        return None
    order = {d: i for i, d in enumerate(DIMENSIONS)}
    return mine.sort_values("dimension", key=lambda s: s.map(order)).reset_index(drop=True)


def study_averages(path=DEFAULT_PATH):
    """
    {dimension: (mean of professors' means, number of professors)} across every
    professor in the file -- the "study average" a report compares against.
    Each professor counts once, regardless of class size.
    """
    df = load_all(path)
    if df is None:
        return {}
    out = {}
    for dim, g in df.dropna(subset=["mean"]).groupby("dimension"):
        out[dim] = (round(float(g["mean"].mean()), 2), int(g["professor_id"].nunique()))
    return out
