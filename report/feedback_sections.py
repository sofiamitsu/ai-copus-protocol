"""
Data for the Faculty Feedback Report sections added on 2026-09-19 (Change 9):
CUCEI perception profile, per-lecture golden reference comparisons, and behavior-perception
linking observations. Layout lives in report/pdf_report.py; this module only
computes what those sections say.

Every section degrades to a clearly-labelled placeholder when its input is not
there yet (no CUCEI scores, no golden lectures), so the report always builds.
"""
import os

import pandas as pd

from utils.cucei import (
    DIMENSION_MEANINGS, DIMENSIONS, SCALE_MIDPOINT, load_scores,
)
from utils.professor_ids import load_mapping

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "data")

# Within this distance of the comparison point a score reads as "in line with".
SAME_BAND = 0.10

# Which CUCEI dimension each COPUS behavior is paired with in the linking
# observations, and how to describe the behavior to the professor.
#
# THIS IS A DESIGN CHOICE, NOT A VALIDATED MAPPING: the CUCEI asks students about
# the course overall, not about specific observed behaviors. The pairings follow
# the dimension definitions (Fraser, Treagust & Dennis, 1986) -- e.g. MG and 1o1
# are the behaviors that most directly create the instructor-student contact
# Personalization asks about -- and the report words every link as an
# observation, never as a cause.
BEHAVIOR_LINKS = {
    "MG":  ("Personalization", "moved through the class guiding student work"),
    "1o1": ("Personalization", "held extended one-on-one discussions with students"),
    "AnQ": ("Involvement", "answered student questions"),
    "PQ":  ("Involvement", "posed questions to the class"),
    "CQ":  ("Involvement", "asked clicker or polling questions"),
    "FUp": ("Involvement", "followed up on class activities with feedback"),
    "D/V": ("Innovation", "showed demonstrations, simulations or video"),
    "RtW": ("Task Orientation", "worked through material in real time on the board"),
    "Adm": ("Task Orientation", "spent time on course logistics"),
    "Lec": ("Involvement", "lectured"),
}
# Lecturing is present in nearly every window of most classes; it is only used
# for an observation when too few other behaviors were seen to fill the section.
FALLBACK_ONLY = {"Lec"}
MAX_OBSERVATIONS = 3


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #
def resolve_identity(output_dir, meta=None):
    """
    (professor_id, professor_name, course_name) for a run's output dir.

    Values typed into the app (meta) win; lecture_professor_mapping.csv fills in
    anything left blank. This is what fixes the "Professor ( , )" header.
    """
    meta = meta or {}
    mapping = load_mapping(output_dir)
    rows = list(mapping.values())
    ids = sorted({r["professor_id"] for r in rows if r.get("professor_id")})
    if len(ids) > 1:
        print(f"[warn] {output_dir} mixes professors {ids}; the report uses {ids[0]}.")

    def first(field):
        return next((r[field] for r in rows if r.get(field)), "")

    professor_id = ids[0] if ids else ""
    name = (meta.get("professor") or "").strip() or first("professor_name")
    course = (meta.get("course") or "").strip() or first("course_name")
    return professor_id, name, course


# --------------------------------------------------------------------------- #
# Behavioral profile helpers
# --------------------------------------------------------------------------- #
def code_shares(results_df):
    """{code: % of windows the code appears in} for a pooled results DataFrame."""
    n = len(results_df)
    if not n:
        return {}
    counts = {}
    for cell in results_df.get("copus_codes", pd.Series(dtype=str)).fillna(""):
        for code in {c.strip() for c in str(cell).split("|") if c.strip()}:
            counts[code] = counts.get(code, 0) + 1
    return {c: round(100.0 * k / n, 1) for c, k in counts.items()}


# --------------------------------------------------------------------------- #
# Golden reference lectures
# --------------------------------------------------------------------------- #
DEFAULT_GOLDEN_DIR = os.path.join(REPO_ROOT, "golden")
GOLDEN_MANIFEST = "golden_lectures.csv"
GOLDEN_RESULTS = "results_multimodal.csv"


def golden_lectures(golden_dir=DEFAULT_GOLDEN_DIR):
    """
    One behavioral profile per golden reference lecture, in manifest order, or []
    if none are available.

    Each golden lecture is by a different instructor, so they are NEVER pooled:
    the report compares the professor with each one separately. Lectures come
    from <golden_dir>/golden_lectures.csv (folder, professor_name, lecture_title,
    youtube_url); a results folder not listed there is ignored, so every
    comparison carries a name and a link the professor can follow.

    Shares use the same "% of windows" measure as the professor's own profile,
    so the two are directly comparable (both AI multimodal coding).

    Returns [{"professor_name", "lecture_title", "youtube_url", "shares",
              "n_windows", "results"}].
    """
    manifest = os.path.join(golden_dir, GOLDEN_MANIFEST)
    if not os.path.exists(manifest):
        return []
    rows = pd.read_csv(manifest, dtype=str).fillna("")
    out = []
    for _, row in rows.iterrows():
        folder = row.get("folder", "").strip()
        path = os.path.join(golden_dir, folder, GOLDEN_RESULTS)
        try:
            df = pd.read_csv(path)
        except Exception as e:  # noqa: BLE001 -- one bad file must not sink the report
            print(f"[warn] skipping golden lecture '{folder}': {e}")
            continue
        if df.empty:
            print(f"[warn] skipping golden lecture '{folder}': {path} has no windows")
            continue
        out.append({
            "professor_name": row.get("professor_name", "").strip(),
            "lecture_title": row.get("lecture_title", "").strip(),
            "youtube_url": row.get("youtube_url", "").strip(),
            "shares": code_shares(df),
            "n_windows": len(df),
            "results": df,
        })
    return out


# --------------------------------------------------------------------------- #
# CUCEI
# --------------------------------------------------------------------------- #
def cucei_profile(professor_id, data_dir=DEFAULT_DATA_DIR):
    """
    Rows for the CUCEI section, or None if scores are not available yet.

    Each row: dimension, mean, sd, n, interpretation.

    Only this professor's own scores are used. Scores are read against the fixed
    scale midpoint, never against other professors: a report must not reveal,
    or be shaped by, anyone else's results.
    """
    path = os.path.join(data_dir, "cucei_scores.csv")
    workbooks = os.path.join(data_dir, "cucei")
    if not professor_id:
        return None
    scores = load_scores(professor_id, path, workbooks)
    if scores is None:
        return None
    rows = []
    for _, r in scores.iterrows():
        dim = r["dimension"]
        rows.append({
            "dimension": dim,
            "mean": r["mean"],
            "sd": r["sd"],
            "n": int(r["n_respondents"]) if pd.notna(r["n_respondents"]) else None,
            "interpretation": interpret_dimension(dim, r["mean"]),
        })
    return rows


def _comparison(mean):
    """(direction, phrase) comparing a score to the scale midpoint."""
    label = f"the scale midpoint ({SCALE_MIDPOINT:.1f})"
    diff = mean - SCALE_MIDPOINT
    if abs(diff) < SAME_BAND:
        return "same", f"in line with {label}"
    return ("above", f"above {label}") if diff > 0 else ("below", f"below {label}")


def interpret_dimension(dim, mean):
    """One plain-language line for a CUCEI dimension score."""
    if mean is None or pd.isna(mean):
        return "No complete responses for this dimension."
    _, phrase = _comparison(float(mean))
    meaning = DIMENSION_MEANINGS[dim]
    return f"{meaning[0].upper()}{meaning[1:]} — {phrase}."


# --------------------------------------------------------------------------- #
# Behavior-perception linking
# --------------------------------------------------------------------------- #
def linking_observations(shares, cucei_rows):
    """
    Up to MAX_OBSERVATIONS sentences pairing an observed behavior with the CUCEI
    dimension it relates to. Returns [] when there are no CUCEI scores.

    Behaviors are taken most-frequent first, one per CUCEI dimension, skipping
    lecturing unless nothing else fills the section. Every sentence is worded as
    an observation, never a cause.
    """
    if not cucei_rows:
        return []
    by_dim = {r["dimension"]: r for r in cucei_rows}

    def candidates(allow_fallback):
        ranked = sorted(((shares.get(code, 0.0), code) for code in BEHAVIOR_LINKS),
                        reverse=True)
        return [(pct, code) for pct, code in ranked
                if pct > 0 and (allow_fallback or code not in FALLBACK_ONLY)
                and BEHAVIOR_LINKS[code][0] in by_dim]

    chosen, used_dims = [], set()
    for allow_fallback in (False, True):
        for pct, code in candidates(allow_fallback):
            dim = BEHAVIOR_LINKS[code][0]
            if dim in used_dims or len(chosen) >= MAX_OBSERVATIONS:
                continue
            chosen.append((pct, code))
            used_dims.add(dim)
        if len(chosen) >= 2:
            break

    out = []
    for pct, code in chosen:
        dim, phrase = BEHAVIOR_LINKS[code]
        row = by_dim[dim]
        direction, cmp_phrase = _comparison(float(row["mean"]))
        if direction == "above":
            reading = f" This may be part of why students rate {dim} favorably."
        elif direction == "below":
            reading = " It may be worth asking students how this comes across."
        else:
            reading = ""
        out.append(
            f"You {phrase} ({code}) in {pct:g}% of analyzed segments. Your "
            f"<b>{dim}</b> score is {float(row['mean']):.2f}/4, {cmp_phrase}.{reading}")
    return out


# --------------------------------------------------------------------------- #
# CUCEI-informed recommendation
# --------------------------------------------------------------------------- #
# What to suggest when a dimension is rated low. The behaviors named are the ones
# BEHAVIOR_LINKS already pairs with that dimension, so section 5 and the
# recommendations never disagree about which behavior relates to which score.
# Dimensions with no linked instructor behavior (Student Cohesiveness,
# Satisfaction, Individualization) get no CUCEI-based suggestion: COPUS gives no
# observed behavior to point to.
SUGGESTIONS = {
    "Personalization": "circulating among students while they work, or brief "
                       "one-on-one check-ins, creates the direct contact this "
                       "dimension asks about",
    "Involvement": "posing questions to the whole class, and following up on the "
                   "answers, gives more students a way to take part",
    "Innovation": "an occasional demonstration, simulation or short video varies "
                  "the format of class",
    "Task Orientation": "working through a problem step by step on the board makes "
                        "the structure of the work visible",
}
# Linked behaviors never worth recommending more of.
NOT_SUGGESTED = FALLBACK_ONLY | {"Adm"}

# If any linked behavior already appears in at least this share of segments,
# "do more of it" is not grounded advice -- the low score likely reflects
# something the observation cannot see, so the suggestion says that instead.
ALREADY_COMMON_PCT = 25.0


def cucei_context_lines(cucei_rows):
    """CUCEI scores with their comparison, as prompt context for Gemini."""
    lines = []
    for r in cucei_rows or []:
        if r["mean"] is None or pd.isna(r["mean"]):
            continue
        _, phrase = _comparison(float(r["mean"]))
        lines.append(f"- CUCEI {r['dimension']}: {float(r['mean']):.2f}/4 "
                     f"(n={r['n']}), {phrase}")
    return lines


def cucei_recommendation(shares, cucei_rows):
    """
    One suggestion tied to the lowest-rated CUCEI dimension that sits below its
    comparison point and has a linked behavior, or None.

    Names how often the linked behaviors were actually seen, so the suggestion is
    grounded in this professor's own data rather than generic advice. When those
    behaviors were already common, it does NOT recommend more of them.
    """
    below = []
    for r in cucei_rows or []:
        if r["dimension"] not in SUGGESTIONS or r["mean"] is None or pd.isna(r["mean"]):
            continue
        direction, phrase = _comparison(float(r["mean"]))
        if direction == "below":
            below.append((float(r["mean"]) - SCALE_MIDPOINT, r, phrase))
    if not below:
        return None
    _, row, phrase = min(below, key=lambda b: b[0])
    dim = row["dimension"]
    codes = [c for c, (d, _) in BEHAVIOR_LINKS.items()
             if d == dim and c not in NOT_SUGGESTED]
    seen = ", ".join(f"{c} in {shares.get(c, 0):g}%" for c in codes)
    head = f"Students rated {dim} {float(row['mean']):.2f}/4, {phrase}."
    if max((shares.get(c, 0) for c in codes), default=0) >= ALREADY_COMMON_PCT:
        return (f"{head} The behaviors most related to it were already common in "
                f"your lectures ({seen}), so more of them is unlikely to be the "
                f"answer; asking students about {DIMENSION_MEANINGS[dim]} may tell "
                f"you more than this analysis can.")
    return (f"{head} Related behaviors appeared in few of the analyzed segments "
            f"({seen}); {SUGGESTIONS[dim]}. This is a possibility to explore with "
            f"your students, not a diagnosis.")
