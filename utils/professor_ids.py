"""
Professor identity for every output row.

Professor identity used to live only in folder names ("PROFESSOR 1 (DR RODRIGO)",
"professor 8"), so any CSV that left its folder -- pooled into a thesis table,
unzipped next to another professor's -- lost track of whose lecture it was.
This module derives a stable professor_id from the lecture's path and keeps a
lecture -> professor mapping file for downstream joins.

  /Users/.../PROFESSOR 1 (DR RODRIGO)/Subject1_Lecture1.mp4
      -> professor_id "professor_1", professor_name "Dr. Rodrigo"

Resolution order (see resolve_professor):
  1. an explicit professor_id passed by the caller (CLI flag / UI field)
  2. the nearest "professor N" folder in the lecture path
  3. a slug of the professor name, with a warning -- this is what a Streamlit
     upload gets, since uploads land in a temp dir with no professor folder
"""
import os
import re
import threading

import pandas as pd

MAPPING_FILENAME = "lecture_professor_mapping.csv"
MAPPING_COLUMNS = ["lecture_id", "professor_id", "professor_name", "course_name"]

# Lectures run in parallel and all read-modify-write the same mapping file.
_mapping_lock = threading.Lock()

# "PROFESSOR 1 (DR RODRIGO)", "professor 8", "Professor_3", "prof-2 (Dr. X)"
_FOLDER_RE = re.compile(
    r"^prof(?:essor)?[\s_-]*(\d+)\s*(?:\((?P<name>[^)]*)\))?\s*$", re.IGNORECASE)


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")


def _pretty_name(raw):
    """'DR RODRIGO' -> 'Dr. Rodrigo'."""
    words = [w.capitalize() for w in raw.replace(".", " ").split()]
    if words and words[0] == "Dr":
        words[0] = "Dr."
    return " ".join(words)


def parse_professor_folder(path):
    """
    (professor_id, professor_name) from the nearest 'professor N' folder that
    contains `path`, or (None, None) if no ancestor folder matches.
    """
    if not path:
        return None, None
    parent = os.path.dirname(os.path.abspath(os.path.expanduser(path)))
    while True:
        m = _FOLDER_RE.match(os.path.basename(parent))
        if m:
            name = _pretty_name(m.group("name")) if m.group("name") else ""
            return f"professor_{int(m.group(1))}", name
        up = os.path.dirname(parent)
        if up == parent:
            return None, None
        parent = up


def resolve_professor(lecture_path, professor_id=None, professor_name=None):
    """
    The (professor_id, professor_name) to stamp on a lecture's outputs.
    Never returns an empty professor_id.
    """
    folder_id, folder_name = parse_professor_folder(lecture_path)
    name = (professor_name or "").strip() or folder_name or ""

    if professor_id:
        if folder_id and folder_id != professor_id:
            print(f"[warn] professor_id '{professor_id}' was given explicitly but "
                  f"{lecture_path} sits in a '{folder_id}' folder; using "
                  f"'{professor_id}'.")
        return professor_id, name
    if folder_id:
        return folder_id, name

    fallback = slug(name) or "unknown_professor"
    print(f"[warn] no 'professor N' folder in the path of {lecture_path}; "
          f"professor_id falls back to '{fallback}'. Pass a professor id, or run "
          f"from the lecture's professor folder, so ids stay consistent across runs.")
    return fallback, name


def upsert_mapping(output_dir, lecture_id, professor_id, professor_name="",
                   course_name=""):
    """
    Add or replace this lecture's row in <output_dir>/lecture_professor_mapping.csv.

    Keyed on lecture_id, so lecture ids must be unique across professors --
    two professors' "lecture_1" would overwrite each other here.
    """
    with _mapping_lock:
        return _upsert_mapping(output_dir, lecture_id, professor_id,
                               professor_name, course_name)


def _upsert_mapping(output_dir, lecture_id, professor_id, professor_name,
                    course_name):
    path = os.path.join(output_dir, MAPPING_FILENAME)
    row = {"lecture_id": lecture_id, "professor_id": professor_id,
           "professor_name": professor_name or "", "course_name": course_name or ""}
    if os.path.exists(path):
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        clash = df[(df["lecture_id"] == lecture_id) & (df["professor_id"] != professor_id)]
        if not clash.empty:
            print(f"[warn] {MAPPING_FILENAME}: lecture_id '{lecture_id}' was mapped "
                  f"to '{clash['professor_id'].iloc[0]}' and is now remapped to "
                  f"'{professor_id}'. Lecture ids must be unique across professors.")
        df = df[df["lecture_id"] != lecture_id]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    os.makedirs(output_dir or ".", exist_ok=True)
    df = df.reindex(columns=MAPPING_COLUMNS).sort_values(["professor_id", "lecture_id"])
    df.to_csv(path, index=False)
    return path


def load_mapping(output_dir):
    """{lecture_id: row dict} from a mapping file, or {} if there is none."""
    path = os.path.join(output_dir, MAPPING_FILENAME)
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return {r["lecture_id"]: r for r in df.to_dict("records")}
