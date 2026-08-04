"""
Thin adapter between the Streamlit UI (app.py) and the Phase 9 PDF builder.

app.py's `_try_generate_pdf(output_dir, meta)` hook expects a
`report.faculty_report.generate_report(output_dir, pdf_path, meta)` entry
point — a simpler interface than `pdf_report.generate_faculty_report`, since
the UI only has a `run.py`-style output directory and a small metadata dict
on hand, not pre-built lecture/kappa/survey structures.
"""
import os

from report.pdf_report import generate_faculty_report, _discover_lectures


def generate_report(output_dir, pdf_path, meta):
    """
    Build the Faculty Feedback Report PDF from a `run.py` output directory.

    `meta` is the dict app.py keeps in `st.session_state.meta`:
        {"professor": ..., "course": ..., "semester": ..., "arm": ..., "primary_arm": ...}
    """
    lecture_results = _discover_lectures(output_dir)
    if not lecture_results:
        raise FileNotFoundError(
            f"no lecture results CSVs found under {output_dir}")

    kappa_csv = os.path.join(output_dir, "combined_kappa.csv")
    survey_csv = os.path.join(output_dir, "survey_analysis.csv")

    return generate_faculty_report(
        professor_name=meta.get("professor") or "Professor",
        course_name=meta.get("course") or "",
        semester=meta.get("semester") or "",
        lecture_results=lecture_results,
        kappa_results=kappa_csv if os.path.exists(kappa_csv) else None,
        survey_data=survey_csv if os.path.exists(survey_csv) else None,
        output_path=pdf_path,
    )
