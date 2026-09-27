"""
Smoke test — builds a Faculty Feedback Report PDF from existing
pipeline_test / ablation_study outputs and asserts the file is a valid,
multi-page PDF. Runs without Gemini credentials (narrative falls back to the
deterministic data-driven summary).
"""
import os

import pandas as pd

from report.pdf_report import (
    generate_faculty_report,
    code_frequency,
    active_learning_split,
    load_kappa_dict,
)

OUT = "output/pdf_test"


def test_helpers():
    df = pd.read_csv("output/pipeline_test/results_multimodal.csv")
    freq = code_frequency(df)
    assert "code" in freq.columns and "pct_of_windows" in freq.columns
    a, p = active_learning_split(df)
    assert 0 <= a <= 100 and 0 <= p <= 100
    kd = load_kappa_dict("output/ablation_study/combined_kappa.csv")
    assert kd, "expected kappa comparisons from combined_kappa.csv"
    print(f"helpers OK — {len(freq)} codes, active {a}% / passive {p}%, "
          f"{len(kd)} kappa comparisons")


def test_full_report():
    os.makedirs(OUT, exist_ok=True)
    # Use the three modality result CSVs as stand-in "lectures" so the per-
    # lecture pages exercise multiple timelines.
    lecture_results = [
        {"lecture_id": "lecture_001",
         "results_csv_path": "output/pipeline_test/results_multimodal.csv"},
        {"lecture_id": "lecture_002",
         "results_csv_path": "output/pipeline_test/results_vision_only.csv"},
        {"lecture_id": "lecture_003",
         "results_csv_path": "output/pipeline_test/results_audio_only.csv"},
    ]
    pdf_path = os.path.join(OUT, "faculty_report.pdf")
    result = generate_faculty_report(
        professor_name="Dr. Example",
        course_name="Intro to Computer Science (CS 101)",
        semester="Fall 2026",
        lecture_results=lecture_results,
        kappa_results="output/ablation_study/combined_kappa.csv",
        survey_data="output/ablation_study/survey_analysis.csv",
        output_path=pdf_path,
    )
    assert result == pdf_path
    assert os.path.exists(pdf_path), "PDF was not written"
    size = os.path.getsize(pdf_path)
    assert size > 5000, f"PDF suspiciously small ({size} bytes)"
    with open(pdf_path, "rb") as f:
        head = f.read(5)
    assert head == b"%PDF-", "output is not a valid PDF"
    print(f"full report OK — {pdf_path} ({size} bytes)")


if __name__ == "__main__":
    test_helpers()
    test_full_report()
    print("\nAll PDF report tests passed.")
