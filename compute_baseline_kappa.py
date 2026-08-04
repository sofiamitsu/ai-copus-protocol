"""
One-off: compute Cohen's kappa on the FULL existing multimodal run.

WHY THIS EXISTS
---------------
On June 28 the pipeline was validated end-to-end and a kappa was recorded.
But that kappa was computed on a 3-window smoke run --- look at
output/pipeline_test/kappa_multimodal/kappa_results.csv: the
human_positive_windows column maxes out at 2. Kappa on n=3 windows is noise,
not a result.

Meanwhile, a full multimodal pass over lecture_001 already ran and left 24
per-chunk JSONs sitting in output/pipeline_test/results/ (chunk_000 through
chunk_023). Those have never been scored against your human coding.

human_coding_lecture_001.csv has 30 windows. compute_kappa() only compares
windows present in BOTH inputs, so the 24-window overlap is what gets used.

That is a real kappa on ~48 minutes of lecture, from data you already paid
Gemini for. This script makes zero API calls and costs nothing.

WHAT TO DO WITH THE NUMBER
--------------------------
Your RQ1 threshold is kappa >= 0.7 on major codes. Read the result as:

  kappa >= 0.7 on Lec/RtW  -> prompt is working. Saturday's coding is the
                              only thing standing between you and results.
  kappa 0.4 - 0.7          -> prompt needs iteration (Phase 4b, still open).
                              Better to learn this now than in October.
  kappa < 0.4              -> stop and diagnose before coding 9 more lectures
                              by hand. Do not paper over it with majority vote.

Ignore codes where human_positive_windows is 0 or 1 --- kappa is unstable at
that count and will show alarming values like -0.5 that mean nothing. Judge on
Lec and RtW, which have real support in this lecture.

RUN:
    uv run python compute_baseline_kappa.py
"""

import os

from aggregate.aggregator import aggregate_results
from validate.validator import compute_kappa

RESULTS_DIR = "output/pipeline_test/results"
HUMAN_CSV = "human_coding_lecture_001.csv"
OUT_DIR = "output/pipeline_test/kappa_full_multimodal"
AI_CSV = os.path.join(OUT_DIR, "results_full_multimodal.csv")


def main():
    if not os.path.isdir(RESULTS_DIR):
        raise SystemExit(f"Missing {RESULTS_DIR} --- run from the repo root.")
    if not os.path.isfile(HUMAN_CSV):
        raise SystemExit(f"Missing {HUMAN_CSV} --- run from the repo root.")

    os.makedirs(OUT_DIR, exist_ok=True)

    n_json = len([f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")])
    print(f"Found {n_json} per-chunk result JSONs in {RESULTS_DIR}\n")

    # Step 1 --- collapse the per-chunk JSONs into one long-format CSV.
    # Same function run.py uses, so the format matches what compute_kappa expects.
    aggregate_results(
        chunks_dir=RESULTS_DIR,
        output_csv=AI_CSV,
        lecture_id="lecture_001",
        arm="multimodal",
    )

    # Step 2 --- score AI against human coding.
    # csv_a is counted as 'human_positive_windows', csv_b as 'ai_positive_windows',
    # so human MUST be csv_a for the column names to mean what they say.
    print()
    df = compute_kappa(
        csv_a=HUMAN_CSV,
        csv_b=AI_CSV,
        output_dir=OUT_DIR,
        output_filename="kappa_full_multimodal.csv",
        label_a="Sofia (human)",
        label_b="AI multimodal (gemini-2.5-flash)",
    )

    print("\n" + "=" * 60)
    print("BASELINE KAPPA --- full multimodal run, lecture_001")
    print("=" * 60)
    print(df.to_string(index=False))
    print()
    print("Judge on rows where human_positive_windows >= 3.")
    print("Log the result in Notion -> Thesis Tracker -> kappa results log.")


if __name__ == "__main__":
    main()
