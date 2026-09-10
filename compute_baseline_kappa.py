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
    uv run python compute_baseline_kappa.py --lecture-id sandel \
        --results-dir output/sandel/results_multimodal
"""

import argparse
import os

from aggregate.aggregator import aggregate_results
from validate.validator import compute_kappa

DEFAULT_LECTURE_ID = "lecture_001"
DEFAULT_RESULTS_DIR = "output/pipeline_test/results"
DEFAULT_OUT_ROOT = "output/pipeline_test/kappa_full_multimodal"


def main(lecture_id, results_dir, human_csv, out_root):
    if not os.path.isdir(results_dir):
        raise SystemExit(f"Missing {results_dir} --- run from the repo root.")
    if not os.path.isfile(human_csv):
        raise SystemExit(f"Missing {human_csv} --- run from the repo root.")

    # Scope every output under the lecture id: scoring a second lecture used to
    # overwrite the first one's CSVs, because both the directory and the
    # lecture_id stamped into each row were hardcoded to lecture_001.
    out_dir = os.path.join(out_root, lecture_id)
    ai_csv = os.path.join(out_dir, f"results_full_multimodal_{lecture_id}.csv")
    os.makedirs(out_dir, exist_ok=True)

    n_json = len([f for f in os.listdir(results_dir) if f.endswith(".json")])
    print(f"Lecture: {lecture_id}")
    print(f"Found {n_json} per-chunk result JSONs in {results_dir}\n")

    # Step 1 --- collapse the per-chunk JSONs into one long-format CSV.
    # Same function run.py uses, so the format matches what compute_kappa expects.
    aggregate_results(
        results_dir=results_dir,
        output_csv=ai_csv,
        lecture_id=lecture_id,
        arm="multimodal",
    )

    # Step 2 --- score AI against human coding.
    # csv_a is counted as 'human_positive_windows', csv_b as 'ai_positive_windows',
    # so human MUST be csv_a for the column names to mean what they say.
    print()
    df = compute_kappa(
        csv_a=human_csv,
        csv_b=ai_csv,
        output_dir=out_dir,
        output_filename=f"kappa_full_multimodal_{lecture_id}.csv",
        label_a="Sofia (human)",
        label_b="AI multimodal (gemini-2.5-flash)",
    )

    print("\n" + "=" * 60)
    print(f"BASELINE KAPPA --- full multimodal run, {lecture_id}")
    print("=" * 60)
    print(df.to_string(index=False))
    print()
    print("Judge on rows where human_positive_windows >= 3.")
    print("Log the result in Notion -> Thesis Tracker -> kappa results log.")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lecture-id", default=DEFAULT_LECTURE_ID)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR,
                        help="Directory of per-chunk result JSONs")
    parser.add_argument("--human-csv", default=None,
                        help="Human coding CSV; defaults to human_coding_<lecture-id>.csv")
    parser.add_argument("--output-dir", default=DEFAULT_OUT_ROOT)
    args = parser.parse_args()

    main(
        lecture_id=args.lecture_id,
        results_dir=args.results_dir,
        human_csv=args.human_csv or f"human_coding_{args.lecture_id}.csv",
        out_root=args.output_dir,
    )
