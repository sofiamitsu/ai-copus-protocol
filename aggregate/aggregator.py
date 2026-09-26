import json
import csv
import os

def aggregate_results(results_dir, output_csv, lecture_id, arm="multimodal",
                      professor_id=""):
    """
    Collects all per-chunk JSON results and writes them to a single CSV.

    Each row in the CSV represents one 2-minute window with columns:
    lecture_id, professor_id, window_index, window_start, window_end, arm,
    model, copus_codes, reasoning

    `model` is read back from each per-chunk JSON rather than passed in, so a
    results dir that mixes models stays traceable row by row.
    """
    results = []

    # Find all JSON result files in the results directory
    json_files = sorted([
        f for f in os.listdir(results_dir)
        if f.endswith(".json")
    ])

    if not json_files:
        print(f"No JSON result files found in {results_dir}")
        return

    for json_file in json_files:
        path = os.path.join(results_dir, json_file)
        with open(path, "r") as f:
            data = json.load(f)

        results.append({
            "lecture_id": lecture_id,
            "professor_id": professor_id,
            "window_index": data.get("chunk_index"),
            "window_start": data.get("window_start"),
            "window_end": data.get("window_end"),
            "arm": arm,
            "model": data.get("model", ""),
            "copus_codes": "|".join(data.get("codes_present", [])),
            "reasoning": data.get("reasoning", "")
        })

    # Write to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    fieldnames = ["lecture_id", "professor_id", "window_index", "window_start",
                  "window_end", "arm", "model", "copus_codes", "reasoning"]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    models = sorted({r["model"] for r in results if r["model"]})
    label = f" (model: {', '.join(models)})" if models else ""
    print(f"Aggregated {len(results)} windows -> {output_csv}{label}")
