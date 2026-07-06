import json
import csv
import os

def aggregate_results(chunks_dir, output_csv, lecture_id, arm="multimodal"):
    """
    Collects all per-chunk JSON results and writes them to a single CSV.
    
    Each row in the CSV represents one 2-minute window with columns:
    lecture_id, window_index, window_start, window_end, arm, copus_codes, reasoning
    """
    results = []

    # Find all JSON result files in the chunks directory
    json_files = sorted([
        f for f in os.listdir(chunks_dir)
        if f.endswith(".json")
    ])

    if not json_files:
        print(f"No JSON result files found in {chunks_dir}")
        return

    for json_file in json_files:
        path = os.path.join(chunks_dir, json_file)
        with open(path, "r") as f:
            data = json.load(f)

        results.append({
            "lecture_id": lecture_id,
            "window_index": data.get("chunk_index"),
            "window_start": data.get("window_start"),
            "window_end": data.get("window_end"),
            "arm": arm,
            "copus_codes": "|".join(data.get("codes_present", [])),
            "reasoning": data.get("reasoning", "")
        })

    # Write to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    fieldnames = ["lecture_id", "window_index", "window_start", 
                  "window_end", "arm", "copus_codes", "reasoning"]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Aggregated {len(results)} windows → {output_csv}")