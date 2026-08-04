# Phase 7 — `run.py` CLI Entry Point

## Context

Currently the pipeline requires running multiple separate scripts manually. This phase creates a single CLI entry point that processes one professor's full data bundle — 3 lectures, 2 sets of manual COPUS coding, and 1 student survey.

## Inputs (one professor bundle)

```
run.py \
  --professor "Dr. Smith" \
  --lectures lecture_A.mp4 lecture_B.mp4 lecture_C.mp4 \
  --lecture-ids lecture_A lecture_B lecture_C \
  --sofia-coding sofia_A.xlsx sofia_B.xlsx sofia_C.xlsx \
  --kaw-coding kaw_A.xlsx kaw_B.xlsx kaw_C.xlsx \
  --survey survey_results.csv \
  --output-dir output/dr_smith \
  --arm multimodal
```

All arguments except `--survey` are required. `--arm` defaults to `multimodal` but accepts `all` to run ablation study.

## Pipeline Steps (per professor)

For each of the 3 lectures:

1. **Chunk** — `chunk/chunker.py` → 2-min windows
2. **Classify** — `classify/classifier.py` → AI COPUS codes (selected arm or all arms)
3. **Aggregate** — `aggregate/aggregator.py` → `results_{arm}.csv`
4. **Convert human coding** — `validate/convert_copus_sheet.py` → long-format CSVs for both Sofia and Dr. Kaw
5. **Validate** — `validate/validator.py`:
   - AI vs Sofia κ
   - AI vs Dr. Kaw κ
   - Sofia vs Dr. Kaw κ (inter-rater reliability)
6. **Dashboard** — `report/dashboard.py` → per-lecture behavioral timeline

After all 3 lectures:

7. **Combined κ table** — aggregate κ scores across all 3 lectures
8. **Professor-level dashboard** — combined behavioral profile
9. **Copy all outputs** to `output/{professor}/` folder

## Output Folder Structure

```
output/dr_smith/
├── lecture_A/
│   ├── chunks/              (2-min window files)
│   ├── results_multimodal/  (per-chunk JSON)
│   ├── results_multimodal.csv
│   ├── human_sofia.csv
│   ├── human_kaw.csv
│   ├── kappa_ai_vs_sofia.csv
│   ├── kappa_ai_vs_kaw.csv
│   ├── kappa_sofia_vs_kaw.csv
│   └── dashboard.html
├── lecture_B/
│   └── (same structure)
├── lecture_C/
│   └── (same structure)
├── combined_kappa.csv
├── professor_dashboard.html
└── survey_analysis.csv     (if survey provided)
```

## Validator Update

`validate/validator.py` currently only compares AI vs human. Needs a small update to also support human vs human comparison (Sofia vs Dr. Kaw). The function signature is the same — `compute_kappa(csv_a, csv_b, output_dir)` — just different inputs.

## Error Handling

- If a Gemini call fails on a chunk, log the error and skip that chunk (don't crash the full run)
- If a human coding Excel doesn't have enough windows to match the lecture, warn and compare available windows only
- Print a progress summary at the end: X chunks processed, Y errors, κ scores

## Dependencies

No new packages needed — uses existing modules.

## Definition of Done

- [ ] `run.py` processes a full 3-lecture professor bundle in one command
- [ ] Produces per-lecture results + combined professor-level outputs
- [ ] Computes 3 κ comparisons per lecture (AI-Sofia, AI-Kaw, Sofia-Kaw)
- [ ] Handles errors gracefully (skips failed chunks, warns on mismatches)
- [ ] Tested on at least 1 full professor bundle
