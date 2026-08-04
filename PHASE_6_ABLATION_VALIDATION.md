# Phase 6 — Full Ablation Validation

## Context

Phase 5 built 3 additional classification arms (vision-only, audio-only, transcript-only) alongside the existing multimodal arm. This phase runs all 4 arms on ground truth lectures and produces the comparison table that answers "which modality contributes most to COPUS classification accuracy?"

## Prerequisites

- Phase 5 complete — all 4 arms produce valid JSON output
- At least 2 ground truth lectures coded by hand (human coding CSVs exist)
- `validate/validator.py` working with `compute_kappa()`

## What to Build

### 1. Batch Runner Script

**File:** `run_ablation.py` (root of THESIS folder)

**Purpose:** Run all 4 arms on all ground truth lectures in one command.

**Usage:**
```bash
uv run python run_ablation.py \
  --lectures lecture_001.mp4 lecture_002.mp4 \
  --lecture-ids lecture_001 lecture_002 \
  --output-dir output/ablation_study
```

**Logic:**
- For each lecture:
  - Chunk it (skip if chunks already exist)
  - Run all 4 arms on all chunks
  - Save results per arm to `output/ablation_study/{lecture_id}/results_{arm}/`
  - Aggregate per arm to `output/ablation_study/{lecture_id}/results_{arm}.csv`

### 2. Comparison Table Generator

**File:** Add `compute_comparison_table()` to `validate/validator.py`

**Purpose:** Compute κ for each (code × arm) combination and output a single comparison table.

**Input:**
- List of arm result CSVs
- Human ground truth CSV
- Output path

**Output CSV format:**
```
code,multimodal_kappa,vision_only_kappa,audio_only_kappa,transcript_only_kappa
Lec,0.75,0.40,0.60,0.55
RtW,0.70,0.65,N/A,N/A
PQ,0.72,N/A,0.68,0.70
...
```

Also print a formatted table to the terminal.

### 3. Updated Dashboard

**File:** Update `report/dashboard.py`

**Add:** `generate_comparison_dashboard()` that shows:
- 4 behavioral timelines stacked (one per arm) for side-by-side visual comparison
- A bar chart of κ per code per arm
- Save as `comparison_dashboard.html`

## Testing

1. Run `run_ablation.py` on 2 ground truth lectures
2. Verify all 4 arms produce results for all windows
3. Generate comparison table — verify κ values are plausible
4. Generate comparison dashboard — verify it renders correctly

## Definition of Done

- [ ] `run_ablation.py` processes multiple lectures × all 4 arms in one command
- [ ] Comparison table CSV generated with κ per code per arm
- [ ] Comparison dashboard HTML shows stacked timelines + bar chart
- [ ] Results validated on at least 2 ground truth lectures
