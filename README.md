## Project

**Bridging the Feedback Gap: Enhancing STEM Instruction through AI-Driven Classroom Analytics**

An AI-driven pipeline that automates the COPUS (Classroom Observation Protocol for Undergraduate STEM) protocol by analyzing lecture videos with Google Gemini and comparing AI classifications against human raters using Cohen's Kappa.

## Tech Stack

| Layer | Choice |
|---|---|
| Language | Python 3.9+ (uv for package management) |
| Video processing | `ffmpeg-python` + system `ffmpeg` |
| LLM | Google Gemini 2.5 Flash (dev) / Pro (final) via `google-genai` SDK |
| Auth | Service account JSON via ADC (`GOOGLE_APPLICATION_CREDENTIALS` env var) |
| PII scrubbing | Microsoft Presidio (local) |
| Stats | `scikit-learn` (`cohen_kappa_score`), `pandas` |
| Visualization | `plotly` |
| PDF generation | `reportlab` + `kaleido` |
| UI | Streamlit (local only) |
| Config | `.env` loaded via `python-dotenv` |

## Current Folder Structure

```
THESIS/
├── chunk/
│   ├── __init__.py
│   └── chunker.py          # chunk_video() — splits .mp4 into 2-min windows
│                            #   (full.mp4 / muted.mp4 / audio.mp3 per window)
├── classify/
│   ├── __init__.py
│   └── classifier.py       # get_gemini_client() + classify_chunk_multimodal()
│                            #   and the 3 ablation-arm variants (vision/audio/transcript-only)
├── scrub/
│   ├── __init__.py
│   └── scrubber.py         # scrub_transcript() — local Presidio PII scrubbing
├── aggregate/
│   ├── __init__.py
│   └── aggregator.py       # aggregate_results() — per-chunk JSON → results.csv
├── report/
│   ├── __init__.py
│   ├── dashboard.py         # generate_dashboard() / generate_comparison_dashboard()
│   ├── pdf_report.py         # Phase 9 — generate_faculty_report() → PDF
│   └── faculty_report.py     # adapter: generate_report(output_dir, pdf_path, meta), used by app.py
├── validate/
│   ├── __init__.py
│   ├── validator.py         # compute_kappa(), compute_comparison_table() — Cohen's κ
│   └── convert_copus_sheet.py  # Excel template → long-format CSV
├── output/                  # all pipeline outputs go here (per professor/lecture)
├── run.py                   # Phase 7 — CLI entry point, one full professor bundle
├── run_ablation.py          # Phase 6 — batch runner across all 4 modality arms
├── app.py                   # Phase 8 — Streamlit UI wrapping run.py
├── test_chunk.py
├── test_classify.py
├── test_pipeline.py         # end-to-end: chunk → classify → aggregate (+ ARM_CONFIG)
├── test_validate.py
├── test_pdf_report.py       # Phase 9 smoke test
├── gcp-key.json             # (gitignored) service account key
├── .env                     # GOOGLE_APPLICATION_CREDENTIALS, PROJECT, LOCATION
├── .gitignore
└── pyproject.toml
```

## COPUS Codes (Instructor Only — 12 codes)

| Code | Meaning |
|---|---|
| Lec | Lecturing — presenting content |
| RtW | Real-time writing on board/doc camera |
| FUp | Follow-up/feedback on activity to class |
| PQ | Posing non-clicker question (non-rhetorical) |
| CQ | Asking a clicker question |
| AnQ | Answering student questions with class listening |
| MG | Moving through class guiding student work |
| 1o1 | One-on-one extended discussion |
| D/V | Showing demo, experiment, simulation, video |
| Adm | Administration — assign homework, return tests |
| W | Waiting when instructor could be engaging |
| O | Other — explain in comments |

## Phase Map

| Phase | Status | File |
|---|---|---|
| 0–4 | ✅ Done | (built in prior sessions) |
| 4b | 🟡 In progress | Prompt iteration (manual, not a Claude Code task) |
| 5 | ✅ Done | `PHASE_5_ABLATION_ARMS.md` → `ARM_CONFIG` in `test_pipeline.py`, arm fns in `classify/classifier.py` |
| 6 | ✅ Done | `PHASE_6_ABLATION_VALIDATION.md` → `run_ablation.py`, `compute_comparison_table` in `validate/validator.py` |
| 7 | ✅ Done | `PHASE_7_CLI_ENTRY_POINT.md` → `run.py` |
| 8 | ✅ Done | `PHASE_8_STREAMLIT_UI.md` → `app.py` |
| 9 | ✅ Done | `PHASE_9_PDF_REPORT.md` → `report/pdf_report.py` (+ `report/faculty_report.py` adapter for `app.py`) |
| 10 | ⬜ Next | `PHASE_10_REAL_DATA_RUNS.md` (operational, not code) |

> ⚠️ **Before Phase 10:** see `FOLLOWUPS.md` — two open issues (multimodal prompt
> missing CQ/1o1/W, and human coding hard-capped at 30 windows) affect the
> validity of κ results.

## Execution Order

Phases must be built in order — each depends on the prior:
1. Phase 5 (ablation arms) — extends `classify/`
2. Phase 6 (ablation validation) — uses Phase 5 output
3. Phase 7 (`run.py`) — orchestrates all modules
4. Phase 8 (Streamlit) — wraps `run.py` in UI
5. Phase 9 (PDF report) — generates downloadable deliverable
6. Phase 10 (real runs) — execution, not development

## Key Constraints

- **ZDR**: All Gemini calls use Vertex AI with project-level caching disabled
- **No cloud storage for video**: Raw `.mp4` stays on local machine + USF Box only
- **Presidio runs locally**: PII scrubbing never sends transcript to cloud
- **IRB compliance**: De-identified data only in outputs. No student names, no course numbers in exports.
- **Model**: `gemini-2.5-flash` for dev, `gemini-2.5-pro` for final validation

## How to Run

### 0. One-time setup

```bash
uv sync
```

Confirm `.env` (repo root) has all three variables set, and `gcp-key.json` exists:
```
GOOGLE_APPLICATION_CREDENTIALS=./gcp-key.json
GOOGLE_CLOUD_PROJECT=<your-gcp-project>
GOOGLE_CLOUD_LOCATION=<vertex-region, e.g. us-central1>
```
`ffmpeg` must be on `PATH` (`brew install ffmpeg` if missing) — `chunk_video()` shells out to it.

Every command below is run with `uv run python ...` so it uses the project's `.venv` — no manual activation needed. `run.py` and `run_ablation.py` both call `load_dotenv()` themselves.

### 1. Full pipeline for one professor (Phase 7 CLI)

The main entry point. Give it 1–3 lecture videos, each lecture's human COPUS coding (Excel), and an optional survey CSV:

```bash
uv run python run.py \
  --professor "Dr. Smith" \
  --lectures lecture_A.mp4 lecture_B.mp4 lecture_C.mp4 \
  --lecture-ids lecture_A lecture_B lecture_C \
  --sofia-coding sofia_A.xlsx sofia_B.xlsx sofia_C.xlsx \
  --kaw-coding kaw_A.xlsx kaw_B.xlsx kaw_C.xlsx \
  --survey survey_results.csv \
  --output-dir output/dr_smith \
  --arm multimodal
```

- `--arm all` runs the full 4-arm ablation (multimodal/vision_only/audio_only/transcript_only) instead of just one.
- `--kaw-coding` is optional — omit it (or don't pass a path for a given lecture) to skip the AI-vs-Kaw and Sofia-vs-Kaw comparisons.
- `--max-chunks N` caps windows per lecture, useful for a quick smoke run before committing to a full-length video.
- Re-running is idempotent: chunking is skipped if chunks already exist under `output-dir/<lecture_id>/chunks`.
- Outputs land under `output/dr_smith/`: per-lecture `results_*.csv`, `dashboard.html`, `kappa_*.csv`, plus professor-level `combined_kappa.csv`, `professor_dashboard.html`, and `survey_analysis.csv`.

### 2. Ablation study across all 4 modality arms (Phase 6)

For validating the classifier itself against ground-truth human coding (not the professor deliverable flow):

```bash
uv run python run_ablation.py \
  --lectures lecture_001.mp4 lecture_002.mp4 \
  --lecture-ids lecture_001 lecture_002 \
  --output-dir output/ablation_study
```

Expects human ground truth at `human_coding_<lecture_id>.csv` in the repo root (override with `--human-dir`). Produces `comparison_table.csv` and a comparison dashboard per lecture, plus a combined kappa table.

### 3. Streamlit UI (Phase 8) — wraps run.py

```bash
uv run streamlit run app.py
```

Opens in the browser. Fill in professor/course/semester, upload lecture video(s) + Excel coding sheets + optional survey CSV, pick the arm, and run. Once finished, the results page has three tabs (κ scores, timelines, survey) and two downloads: **Faculty Feedback Report (PDF)** (Phase 9, now wired up) and a raw-results ZIP.

### 4. Generate a Faculty Feedback Report PDF standalone (Phase 9)

If you already have a `run.py`-style output directory and just want the PDF without going through the UI:

```bash
uv run python -m report.pdf_report \
  --output-dir output/dr_smith \
  --professor "Dr. Smith" \
  --course "CS 101" \
  --semester "Fall 2026"
```

Writes `output/dr_smith/faculty_report.pdf` by default (override with `--pdf`). Narratives use Gemini when credentials are available and fall back to a deterministic data-driven summary otherwise — the PDF always builds.

### 5. Tests

```bash
uv run python test_chunk.py
uv run python test_classify.py
uv run python test_pipeline.py
uv run python test_validate.py
uv run python test_pdf_report.py
```

Each is a standalone script (not pytest) — run directly with `uv run python`.
