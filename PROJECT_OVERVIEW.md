# Thesis Pipeline — Project Overview for Claude Code

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
│   └── classifier.py       # get_gemini_client() + PROMPT_COPUS_UNIFIED (one prompt,
│                            #   all 12 codes, shared by all 4 arms) + the 4 arm entry points
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
│   └── convert_copus_sheet.py  # Excel template → long-format CSV (sparse-window aware)
├── utils/
│   ├── __init__.py
│   └── sparse_windows.py    # compute_sparse_windows() — which chunks to code, from duration
├── output/                  # (gitignored) all pipeline outputs — includes raw A/V chunks
├── run.py                   # Phase 7 — CLI entry point, one full professor bundle
├── run_ablation.py          # Phase 6 — batch runner across all 4 modality arms
├── app.py                   # Phase 8 — Streamlit UI wrapping run.py
├── test_chunk.py
├── test_classify.py
├── test_pipeline.py         # end-to-end: chunk → classify → aggregate (+ ARM_CONFIG)
├── test_validate.py
├── test_pdf_report.py       # Phase 9 smoke test
├── test_sparse_windows.py   # sparse-window acceptance tests (no API, no cost)
├── test_kappa_na.py         # kappa N/A + MEAN acceptance tests (no API, no cost)
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
- **Model**: `gemini-2.5-flash` for dev, `gemini-2.5-pro` for final validation — selected
  per run with `--model` on `run.py` / `run_ablation.py` (default Flash), and recorded in a
  `model` column on every results row so a result can always be traced to the model that
  produced it.

## Inputs You Need to Supply

Everything the pipeline needs, per lecture. Only the first two are things you author; the
rest is derived or already set up.

| Input | Per | Format | Notes |
|---|---|---|---|
| Lecture video | lecture | `.mp4`, `.mov`, `.mkv`, `.m4v`, `.avi`, `.webm` | Any length or size. For multi-GB files paste the **path** into the app rather than uploading. |
| Your COPUS coding | lecture | filled `.xlsx` | **Generated for you** by `--make-template`; you just fill it in. |
| Student survey | professor | `.csv` | Optional. |
| Window indices | lecture | `--windows 0,1,...` | **Derived** — from the sparse calculator, not authored by hand. |
| Lecture ID | lecture | `--lecture-ids` / app field | Names the output folder, the `lecture_id` column, and the coding sheet. **Must be unique per lecture** — reusing one overwrites the earlier results. |
| Model | run | `--model` | Defaults to Flash; pass `gemini-2.5-pro` for final validation. |
| Credentials | once | `.env` + `gcp-key.json` | Already configured; unchanged by this work. |

### The one new requirement: the Excel sheet must match the sampled windows

This is the only real change to your manual workflow, and getting it wrong makes kappa
compare mismatched windows.

> **Using the Streamlit app instead?** It does steps 1 and 2 for you — upload the lecture,
> download the sheet it generates, code it, upload it back, press Run. See
> *3 · Streamlit UI* below. The rest of this section is the CLI equivalent.

**Per lecture, two commands:**

1. **Get the windows and the coding sheet in one step** — point it at the video and it
   probes the duration itself:

   ```bash
   uv run python -m utils.sparse_windows --video lecture_001.mp4 --make-template sofia_001.xlsx
   ```

   `--duration` also works if you'd rather type the running time: `48.33` (minutes),
   `75:30` (MM:SS), or `1:15:30` (HH:MM:SS).

   It prints the scheme, each block's minute range, the comma-separated indices, and writes
   `sofia_001.xlsx` containing **exactly the sampled windows** — one row each, already
   labelled with that window's real minute range.

2. **Code it top to bottom, then run** with the indices it printed:

   ```bash
   uv run python run.py \
     --professor "Dr. Smith" \
     --lectures lecture_001.mp4 --lecture-ids lecture_001 \
     --sofia-coding sofia_001.xlsx \
     --output-dir output/dr_smith \
     --windows 0,1,2,3,4,5,6,7,10,11,12,13,14,15,16,17,20,21,22,23,24,25,26,27,30,31,32,33,34,35,36,37
   ```

### Why you can't reuse the old 30-row sheet

The generated sheet is not the old template with gaps left blank. Sheet rows stay
contiguous while chunk indices jump, so the two drift apart:

| Sheet row | Column A | Chunk index |
|---|---|---|
| 11 | `14-16` | 7 |
| 12 | `20-22` | **10** |
| 19 | `34-36` | **17** |

Row 11 is chunk 7, but row 12 is chunk **10** — chunks 8 and 9 are in a gap and are never
coded or classified. The generated sheet carries the chunk index and any block gap in
reference columns to the right of the codes (the converter ignores them), so you can always
see which window a row belongs to.

**Never leave a row blank to mean "not coded."** kappa cannot distinguish that from "I
observed nothing here," and it will score every AI code in that window as a false positive.

`convert_copus_sheet.py` cross-checks every row against column A and prints a `[warn]` on
any disagreement. **If you see those warnings, stop** — the sheet is coded against the wrong
windows and the kappa would be meaningless.

### Two things to decide before coding the real 9

- **Trailing partial windows.** A lecture rarely divides evenly, so the last chunk is short
  (19.9s on lecture_001). Leaving it in cost **0.35 kappa on Lec** — not because the AI was
  wrong, but because that window was never coded, so kappa read it as "the human saw
  nothing" and every AI code became a false positive. Either code it or drop it from
  `--windows`; don't leave the row blank. The generated sheet flags partial windows in its
  note column, and `run_ablation.py` warns about empty human windows.
- **Never leave a row blank to mean "not coded."** kappa cannot tell that apart from "I
  observed nothing here."

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

### 0b. Pick the windows to code (sparse sampling)

Lectures run 60 min–2 hrs, but manual coding covers only fixed blocks. The AI has to
classify the **same** chunk indices or kappa compares mismatched windows. Compute the
schedule from the lecture's duration:

```bash
uv run python -m utils.sparse_windows --duration 120 --window-size 2
```

It prints the scheme, each block's minute range (for eyeball verification), and the
comma-separated chunk indices to paste into `--windows`.

| Duration | Scheme | Sampled |
|---|---|---|
| >= 75 min | 4 blocks: first + 2 middle at equal gaps + last | 32 chunks (64 min) |
| 48–75 min | 3 blocks: first + centered middle + last | 24 chunks (48 min) |
| < 48 min | full lecture, with a warning | all chunks |

A block is 8 chunks x 2 min = 16 min.

**Predicting the chunk count.** A lecture always produces

```
N = ceil(duration_seconds / window_seconds)
```

chunks, indexed `0 .. N-1`. The final chunk may be **partial** — a 48.33-min lecture at
2-min windows gives 25 chunks, the last covering only 2880–2899.9s (19.9s). Partial
windows are kept, not truncated, so a 20-second window can end up coded as if it were a
full 2-minute window; decide deliberately whether to include it in kappa.

**The Excel sheet must be coded on the sampled windows.** `run.py` passes the same index
list to `convert_copus_sheet.convert()` that it gave the classifier, so sheet row 4+i
becomes `window_index windows[i]` rather than `i`. The converter cross-checks each row
against the sheet's own minute label in column A and warns on any disagreement — if you
see those warnings, the sheet is coded on the wrong windows and kappa would be meaningless.

### 1. Full pipeline for one professor (Phase 7 CLI)

The main entry point. Give it 1–3 lecture videos, each lecture's human COPUS coding (Excel), and an optional survey CSV:

```bash
uv run python run.py \
  --professor "Dr. Smith" \
  --lectures lecture_A.mp4 lecture_B.mp4 lecture_C.mp4 \
  --lecture-ids lecture_A lecture_B lecture_C \
  --sofia-coding sofia_A.xlsx sofia_B.xlsx sofia_C.xlsx \
  --survey survey_results.csv \
  --output-dir output/dr_smith \
  --arm multimodal
```

- `--arm all` runs the full 4-arm ablation (multimodal/vision_only/audio_only/transcript_only) instead of just one.
- `--windows 0,1,2,3,...` processes only those chunk indices (see 0b above). Applies to every lecture in the run, and is passed to the human-coding converter too so AI and human windows line up by construction. Mutually exclusive with `--max-chunks`.
- `--model` selects the Gemini model: `gemini-2.5-flash` (default) or `gemini-2.5-pro`. The model is recorded per row in a `model` column of every results CSV, so any result is traceable to what produced it. Transcription inside the `transcript_only` arm stays pinned to Flash (it is ASR, not classification) and is recorded separately as `transcribe_model`.
- `--max-chunks N` caps windows per lecture, useful for a quick smoke run before committing to a full-length video.
- Re-running is idempotent: chunking is skipped if the chunks for *this window set* already exist under `output-dir/<lecture_id>/chunks`. A chunk that fails classification is retried once and then reported by index — it is never silently dropped.
- Outputs land under `output/dr_smith/`: per-lecture `results_*.csv`, `dashboard.html`, `kappa_*.csv`, plus professor-level `combined_kappa.csv`, `professor_dashboard.html`, and `survey_analysis.csv`.

### 2. Ablation study across all 4 modality arms (Phase 6)

For validating the classifier itself against ground-truth human coding (not the professor deliverable flow):

```bash
uv run python run_ablation.py \
  --lectures lecture_001.mp4 lecture_002.mp4 \
  --lecture-ids lecture_001 lecture_002 \
  --output-dir output/ablation_study \
  --windows 0,1,2,3,4,5,6,7 \
  --model gemini-2.5-flash
```

Expects human ground truth at `human_coding_<lecture_id>.csv` in the repo root (override with `--human-dir`). Produces `comparison_table.csv` and a comparison dashboard per lecture, plus a combined kappa table.

All four arms are held identical except for the modality of the input: the **same prompt**
(`PROMPT_COPUS_UNIFIED`, all 12 instructor codes), the **same chunk indices** (`--windows`),
and the **same model** (`--model`). That is what makes the per-code x per-arm kappa columns
comparable. Each arm differs only in what it is handed — video+audio, silent video, audio,
or a scrubbed transcript.

### Reading a kappa table

- **`N/A`** means kappa is *undefined*, not low: one of the raters never used that code, so
  the constant rater forces `po == pe` and Cohen's kappa collapses to exactly 0.0. Reporting
  it as 0.00 would read as "the raters strongly disagreed" when it actually means "nobody
  rated this."
- The **`MEAN`** row averages only the numeric kappas — `N/A` codes are excluded rather than
  counted as zero.
- A `[warn] ... a rater marked all N windows` line means that code's kappa is depressed by
  prevalence (e.g. Lec at 96%), not by disagreement. The value is still reported.

### 3. Streamlit UI (Phase 8) — wraps run.py

```bash
uv run streamlit run app.py
```

**This is the recommended way to run a lecture — it needs no terminal beyond the line
above.**

Each lecture takes either a **path on this machine** (recommended — nothing is copied, no
size limit) or a browser upload (capped at 10GB, and slow past a couple of GB since the file
is pushed through the browser and buffered server-side). Since the app runs locally, the
path option is almost always the right one for real recordings.
 Upload a lecture and the app probes its duration, picks the sampling scheme, shows
the blocks to watch, and builds the coding sheet for you to download. Code the sheet, upload
it, press Run.

Per lecture it shows duration, scheme, and window count, an expandable list of the blocks
and their minute ranges, a pre-filled (editable) **Windows** box, and a **⬇️ Coding sheet**
download. Each lecture gets its **own** window set computed from its own duration — unlike
`run.py --windows`, which applies one list to every lecture in the call.

The sidebar carries the classification arm and the **Gemini model**.

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

### 5. Generate a coding sheet for one lecture

```bash
uv run python -m utils.sparse_windows --video lecture_001.mp4 --make-template sofia_001.xlsx
```

Writes a `.xlsx` holding exactly the sampled windows, with each window's minute range in
column A and its chunk index in a reference column. See **Inputs You Need to Supply**.

### 6. Tests

These need no credentials and make no API calls — run them after any change:

```bash
uv run python test_sparse_windows.py
uv run python test_kappa_na.py
```

These need existing pipeline output and/or Gemini credentials:

```bash
uv run python test_chunk.py
uv run python test_classify.py
uv run python test_pipeline.py
uv run python test_validate.py
uv run python test_pdf_report.py
```

Each is a standalone script (not pytest) — run directly with `uv run python`.

## Changelog — 2026-08-24 engine changes

Five changes from the Aug 18 hand-off, plus fixes found while verifying them.

**1. Sparse-window sampling.** New `utils/sparse_windows.py` computes which chunks to code
from a lecture's duration (4-block >= 75 min, 3-block 48-75, full lecture below that), with
a CLI that prints indices and minute ranges. `run.py` and `run_ablation.py` take
`--windows`; the chunker cuts only those windows, and `convert_copus_sheet.py` maps sheet
rows onto the same indices so AI and human windows align by construction. Its hardcoded
30-row limit is gone.

**2. `--model` flag.** `gemini-2.5-flash` (default) or `gemini-2.5-pro` on `run.py` and
`run_ablation.py`, threaded to every arm and recorded in a `model` column on every results
row. Transcription stays pinned to Flash and is logged separately as `transcribe_model`.

**3. Kappa N/A + MEAN.** A code that either rater never used now reports `N/A` instead of
`0.00` — with one rater constant, Cohen's kappa collapses to exactly 0.0, which read as
"strong disagreement" when it meant "nobody rated this". A `MEAN` row was added to every
kappa table, averaging numeric values only.

**4. Window count mismatch — root cause.** The "30 human vs 23 AI" gap was three separate
bugs: the converter's hardcoded 30 rows (rows 24-29 were blank padding past the end of a
48.33-min lecture, never ground truth); `MAX_CHUNKS = 24` capping the 25 chunks produced;
and one chunk failing into a bare `except: continue`. Failures are now retried once and
reported by index. Chunk count is `N = ceil(duration / window_size)`, and the chunker's
"expected chunks" print was off by one on exact multiples.

**5. Ablation.** All four arms now share one prompt (`PROMPT_COPUS_UNIFIED`, all 12 codes),
the same `--windows`, and the same `--model`. The modality is the independent variable and
lives purely in the input each arm receives. **This is a methods change** — it supersedes
FOLLOWUPS #1, which had concluded the per-arm prompt restrictions should stay.

### Also fixed (found while verifying)

- **Swapped kappa count columns.** `run.py` passed the AI CSV as `csv_a`, so
  `human_positive_windows` / `ai_positive_windows` were inverted in every
  `kappa_ai_vs_*.csv` it wrote. Kappa is symmetric so the kappa values were unaffected, but
  any "AI over-fires code X by Nx" claim read off those columns is backwards.
- **No request timeout.** A stalled Vertex call hung a run for 27 minutes at 0% CPU with no
  log line; retry logic catches exceptions, not hangs. Now 300s, via `GEMINI_TIMEOUT_SECONDS`.
- **`--kaw-coding` was `required=True`** despite the pipeline fully supporting its absence.
- **All-N/A codes vanished from the PDF** instead of rendering as N/A, and the inter-rater
  narrative would have averaged the new MEAN row into itself.
- **`output/` was not gitignored** — 283MB and 84 raw A/V chunks after one 4-arm run.

### Removed 2026-08-26 — second-coder comparisons

`AI vs Dr. Kaw` and `Sofia vs Dr. Kaw` (inter-rater reliability) were removed from the
pipeline: the `--kaw-coding` flag, the `kappa_ai_vs_kaw.csv` / `kappa_sofia_vs_kaw.csv`
outputs, the two `combined_kappa.csv` columns, and the PDF's inter-rater paragraph.
`compute_kappa()` itself is unchanged and still compares any two coding CSVs, so a second
coder can be reinstated from git history if one is ever added.

### Fixed 2026-08-26 — lectures were all labelled `lecture_1`

The Streamlit app derived each lecture's id from its **upload slot** (`lecture_1`,
`lecture_2`, `lecture_3`), discarding the filename. Running one lecture at a time meant
every run produced `lecture_1`: the app writes to a fresh temp dir so nothing clobbered on
disk, but every export — results CSV, output folder, and coding sheet (`sofia_lecture_1.xlsx`)
— carried the same name, so saving two lectures to one place silently overwrote the first.

The app now derives the id from the video filename (editable per lecture), uses it for the
output folder, the `lecture_id` column, and the sheet filename, and refuses to run on
duplicate or empty ids.

Note the CLI was never affected: `run.py` takes explicit `--lecture-ids` and has always
written per-lecture results to `<output-dir>/<lecture_id>/`. `combined_kappa.csv` sits at the
professor level by design — it is the pooled table across that professor's lectures.
