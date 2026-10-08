# Changelog

## 2026-10-08 — reliability outputs for the thesis draft

All four are additive: per-code κ / AC1 values and the ablation-deltas script are
unchanged.

1. **AC1 via irrCAC.** `gwet_ac1()` calls `irrCAC.raw.CAC(...).gwet()` instead of the
   hand-rolled Gwet (2008) formula (landed in 3f976ea). The N/A guards (no windows;
   neither rater used the code) are unchanged. A new test checks it against the old
   formula on the Lec 23/24 case: both give 0.957.
2. **Precision and recall per code.** `precision` = TP / (TP + FP) and `recall` =
   TP / (TP + FN), 3 decimals, `N/A` on a zero denominator. Added as the last columns of
   `combined_kappa.csv` (`ai_vs_sofia_precision`, `ai_vs_sofia_recall`) and
   `comparison_table.csv` (`<arm>_precision`, `<arm>_recall`), and to the Faculty
   Feedback Report's per-code table. Blank on summary rows.
3. **Aggregate agreement rows.** `overall_raw_agreement_pct`, `pooled_kappa` and
   `prevalence_weighted_mean_kappa` at the bottom of both tables, over the codes the
   human marked. The PDF shows them in a smaller "Overall Agreement" note under the
   per-code table, labelled supplementary to the per-code view.
4. **Paste-ready threshold rows.** The `codes_clearing_kappa_0.7` / `codes_clearing_ac1_0.7`
   cells now read `"3 of 5"` instead of a bare count, so Table 6.1 can be pasted from
   the CSV. `n_codes_human_observed` is still filled in, and the PDF reads both formats.

## 2026-09 — parallel runs and repo cleanup

- **Parallel execution.** Lectures, ablation arms and windows now run concurrently. A
  single cap on in-flight Gemini requests (`--workers`, default 8; the app's *Parallel
  requests* setting) keeps the total inside Vertex quota. `--workers 1` reproduces the old
  sequential behaviour. Results are unchanged: every window is independent and decoding
  is pinned. ffmpeg cuts within a lecture also run 4 at a time.
- **Retry backoff.** Failed windows are retried with exponential backoff and jitter
  (longer on 429 rate limits) instead of immediately.
- **Cancel** in the app now stops within seconds instead of after the current lecture.
- **App fixes.** Per-lecture κ tables and timelines were only listed for lecture IDs
  starting with `lecture_` — lectures named after their video file never showed. The PDF
  report was regenerated (with Gemini calls) on every click of the results page; it is
  now built once per run.
- **Cleanup.** `run_arm`/`ARM_CONFIG` moved from `test_pipeline.py` to
  `classify/runner.py`; tests moved to `tests/`, offline analysis scripts to `scripts/`,
  build-phase specs to `docs/development/`. Removed `main.py`, a duplicate phase spec, and
  three scratch scripts (`test_chunk.py`, `test_classify.py`, `test_validate.py`).

## 2026-08-24 — engine changes

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

## 2026-08-26 — removed — second-coder comparisons

`AI vs Dr. Kaw` and `Sofia vs Dr. Kaw` (inter-rater reliability) were removed from the
pipeline: the `--kaw-coding` flag, the `kappa_ai_vs_kaw.csv` / `kappa_sofia_vs_kaw.csv`
outputs, the two `combined_kappa.csv` columns, and the PDF's inter-rater paragraph.
`compute_kappa()` itself is unchanged and still compares any two coding CSVs, so a second
coder can be reinstated from git history if one is ever added.

## 2026-08-26 — fixed: — lectures were all labelled `lecture_1`

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
