# Follow-ups — Known Issues & Deferred Fixes

Issues found during the Phase 9 review (2026-08-03) by reading every pipeline
module end-to-end. None of these block the code from running; they affect the
*validity* or *completeness* of results, so they should be resolved before
Phase 10 real-data runs produce numbers that go in the thesis.

Ordered by impact on results.

---

## 1. The multimodal arm cannot emit CQ, 1o1, or W

**Severity: high — silently caps κ for 3 of 12 codes.**

**Where:** `PROMPT_MULTIMODAL` in `classify/classifier.py:56` (code definitions
at lines 67–108).

**Problem:** The multimodal prompt defines only **9** codes:

> Lec, RtW, PQ, AnQ, FUp, MG, D/V, Adm, O

The human Excel template (`validate/convert_copus_sheet.py:6`,
`INSTRUCTOR_CODES`) codes all **12**:

> Lec, RtW, FUp, PQ, **CQ**, AnQ, MG, **1o1**, D/V, Adm, **W**, O

So CQ, 1o1, and W are structurally unreachable for the primary arm. If a human
rater marks any of them, the AI can never agree — κ for that code is pinned at
0 or reported "N/A" (no AI-side variation). This is not a model performance
result, it is a prompt omission, and it will look like a model failure in the
validation table.

**Per-arm code coverage as currently written:**

| Arm | Codes the prompt permits | Count |
|---|---|---|
| multimodal | Lec, RtW, PQ, AnQ, FUp, MG, D/V, Adm, O | 9 |
| vision_only | Lec, RtW, MG, D/V, 1o1, W | 6 |
| audio_only | Lec, PQ, AnQ, FUp, CQ, Adm | 6 |
| transcript_only | Lec, PQ, AnQ, FUp, CQ, Adm | 6 |

The *restrictions on the three ablation arms are deliberate and correct* — you
genuinely cannot hear a clicker question in muted video, and the prompts say so
explicitly ("Never guess at codes that require hearing something"). Those should
stay as they are; they are the independent variable of the ablation.

Only the **multimodal** omission appears unintentional: that arm has both video
and audio, so it should be able to reach all 12 codes.

**Fix:** Add CQ, 1o1, and W definitions to `PROMPT_MULTIMODAL`, matching the
style of the existing entries (definition + explicit "Do NOT mark for"
exclusions). Suggested definitions, consistent with the other prompts and the
COPUS protocol:

- **CQ — Clicker question.** Instructor is asking a clicker/polling question
  (verbally referenced or visibly on screen).
- **1o1 — One-on-one.** Instructor is in extended discussion with a single
  student or small group, not addressing the whole class.
- **W — Waiting.** Instructor is idle when they could be engaging — not
  presenting, writing, moving, or responding.

**⚠️ Caveat — invalidates cached results.** `run_arm()` in `test_pipeline.py:45`
skips any chunk whose `chunk_NNN_result.json` already exists, so changing the
prompt will NOT re-classify existing chunks. After editing the prompt you must
delete `output/**/results_multimodal/` for any run whose numbers you intend to
use. Re-running costs Gemini calls.

**Definition of done:**
- [ ] `PROMPT_MULTIMODAL` documents all 12 instructor codes
- [ ] Existing `results_multimodal/` JSONs deleted for runs that will be re-used
- [ ] Re-run and confirm CQ/1o1/W can appear in output when present

---

## 2. Human coding is hard-capped at 30 windows (60 minutes)

**Severity: high for full-length lectures, invisible when it happens.**

**Where:** `validate/convert_copus_sheet.py:19` — `for window_idx in range(30):`

**Problem:** The converter emits exactly 30 rows, always — windows 0–29,
covering 0–3600s. Two consequences:

- **Lectures longer than 60 min:** human codes for windows 30+ are dropped.
  `kappa_by_code()` only compares windows present in *both* raters
  (`validate/validator.py:66-77`), so κ is silently computed on the first 60
  minutes only. `run.py:_warn_window_mismatch` will print a `[warn]`, but the
  run continues and the resulting κ looks complete.
- **Lectures shorter than 60 min:** 30 rows are always written, so trailing rows
  are empty-coded. Harmless for κ (unmatched windows are excluded) but it makes
  `human_*.csv` misleading to read directly.

A 75-minute lecture loses its last ~15 minutes of ground truth. A 90-minute
lecture loses a third.

**Fix:** Derive the window count instead of hardcoding it. Options, cheapest
first:

1. Read until the first fully-empty row past a minimum, or until the sheet's
   `ws.max_row`, rather than a fixed 30.
2. Add an optional `n_windows` parameter to `convert()`, and have `run.py` pass
   the AI-side chunk count (it already computes `n` in `process_lecture`).

Option 2 is more explicit and makes the AI/human window counts agree by
construction. The Excel template itself may also need more than 30 data rows.

**Definition of done:**
- [ ] `convert()` no longer hardcodes 30
- [ ] Verified against a >60 min lecture: all human windows survive to the CSV
- [ ] Confirm the .xlsx template has enough rows for the longest lecture

---

## 3. "Presidio runs locally" is only true for the transcript arm

**Severity: documentation / IRB accuracy — no code bug.**

**Where:** `PROJECT_OVERVIEW.md` Key Constraints; actual scrubbing at
`classify/classifier.py:364` inside `classify_chunk_transcript_only`.

**Problem:** `scrub_transcript()` is called in exactly one place — the
`transcript_only` arm. The other three arms (`multimodal`, `vision_only`,
`audio_only`) send **raw video or audio bytes** straight to Vertex AI. Nothing
is scrubbed, and nothing *can* be: Presidio operates on text, not MP4/MP3.

This is not a defect — it is inherent to sending A/V to a multimodal model — but
the current phrasing ("Presidio runs locally: PII scrubbing never sends
transcript to cloud") reads as though all cloud traffic is de-identified. For an
IRB protocol that distinction matters.

What *is* accurate:
- The transcript arm never writes a raw transcript to disk (`del raw_transcript`,
  `classify/classifier.py:365`) and only ever persists the scrubbed version, and
  only when `save_transcripts=True`.
- Raw video stays local + USF Box; only 2-minute chunks transit to Vertex.

**Fix:** Restate the constraint precisely, e.g.:

> **Presidio (transcript arm only):** In the `transcript_only` arm, audio is
> transcribed, then PII-scrubbed locally with Presidio before any text is sent
> for classification; the raw transcript is never written to disk. The
> `multimodal`, `vision_only`, and `audio_only` arms send A/V chunks directly to
> Vertex AI and are not text-scrubbable.

**Definition of done:**
- [ ] `PROJECT_OVERVIEW.md` constraint reworded
- [ ] Thesis methods section + IRB language matches

---

## 4. Streamlit UI never surfaces the cross-arm comparison table

**Severity: medium — UX gap, results exist on disk.**

**Where:** `app.py` — no reference to `comparison_table` or
`comparison_dashboard` anywhere in the file.

**Problem:** The arm selector at `app.py:257` offers `"all"`, which runs the full
4-arm ablation. `run.py:process_lecture` then writes `comparison_table.csv` and
`comparison_dashboard.html` per lecture. Neither is ever read back by the UI, so
the single most interesting ablation output — **which modality arm agrees best
with human coding** — is written to disk and never shown. The user would have to
find the CSV manually.

The κ tab (`app.py:420-445`) shows combined + per-lecture κ for the three
AI/human comparisons, but only for the *primary* arm.

**Fix:** Add a fourth tab (e.g. "Arm Comparison"), rendered only when
`comparison_table.csv` exists, showing the pooled table plus each lecture's
`comparison_dashboard.html` via the existing `_embed_html()` helper.

**Definition of done:**
- [ ] Comparison tab appears for `--arm all` runs, hidden otherwise
- [ ] Pooled `comparison_table.csv` displayed
- [ ] Per-lecture comparison dashboards embedded

---

## Minor / cosmetic

- **`google-genai` is not a direct dependency.** `classify/classifier.py:3`
  imports `from google import genai`, which currently resolves only as a
  transitive dep of `google-cloud-aiplatform`. It works today, but a future
  resolution could break it. Consider `uv add google-genai` to pin it explicitly.
- **`aggregate_results()` parameter name is misleading.** Its first parameter is
  called `chunks_dir` (`aggregate/aggregator.py:5`) but every caller passes a
  *results* directory. Rename to `results_dir` for clarity.
- **`test_pipeline.py` has module-level hardcoded paths** (`LECTURE_PATH =
  "videoplayback.mp4"`, `MAX_CHUNKS = 24`) that only apply to its `__main__`
  block, but the module is also imported by `run.py` and `run_ablation.py` for
  `ARM_CONFIG`/`run_arm`. Harmless, slightly confusing.
