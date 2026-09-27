# Phase 8 — Streamlit UI

## Context

The pipeline works end-to-end via CLI (`run.py` from Phase 7). This phase wraps it in a local web interface so Sofia can process 3 professors × 3 lectures each without running terminal commands. Runs locally at `localhost:8501` — no cloud deployment, no data leaves the machine.

## Install

```bash
uv add streamlit
```

## File

**`app.py`** in the root of the THESIS folder.

Run with: `uv run streamlit run app.py`

## UI Layout

### Sidebar: Professor Info
- Text input: Professor name
- Text input: Course name (e.g. "EGN 3000 - Engineering Analysis")
- Text input: Semester (e.g. "Summer 2026")

### Main Area: File Uploads

**Section 1: Lecture Videos**
- 3 file uploaders for `.mp4` files
- Labels: "Lecture 1", "Lecture 2", "Lecture 3"
- Show file name + size after upload

**Section 2: Manual COPUS Coding (Sofia)**
- 3 file uploaders for `.xlsx` files
- Labels: "Sofia Coding — Lecture 1", etc.
- Must match lecture order

**Section 3: Manual COPUS Coding (Dr. Kaw)**
- 3 file uploaders for `.xlsx` files
- Labels: "Dr. Kaw Coding — Lecture 1", etc.

**Section 4: Student Survey**
- 1 file uploader for `.csv`
- Label: "Student Engagement Survey (SCCCEI/CUCEI)"
- Optional — show "(optional)" in label

### Run Button

Large button: "🚀 Run Pipeline"

Disabled until at least:
- 1 lecture video uploaded
- 1 Sofia coding Excel uploaded

### Progress Display

After clicking Run:
- Progress bar showing overall completion
- Status text per step: "Chunking Lecture 1...", "Classifying chunk 3/25...", "Computing κ..."
- Real-time log output in an expander

### Results Display

After pipeline completes:

**Tab 1: κ Scores**
- Table showing per-code κ for each comparison (AI-Sofia, AI-Kaw, Sofia-Kaw)
- Color coding: green ≥ 0.7, yellow 0.5-0.7, red < 0.5

**Tab 2: Behavioral Timelines**
- Embedded plotly charts (one per lecture)
- Use `st.plotly_chart()` to render inline

**Tab 3: Survey Analysis**
- Display survey summary stats (if survey uploaded)
- AI-generated feedback narrative (if implemented in Phase 9)

### Download Button

After results are ready:
- "📥 Download Faculty Feedback Report (PDF)" — downloads the PDF from Phase 9
- "📥 Download Raw Results (ZIP)" — zips the entire output folder

## Architecture

The Streamlit app should:
1. Save uploaded files to a temp directory
2. Call the same functions from `run.py` / existing modules
3. Capture print output and display in the UI
4. Clean up temp files after download

**Important:** Do NOT re-implement pipeline logic in the Streamlit app. Import and call existing functions from `chunk/`, `classify/`, `aggregate/`, `report/`, `validate/`.

## File Size Handling

Lecture `.mp4` files can be 500MB–2GB. Streamlit's default upload limit is 200MB.

Add to `.streamlit/config.toml`:
```toml
[server]
maxUploadSize = 3000
```

## Error Handling

- If a Gemini API call fails, show error in the UI but don't crash
- If an uploaded file is the wrong format, show a clear error message
- Add a "Cancel" button that stops the pipeline mid-run

## Definition of Done

- [ ] Streamlit app runs at localhost:8501
- [ ] All file uploaders work for the full professor bundle
- [ ] Pipeline runs end-to-end from the UI
- [ ] Progress bar shows real-time status
- [ ] κ table and behavioral timelines render inline
- [ ] Download buttons work for PDF and ZIP
- [ ] Tested with 1 full professor bundle
