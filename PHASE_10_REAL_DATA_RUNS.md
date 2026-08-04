# Phase 10 — Run on Real Professors

## Context

This is the final execution phase — running the validated pipeline on real USF classroom data collected in July 2026. Not a coding phase; this is operational.

## Prerequisites

- All prior phases complete
- Prompt iteration finalized — κ ≥ 0.7 on major codes
- At least 1 professor's full data bundle ready:
  - 3 lecture `.mp4` recordings (from USF classroom, stored on USF Box)
  - 3 COPUS Excel codings from Sofia
  - 3 COPUS Excel codings from Dr. Kaw
  - 1 student survey CSV (SCCCEI/CUCEI from Qualtrics)

## Data Handling Protocol

1. Download lecture `.mp4` from USF Box to local machine
2. Process through pipeline (Streamlit UI or CLI)
3. After pipeline completes, delete local `.mp4` copies
4. Upload output PDFs and CSVs to USF Box for archival
5. Raw video NEVER touches AWS, GCP storage, or any non-USF platform
6. All Gemini API calls use ZDR (already configured)

## Execution Plan

| Professor | Data Ready | Pipeline Run | Report Delivered |
|---|---|---|---|
| Professor 1 | TBD | TBD | TBD |
| Professor 2 | TBD | TBD | TBD |
| Professor 3 | TBD | TBD | TBD |

## Delivery

- Faculty Feedback Reports delivered via secure USF email or USF Box shared folder
- Reports are for personal pedagogical development only (per IRB protocol)
- Individual student survey responses are never shared with faculty
- Faculty receive only aggregated engagement scores mapped against behavioral timeline

## Post-Run

- Archive all de-identified results to USF Box (5-year retention per IRB)
- Delete raw video from local machine
- Record any pipeline errors or anomalies for thesis methods chapter

## Definition of Done

- [ ] All 3 professors processed through the pipeline
- [ ] 3 Faculty Feedback Reports generated as PDFs
- [ ] Reports delivered to professors
- [ ] Raw video deleted from local machine
- [ ] De-identified results archived on USF Box
