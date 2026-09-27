# Phase 5 — Ablation Arms

## Context

The thesis pipeline already has a working **multimodal arm** in `classify/classifier.py` that sends `chunk_full.mp4` (video + audio) to Gemini and gets COPUS codes back. We need 3 additional arms to compare how each modality contributes to classification accuracy.

## Existing Code to Reference

- `classify/classifier.py` — contains `classify_chunk_multimodal()` and `get_gemini_client()`
- `chunk/chunker.py` — already produces `chunk_full.mp4`, `chunk_muted.mp4`, `chunk_audio.mp3` per window
- `validate/validator.py` — computes Cohen's κ per code
- `.env` — contains `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, `GOOGLE_APPLICATION_CREDENTIALS`

## What to Build

### 1. Vision-Only Arm

**File:** Add `classify_chunk_vision_only()` to `classify/classifier.py`

**Input:** `chunk_muted.mp4` (video with no audio)

**Prompt guidance:** Focus on spatial/movement codes only — what can be determined from video alone without hearing anything:
- `RtW` — writing on board or document camera (visible hand motion, facing board)
- `MG` — moving through student area, circulating
- `D/V` — visible demo, experiment, simulation on screen
- `1o1` — leaning into one student/group, not addressing class
- `W` — instructor standing idle, not engaged
- `Lec` — standing at podium, gesturing at slides (but can't confirm without audio)

**Key instruction in prompt:** "You can ONLY see video with no audio. Do not infer verbal behaviors. Only code what is visually observable."

**Output:** Same JSON schema as multimodal arm:
```json
{
  "chunk_index": <number>,
  "window_start": <seconds>,
  "window_end": <seconds>,
  "codes_present": ["code1", "code2"],
  "reasoning": "brief explanation"
}
```

### 2. Audio-Only Arm

**File:** Add `classify_chunk_audio_only()` to `classify/classifier.py`

**Input:** `chunk_audio.mp3`

**Prompt guidance:** Focus on verbal/auditory codes only:
- `Lec` — continuous instructor monologue
- `PQ` — instructor posing a question (rising intonation, pause after)
- `AnQ` — instructor responding to a student voice
- `FUp` — instructor giving feedback after student response
- `CQ` — instructor referencing clicker/poll
- `Adm` — administrative announcements (homework, deadlines)

**Key instruction in prompt:** "You can ONLY hear audio. Do not infer visual behaviors. Only code what is audibly detectable."

**Output:** Same JSON schema.

### 3. Transcript-Only Arm (2-call arm)

**File:** Add `classify_chunk_transcript_only()` to `classify/classifier.py`

**This arm requires 2 Gemini calls + Presidio scrubbing in between:**

**Call 1:** Send `chunk_audio.mp3` to Gemini with prompt: "Transcribe this audio clip verbatim. Return only the transcript text, nothing else."

**Scrub step:** Run the transcript through `scrub/scrubber.py` (Presidio) to remove PII.

**Call 2:** Send the scrubbed transcript text to Gemini with the classification prompt. Focus on text-only codes:
- `Lec` — declarative content statements
- `PQ` — question directed at students
- `AnQ` — response to a student question
- `FUp` — feedback on prior student response
- `CQ` — reference to clicker/poll
- `Adm` — administrative content

**Key instruction in prompt:** "You only have a text transcript. No audio tone, no video. Only code what the text content clearly indicates."

**Output:** Same JSON schema.

### 4. Presidio Scrubber Module

**File:** `scrub/scrubber.py` (new file, also create `scrub/__init__.py`)

**Dependencies:** `presidio-analyzer`, `presidio-anonymizer`

**Install:** `uv add presidio-analyzer presidio-anonymizer`

**Function:**
```python
def scrub_transcript(text: str) -> str:
    """
    Removes PII (names, emails, phone numbers) from transcript text.
    Returns scrubbed text with PII replaced by [REDACTED].
    """
```

Uses `presidio-analyzer` with the `en_core_web_lg` spaCy model to detect entities, then `presidio-anonymizer` to replace them.

## Integration with Pipeline

Update `test_pipeline.py` (or the future `run.py`) to accept an `--arm` flag:
- `multimodal` (default, existing)
- `vision_only`
- `audio_only`
- `transcript_only`
- `all` — runs all 4 arms sequentially

Each arm writes results to a separate subfolder:
```
output/pipeline_test/results_multimodal/
output/pipeline_test/results_vision_only/
output/pipeline_test/results_audio_only/
output/pipeline_test/results_transcript_only/
```

The aggregator and validator should accept an `arm` parameter to compare any arm against human ground truth.

## Model

Use `gemini-2.5-flash` for all development. Switch to `gemini-2.5-pro` for final validation runs only.

## Testing

After building all 3 arms:
1. Run each arm on lecture_001 (first 3 chunks for speed)
2. Verify each produces valid JSON output
3. Run validator comparing each arm to `human_coding_lecture_001.csv`
4. Produce a comparison table: code × arm × κ

## Definition of Done

- [ ] `classify_chunk_vision_only()` works and produces COPUS codes
- [ ] `classify_chunk_audio_only()` works and produces COPUS codes
- [ ] `classify_chunk_transcript_only()` works (transcribe → scrub → classify)
- [ ] `scrub/scrubber.py` strips PII from transcript text
- [ ] All 4 arms produce results in the same JSON schema
- [ ] Validator can compare any arm against human ground truth
- [ ] 3-chunk test run for each arm completes without errors
