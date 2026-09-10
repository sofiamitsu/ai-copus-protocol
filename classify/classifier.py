import os
import json
from google import genai
from google.genai import types

from scrub.scrubber import scrub_transcript

DEFAULT_MODEL = "gemini-2.5-flash"
VALID_MODELS = ["gemini-2.5-flash", "gemini-2.5-pro"]

# Transcription is ASR, not classification -- it is not the variable under study,
# so it stays pinned to Flash regardless of --model. Both are recorded in the
# per-chunk JSON so the choice is visible in the output rather than buried.
TRANSCRIBE_MODEL = "gemini-2.5-flash"

# Vertex calls have no timeout by default, so a stalled request blocks forever:
# a hung chunk silently froze a 4-arm run for 27 minutes with 0% CPU and no log
# line. A hang is worse than an error -- an error gets retried and reported.
# Override with GEMINI_TIMEOUT_SECONDS if a long chunk legitimately needs more.
REQUEST_TIMEOUT_SECONDS = int(os.environ.get("GEMINI_TIMEOUT_SECONDS", "300"))


def get_gemini_client():
    """
    Creates and returns an authenticated Gemini client.
    Reads credentials from environment variables set in .env
    """
    client = genai.Client(
        vertexai=True,
        project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        location=os.environ.get("GOOGLE_CLOUD_LOCATION"),
        # HttpOptions.timeout is in MILLISECONDS.
        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_SECONDS * 1000),
    )
    return client


def _call_gemini_and_parse(client, contents, chunk_index, output_dir=None,
                            window_start=None, window_end=None,
                            model=DEFAULT_MODEL, extra_metadata=None):
    response = client.models.generate_content(model=model, contents=contents)

    # Clean the response (strip markdown code fences if present)
    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    result = json.loads(raw)
    if isinstance(result, list):
        result = result[0]

    # Override with actual values since Gemini might get these wrong
    result["chunk_index"] = chunk_index
    result["window_start"] = window_start
    result["window_end"] = window_end
    # Stamp the model so every result is traceable to what produced it.
    result["model"] = model
    if extra_metadata:
        result.update(extra_metadata)

    print(f"Chunk {chunk_index}: {result.get('codes_present')}")

    # Save to JSON file if output_dir provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"chunk_{chunk_index:03d}_result.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved -> {out_path}")

    return result


# ---------------------------------------------------------------------------
# One prompt, all four arms.
#
# Every arm is given the identical instructions and the identical 12-code
# universe (matching INSTRUCTOR_CODES in validate/convert_copus_sheet.py, so the
# AI and the human sheet can finally agree on every code). The modality is the
# independent variable and lives purely in the INPUT: full video+audio, silent
# video, audio only, or a scrubbed text transcript.
#
# This replaces the four per-arm prompts, which exposed four different code
# subsets and made the per-code x per-arm kappa table structurally holey.
# ---------------------------------------------------------------------------
PROMPT_COPUS_UNIFIED = """You are a trained COPUS observer (Classroom Observation Protocol for \
Undergraduate STEM, Smith et al. 2013). You are coding a 2-minute window of a \
university STEM lecture for INSTRUCTOR behaviors only.

# What you have been given
You may receive any ONE of: video with audio, silent video with no audio track, \
audio with no video, or a text transcript. Code ONLY what the material you were \
actually given evidences. If a code would require something you cannot observe in \
this material -- you cannot hear a clicker question in silent video, you cannot see \
writing in an audio clip -- then simply do not mark it. That absence is expected \
and correct; never guess to fill a gap.

# Your prior expectation
Most 2-minute windows of a standard lecture contain ONLY "Lec", or "Lec" + "RtW". \
Codes like MG, D/V, AnQ, CQ, 1o1, and Adm are EVENTS -- they occur in a minority of \
windows and require specific, unambiguous evidence. An empty-looking window \
coded as just ["Lec"] is a normal, correct answer. Finding many codes in one window \
is rare and suspicious.

# Codes and their exact definitions

**Lec -- Lecturing.** Instructor is presenting content, deriving results, or \
explaining a problem solution to the class.
- In windows dominated by Q&A dialogue (PQ + student responses + FUp), mark Lec \
ONLY if there is substantial content presentation ALONGSIDE the dialogue, not \
merely when the instructor speaks to facilitate, acknowledge student answers, \
or transition between questions.

**RtW -- Real-time writing.** Instructor is VISIBLY writing, in real time, on a \
board, document camera, or tablet. The writing must be actively happening.
- Do NOT mark for: pointing at pre-written slides, gesturing at projected content, \
holding a marker without writing, or advancing slides.

**FUp -- Follow-up/feedback.** Instructor gives feedback to the whole class on a \
question or activity the students just did (reviewing answers, discussing what \
groups found, reformulating or evaluating a student's response for the class \
before moving on).
- Do NOT mark for ordinary lecturing that references earlier material.

**PQ -- Posing a question.** Instructor asks the class a non-clicker question \
about the CONTENT and creates space for a student response. Real PQ examples: \
"Why does this loop terminate?", "What happens if we change x to y?", \
"Which case are we in here?".
- Do NOT mark verbal check-ins: "Any questions?", "Questions?", "Does anyone \
have a question?", "You agree?", "Right?", "Okay?", "Make sense?"
- Do NOT mark when the instructor is framing a problem or hypothetical they are \
about to solve or explore themselves: "The question is what happens when...", \
"One question is where does it land...", "At what angle should we fire..."
- Do NOT mark rhetorical questions, questions the instructor immediately answers \
themselves, or thinking-aloud during derivation.
- Do NOT mark transitional filler.

**CQ -- Clicker question.** Instructor is asking a clicker/polling question, either \
referenced verbally ("pull up your clickers", "vote now") or visibly posed on screen \
as a poll.
- Do NOT mark for an ordinary question with no clicker/poll involved -- that is PQ.

**AnQ -- Answering a student question.** A student asks an UNPROMPTED, spontaneous \
question (raised hand, calls out, interrupts) and the instructor listens and \
responds while the class attends. The student must have INITIATED with a question, \
not merely responded to something the instructor asked.
- Do NOT mark AnQ when a student is responding to a PQ (Posed Question) the \
instructor asked. Even if the instructor engages with the student's answer at \
length, that is Socratic dialogue -- the instructor's response is FUp (if \
reformulating/evaluating for the class) or Lec (if continuing exposition), \
NOT AnQ.
- Do NOT mark unless there is direct evidence a student ASKED something in THIS \
window (not merely responded to a question).
- Instructor answering their own posed question is Lec, not AnQ.

**MG -- Moving and guiding.** Requires BOTH: (1) students are actively working on an \
assigned task (group work, worksheet, problem), AND (2) the instructor is moving \
through the student area guiding that work.
- Do NOT mark for: pacing at the front, walking to the board, stepping toward the \
audience while lecturing, or any movement while students are only listening.

**1o1 -- One-on-one.** Instructor is in extended discussion with a single student or \
one small group, not addressing the whole class.
- Do NOT mark for a brief exchange that the whole class is attending to -- that is AnQ.

**D/V -- Demo/video.** Instructor is running a physical demonstration, experiment, \
simulation, video, or animation.
- Do NOT mark for: static slides, figures, diagrams, equations, or photos on slides. \
A slide is not a demo.

**Adm -- Administration.** Assigning homework, discussing exams/logistics, returning \
tests, announcements about the course, including recommendations to purchase \
materials, sign up for sections, or use specific course resources.

**W -- Waiting.** Instructor is idle for an EXTENDED duration (30+ seconds) when \
they could be engaging -- not presenting, writing, moving, or responding to anyone.
- Do NOT mark for brief pauses mid-sentence, while advancing a slide, or the \
natural 5-25 second gap after posing a PQ while students think. Only mark W when \
the instructor is genuinely disengaged for a sustained period with no apparent \
purpose.

**O -- Other.** Instructor behavior that clearly fits none of the above (explain in \
reasoning).

# Decision procedure
1. Take in the full window first.
2. For each candidate code, identify the SPECIFIC moment (timestamp, observable \
action, audible line, or transcript phrase) that justifies it. No specific moment = \
do not mark the code.
3. Apply the "Do NOT mark" exclusions strictly. When two codes seem to overlap, \
prefer the more specific evidence-based one; default to Lec when in doubt.
4. It is better to MISS a marginal code than to invent one. False positives are \
worse than false negatives in this task.

# Output format
Return ONLY this JSON, with reasoning FIRST:
{
  "chunk_index": <number>,
  "window_start": <number>,
  "window_end": <number>,
  "reasoning": "For each code marked: the specific moment justifying it. \
If only Lec: state what the instructor was doing.",
  "codes_present": ["..."]
}
"""


PROMPT_TRANSCRIBE = """Transcribe this audio clip verbatim. Return only the transcript text, \
nothing else."""


def classify_chunk_multimodal(video_path, chunk_index, output_dir=None,
                               window_start=None, window_end=None,
                               model=DEFAULT_MODEL):
    client = get_gemini_client()

    with open(video_path, "rb") as f:
        video_bytes = f.read()

    contents = [
        types.Part.from_bytes(data=video_bytes, mime_type="video/mp4"),
        types.Part.from_text(text=PROMPT_COPUS_UNIFIED),
    ]

    return _call_gemini_and_parse(client, contents, chunk_index, output_dir,
                                  window_start, window_end, model=model)


def classify_chunk_vision_only(video_path, chunk_index, output_dir=None,
                                window_start=None, window_end=None,
                                model=DEFAULT_MODEL):
    client = get_gemini_client()

    with open(video_path, "rb") as f:
        video_bytes = f.read()

    contents = [
        types.Part.from_bytes(data=video_bytes, mime_type="video/mp4"),
        types.Part.from_text(text=PROMPT_COPUS_UNIFIED),
    ]

    return _call_gemini_and_parse(client, contents, chunk_index, output_dir,
                                  window_start, window_end, model=model)


def classify_chunk_audio_only(audio_path, chunk_index, output_dir=None,
                               window_start=None, window_end=None,
                               model=DEFAULT_MODEL):
    client = get_gemini_client()

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    contents = [
        types.Part.from_bytes(data=audio_bytes, mime_type="audio/mpeg"),
        types.Part.from_text(text=PROMPT_COPUS_UNIFIED),
    ]

    return _call_gemini_and_parse(client, contents, chunk_index, output_dir,
                                  window_start, window_end, model=model)


def classify_chunk_transcript_only(audio_path, chunk_index, output_dir=None,
                                    window_start=None, window_end=None,
                                    model=DEFAULT_MODEL,
                                    save_transcripts=False):
    client = get_gemini_client()

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    # Step 1: transcribe verbatim. Raw transcript lives only in memory, never on disk.
    transcribe_contents = [
        types.Part.from_bytes(data=audio_bytes, mime_type="audio/mpeg"),
        types.Part.from_text(text=PROMPT_TRANSCRIBE),
    ]
    transcribe_response = client.models.generate_content(
        model=TRANSCRIBE_MODEL,
        contents=transcribe_contents,
    )
    raw_transcript = transcribe_response.text.strip()

    # Step 2: scrub PII locally (Presidio, no network call).
    scrubbed_transcript = scrub_transcript(raw_transcript)
    del raw_transcript

    # Step 3: classify from the scrubbed text only.
    classify_contents = [
        types.Part.from_text(text=PROMPT_COPUS_UNIFIED + "\n\nTranscript:\n" + scrubbed_transcript),
    ]
    result = _call_gemini_and_parse(
        client, classify_contents, chunk_index, output_dir,
        window_start, window_end, model=model,
        extra_metadata={"transcribe_model": TRANSCRIBE_MODEL},
    )

    if save_transcripts and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        transcript_path = os.path.join(output_dir, f"chunk_{chunk_index:03d}_transcript_scrubbed.txt")
        with open(transcript_path, "w") as f:
            f.write(scrubbed_transcript)

    return result
