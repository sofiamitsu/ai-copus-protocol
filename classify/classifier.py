import os
import json
from google import genai
from google.genai import types

from scrub.scrubber import scrub_transcript


def get_gemini_client():
    """
    Creates and returns an authenticated Gemini client.
    Reads credentials from environment variables set in .env
    """
    client = genai.Client(
        vertexai=True,
        project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        location=os.environ.get("GOOGLE_CLOUD_LOCATION"),
    )
    return client


def _call_gemini_and_parse(client, contents, chunk_index, output_dir=None,
                            window_start=None, window_end=None, model="gemini-2.5-flash"):
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

    print(f"Chunk {chunk_index}: {result.get('codes_present')}")

    # Save to JSON file if output_dir provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"chunk_{chunk_index:03d}_result.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved → {out_path}")

    return result


PROMPT_MULTIMODAL = """You are a trained COPUS observer (Classroom Observation Protocol for \
Undergraduate STEM, Smith et al. 2013). You are coding a 2-minute video window of a \
university STEM lecture for INSTRUCTOR behaviors only.

# Your prior expectation
Most 2-minute windows of a standard lecture contain ONLY "Lec", or "Lec" + "RtW". \
Codes like MG, D/V, AnQ, and Adm are EVENTS — they occur in a minority of windows \
and require specific, unambiguous visual or audio evidence. An empty-looking window \
coded as just ["Lec"] is a normal, correct answer. Finding many codes in one window \
is rare and suspicious.

# Codes and their exact definitions

**Lec — Lecturing.** Instructor is presenting content, deriving results, or \
explaining a problem solution to the class.

**RtW — Real-time writing.** Instructor is VISIBLY writing, in real time, on a \
board, document camera, or tablet. The writing must be actively happening on screen.
- Do NOT mark for: pointing at pre-written slides, gesturing at projected content, \
holding a marker without writing, or advancing slides.

**PQ — Posing a question.** Instructor asks the class a non-clicker, non-rhetorical \
question and expects an answer.
- Do NOT mark for: rhetorical questions ("Right?", "Does that make sense?", "Okay?"), \
questions the instructor immediately answers themselves, or transitional filler.

**AnQ — Answering a student question.** A STUDENT's question must be audible or \
clearly indicated (raised hand acknowledged, instructor repeats the question), and \
the instructor listens/responds while the class attends.
- Do NOT mark unless there is direct evidence a student actually asked something in \
THIS window. Instructor answering their own posed question is Lec, not AnQ.

**FUp — Follow-up/feedback.** Instructor gives feedback to the whole class on a \
question or activity the students just did (reviewing answers, discussing what \
groups found).
- Do NOT mark for ordinary lecturing that references earlier material.

**MG — Moving and guiding.** Requires BOTH: (1) students are actively working on an \
assigned task (group work, worksheet, problem), AND (2) the instructor is moving \
through the student area guiding that work.
- Do NOT mark for: pacing at the front, walking to the board, stepping toward the \
audience while lecturing, or any movement while students are only listening.

**D/V — Demo/video.** Instructor is running a physical demonstration, experiment, \
simulation, video, or animation.
- Do NOT mark for: static slides, figures, diagrams, equations, or photos on slides. \
A slide is not a demo.

**Adm — Administration.** Assigning homework, discussing exams/logistics, returning \
tests, announcements about the course.

**O — Other.** Instructor behavior that clearly fits none of the above (explain in \
reasoning).

# Decision procedure
1. Watch/listen to the full window first.
2. For each candidate code, identify the SPECIFIC moment (timestamp or observable \
action) that justifies it. No specific moment = do not mark the code.
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
  "reasoning": "For each code marked: the specific observable moment justifying it. \
If only Lec: state what the instructor was doing.",
  "codes_present": ["..."]
}
"""


PROMPT_VISION_ONLY = """You are a trained COPUS observer (Classroom Observation Protocol for \
Undergraduate STEM, Smith et al. 2013). You are coding a 2-minute window of a university \
STEM lecture for INSTRUCTOR behaviors only.

# Critical constraint
You can ONLY see silent video with no audio track. Do NOT infer verbal behaviors (what the \
instructor is saying, whether a question was asked, whether a student spoke). Only code what \
is visually observable from posture, motion, and on-screen content.

# Codes you may consider (visually observable only)

**Lec — Lecturing.** Instructor stands at the podium/front, gestures at slides or the board, \
addresses the class. You cannot confirm content, only posture/positioning consistent with \
presenting.

**RtW — Real-time writing.** Instructor is VISIBLY writing, in real time, on a board, document \
camera, or tablet — active hand motion producing new marks.
- Do NOT mark for: pointing at pre-written slides, gesturing, holding a marker without \
writing, or advancing slides.

**MG — Moving and guiding.** Requires BOTH: (1) students are visibly working at their seats \
(heads down, writing, working in groups), AND (2) the instructor is moving through the \
student area.
- Do NOT mark for: pacing at the front, walking to the board, or any movement while students \
are just watching/listening.

**D/V — Demo/video.** A physical demonstration, experiment, or a video/simulation playing on \
screen is visibly happening.
- Do NOT mark for: static slides, figures, diagrams, equations, or photos.

**1o1 — One-on-one.** Instructor is visibly leaning into or stationed at a single student or \
small group, not addressing the whole class.

**W — Waiting.** Instructor is visibly standing idle, not gesturing, not writing, not moving \
toward anyone — disengaged from any observable activity.

# Decision procedure
1. Watch the full silent window first.
2. For each candidate code, identify the SPECIFIC visual moment that justifies it. No specific \
moment = do not mark the code.
3. Never guess at codes that require hearing something (PQ, AnQ, FUp, CQ, Adm) — you cannot \
mark these from video alone.
4. It is better to MISS a marginal code than to invent one.

# Output format
Return ONLY this JSON, with reasoning FIRST:
{
  "chunk_index": <number>,
  "window_start": <number>,
  "window_end": <number>,
  "reasoning": "For each code marked: the specific visual moment justifying it. If only Lec: \
state what the instructor was visibly doing.",
  "codes_present": ["..."]
}
"""


PROMPT_AUDIO_ONLY = """You are a trained COPUS observer (Classroom Observation Protocol for \
Undergraduate STEM, Smith et al. 2013). You are coding a 2-minute window of a university \
STEM lecture for INSTRUCTOR behaviors only.

# Critical constraint
You can ONLY hear audio with no video. Do NOT infer visual behaviors (writing, moving, \
demos, gestures). Only code what is audibly detectable from speech content and tone.

# Codes you may consider (audibly detectable only)

**Lec — Lecturing.** Instructor is speaking continuously, presenting or explaining content.

**PQ — Posing a question.** Instructor asks the class a non-clicker, non-rhetorical question \
(rising intonation, a pause afterward inviting a response) and expects an answer.
- Do NOT mark for: rhetorical questions ("Right?", "Does that make sense?", "Okay?"), \
questions the instructor immediately answers themselves, or transitional filler.

**AnQ — Answering a student question.** A student's voice is audible asking something, and the \
instructor responds while the class listens.
- Do NOT mark unless a student's voice is actually heard in THIS window. Instructor answering \
their own posed question is Lec, not AnQ.

**FUp — Follow-up/feedback.** Instructor gives spoken feedback to the class on a question or \
activity just completed (reviewing answers aloud, discussing what groups found).
- Do NOT mark for ordinary lecturing that references earlier material.

**CQ — Clicker question.** Instructor verbally references a clicker/poll question ("pull up \
your clickers", "vote now").

**Adm — Administration.** Spoken announcements about homework, exams, logistics, returning \
tests.

# Decision procedure
1. Listen to the full window first.
2. For each candidate code, identify the SPECIFIC audible moment that justifies it. No \
specific moment = do not mark the code.
3. Never guess at codes that require seeing something (RtW, MG, D/V, 1o1, W) — you cannot \
mark these from audio alone.
4. It is better to MISS a marginal code than to invent one.

# Output format
Return ONLY this JSON, with reasoning FIRST:
{
  "chunk_index": <number>,
  "window_start": <number>,
  "window_end": <number>,
  "reasoning": "For each code marked: the specific audible moment justifying it. If only Lec: \
state what the instructor was saying.",
  "codes_present": ["..."]
}
"""


PROMPT_TRANSCRIBE = """Transcribe this audio clip verbatim. Return only the transcript text, \
nothing else."""


PROMPT_TRANSCRIPT_CLASSIFY = """You are a trained COPUS observer (Classroom Observation \
Protocol for Undergraduate STEM, Smith et al. 2013). You are coding a 2-minute window of a \
university STEM lecture for INSTRUCTOR behaviors only.

# Critical constraint
You only have a text transcript. No audio tone, no video. Only code what the text content \
clearly indicates.

# Codes you may consider (text-legible only)

**Lec — Lecturing.** Transcript contains declarative content statements — explaining, \
deriving, describing.

**PQ — Posing a question.** Transcript contains a non-clicker, non-rhetorical question \
directed at students, clearly expecting an answer.
- Do NOT mark for: rhetorical questions ("Right?", "Does that make sense?", "Okay?"), \
questions the instructor immediately answers themselves, or transitional filler.

**AnQ — Answering a student question.** Transcript clearly indicates a student asked \
something and the instructor is responding to it.
- Do NOT mark unless the text clearly shows a student's question in THIS window. Instructor \
answering their own posed question is Lec, not AnQ.

**FUp — Follow-up/feedback.** Transcript shows the instructor giving feedback on a question or \
activity students just completed.
- Do NOT mark for ordinary lecturing that references earlier material.

**CQ — Clicker question.** Transcript references a clicker/poll question.

**Adm — Administration.** Transcript content is administrative — homework, exams, logistics, \
announcements.

# Decision procedure
1. Read the full transcript window first.
2. For each candidate code, identify the SPECIFIC line/phrase that justifies it. No specific \
line = do not mark the code.
3. Never guess at codes that require seeing or hearing tone (RtW, MG, D/V, 1o1, W) — you \
cannot mark these from text alone.
4. It is better to MISS a marginal code than to invent one.

# Output format
Return ONLY this JSON, with reasoning FIRST:
{
  "chunk_index": <number>,
  "window_start": <number>,
  "window_end": <number>,
  "reasoning": "For each code marked: the specific transcript phrase justifying it. If only \
Lec: state what the instructor was saying.",
  "codes_present": ["..."]
}
"""


def classify_chunk_multimodal(video_path, chunk_index, output_dir=None,
                               window_start=None, window_end=None):
    client = get_gemini_client()

    with open(video_path, "rb") as f:
        video_bytes = f.read()

    contents = [
        types.Part.from_bytes(data=video_bytes, mime_type="video/mp4"),
        types.Part.from_text(text=PROMPT_MULTIMODAL),
    ]

    return _call_gemini_and_parse(client, contents, chunk_index, output_dir, window_start, window_end)


def classify_chunk_vision_only(video_path, chunk_index, output_dir=None,
                                window_start=None, window_end=None):
    client = get_gemini_client()

    with open(video_path, "rb") as f:
        video_bytes = f.read()

    contents = [
        types.Part.from_bytes(data=video_bytes, mime_type="video/mp4"),
        types.Part.from_text(text=PROMPT_VISION_ONLY),
    ]

    return _call_gemini_and_parse(client, contents, chunk_index, output_dir, window_start, window_end)


def classify_chunk_audio_only(audio_path, chunk_index, output_dir=None,
                               window_start=None, window_end=None):
    client = get_gemini_client()

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    contents = [
        types.Part.from_bytes(data=audio_bytes, mime_type="audio/mpeg"),
        types.Part.from_text(text=PROMPT_AUDIO_ONLY),
    ]

    return _call_gemini_and_parse(client, contents, chunk_index, output_dir, window_start, window_end)


def classify_chunk_transcript_only(audio_path, chunk_index, output_dir=None,
                                    window_start=None, window_end=None,
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
        model="gemini-2.5-flash",
        contents=transcribe_contents,
    )
    raw_transcript = transcribe_response.text.strip()

    # Step 2: scrub PII locally (Presidio, no network call).
    scrubbed_transcript = scrub_transcript(raw_transcript)
    del raw_transcript

    # Step 3: classify from the scrubbed text only.
    classify_contents = [
        types.Part.from_text(text=PROMPT_TRANSCRIPT_CLASSIFY + "\n\nTranscript:\n" + scrubbed_transcript),
    ]
    result = _call_gemini_and_parse(client, classify_contents, chunk_index, output_dir, window_start, window_end)

    if save_transcripts and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        transcript_path = os.path.join(output_dir, f"chunk_{chunk_index:03d}_transcript_scrubbed.txt")
        with open(transcript_path, "w") as f:
            f.write(scrubbed_transcript)

    return result
