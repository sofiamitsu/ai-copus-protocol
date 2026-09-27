"""
Streamlit UI for the COPUS pipeline.

A local web front end (localhost:8501) wrapping the run.py CLI. It saves
uploaded files to a temp dir, then calls the SAME functions the CLI uses
(`process_lecture`, `build_combined_kappa`, `build_professor_dashboard`
from run.py) — no pipeline logic is re-implemented here.

Run with:  uv run streamlit run app.py

The pipeline runs in a background thread so the UI can stream log output,
show a live progress bar, and offer a Cancel button while it works. Inside that
thread lectures, arms and windows run in parallel, capped by the sidebar's
"Parallel requests" setting.
"""
import io
import math
import os
import queue
import re
import shutil
import tempfile
import threading
import time
import zipfile
import contextlib

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from classify.classifier import DEFAULT_MODEL, VALID_MODELS, set_max_concurrency
from classify.runner import ARM_CONFIG, DEFAULT_WORKERS, Cancelled
from report.faculty_report import generate_report
from validate.validator import SUMMARY_LABELS
from utils.cucei import load_workbook_scores, professor_id_from_filename, stage_data_dir
from utils.professor_ids import parse_professor_folder, slug as professor_slug
from utils.sparse_windows import plan_sparse_windows, probe_duration_minutes, write_template
from run import (
    build_combined_kappa,
    build_professor_dashboard,
    format_elapsed,
    process_lecture,
    run_parallel,
)

WINDOW_SECONDS = 120
KAPPA_COLS = ["ai_vs_sofia_kappa", "ai_vs_sofia_ac1"]


# ---------------------------------------------------------------------------
# Background pipeline runner
# ---------------------------------------------------------------------------
class _QueueWriter:
    """File-like object that funnels captured stdout into a queue for the UI."""

    def __init__(self, q):
        self.q = q

    def write(self, s):
        if s:
            self.q.put(s)

    def flush(self):
        pass


class PipelineRunner:
    """Runs the professor bundle on a daemon thread, exposing progress + logs."""

    def __init__(self, config):
        self.config = config
        self.log_queue = queue.Queue()
        self.cancel_event = threading.Event()
        self.done_event = threading.Event()
        self._lock = threading.Lock()
        self._progress = 0.0
        self._status = "Starting…"
        self._windows_done = 0
        self._windows_total = 1
        self.error = None
        self.cancelled = False
        self.result = None
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def progress(self):
        with self._lock:
            return self._progress, self._status

    def _set(self, p, s):
        with self._lock:
            self._progress = p
            self._status = s

    def _window_done(self):
        """One (lecture, arm, window) resolved -- drives the progress bar."""
        with self._lock:
            self._windows_done += 1
            done, total = self._windows_done, self._windows_total
            # Classification is ~all the run time; keep the last 5% for rollup.
            self._progress = 0.95 * min(done / total, 1.0)
            self._status = f"Classified {done}/{total} windows…"

    def _run(self):
        writer = _QueueWriter(self.log_queue)
        try:
            with contextlib.redirect_stdout(writer):
                self._pipeline()
        except Cancelled:
            self.cancelled = True
            self.log_queue.put("\n[CANCELLED] stopped; partial results are kept "
                               "and reused if you run again with the same inputs\n")
        except Exception as e:  # never crash the UI
            self.error = f"{type(e).__name__}: {e}"
            self.log_queue.put(f"\n[FATAL] {self.error}\n")
        finally:
            self.done_event.set()

    def _pipeline(self):
        cfg = self.config
        lectures = cfg["lectures"]  # list of (path, id, sofia, windows, n_windows)
        workers = cfg.get("workers", DEFAULT_WORKERS)
        started = time.monotonic()
        with self._lock:
            self._windows_total = max(1, len(cfg["arms_to_run"]) *
                                      sum(n for *_, n in lectures))
        self._set(0.0, f"Processing {len(lectures)} lecture(s) in parallel…")
        set_max_concurrency(workers)

        lecture_infos = run_parallel(process_lecture, [
            dict(
                lecture_path=path,
                lecture_id=lid,
                sofia_xlsx=sofia,
                lecture_dir=os.path.join(cfg["output_dir"], lid),
                arms_to_run=cfg["arms_to_run"],
                primary_arm=cfg["primary_arm"],
                # Each lecture carries its own window set — the schedule depends
                # on that lecture's duration, so one global list will not do.
                max_chunks=None if windows else cfg["max_chunks"],
                window_seconds=WINDOW_SECONDS,
                window_indices=windows,
                model=cfg.get("model", DEFAULT_MODEL),
                # Derived from the lecture's "PROFESSOR N (...)" folder when the
                # video is given by path; an upload falls back to a name slug.
                professor_id=cfg.get("professor_id") or None,
                professor_name=cfg.get("professor", ""),
                course_name=cfg.get("course", ""),
                workers=workers,
                cancel_event=self.cancel_event,
                on_window_done=self._window_done,
            )
            for path, lid, sofia, windows, _ in lectures
        ])

        if self.cancel_event.is_set():
            self.cancelled = True
            self.log_queue.put("\n[CANCELLED] stopped before rollup\n")
            return

        self._set(0.96, "Building professor-level outputs…")
        build_combined_kappa(
            lecture_infos, os.path.join(cfg["output_dir"], "combined_kappa.csv")
        )
        build_professor_dashboard(
            lecture_infos,
            os.path.join(cfg["output_dir"], "professor_dashboard.html"),
            WINDOW_SECONDS,
        )

        self._set(1.0, "Done")
        print(f"\nWall time: {format_elapsed(time.monotonic() - started)}")
        self.result = {
            "lecture_infos": lecture_infos,
            "output_dir": cfg["output_dir"],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _slug(text):
    return re.sub(r"[^a-z0-9]+", "_", (text or "professor").lower()).strip("_") or "professor"


# Video containers ffmpeg can chunk. The pipeline itself is format-agnostic --
# chunk_video probes with ffprobe -- so this list only gates the uploader widget.
VIDEO_TYPES = ["mp4", "mov", "m4v", "mkv", "avi", "webm", "mpeg4"]


class _PathFile:
    """Duck-types the bits of Streamlit's UploadedFile the setup screen reads."""

    def __init__(self, path):
        self.name = os.path.basename(path)
        self.size = os.path.getsize(path)


def _persist_upload(uploaded, key):
    """
    Save an uploaded file once per session and return its path.

    ffprobe needs a real file, and Streamlit reruns on every widget interaction —
    rewriting a multi-hundred-MB lecture each time would make the page unusable.
    Keyed on name+size so replacing the file re-saves it.
    """
    if "probe_dir" not in ss:
        ss.probe_dir = tempfile.mkdtemp(prefix="copus_uploads_")
    cache = ss.setdefault("upload_paths", {})
    sig = f"{uploaded.name}:{uploaded.size}"
    if cache.get(key, (None, None))[0] == sig:
        return cache[key][1]
    path = os.path.join(ss.probe_dir, f"{key}_{re.sub(r'[^A-Za-z0-9._-]', '_', uploaded.name)}")
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    cache[key] = (sig, path)
    return path


def _lecture_plan(path):
    """(duration_min, indices, scheme, blocks) for a lecture, cached per path."""
    cache = ss.setdefault("plans", {})
    if path not in cache:
        duration = probe_duration_minutes(path)
        with contextlib.redirect_stdout(io.StringIO()):  # swallow the short-lecture warning
            indices, scheme, blocks = plan_sparse_windows(duration, WINDOW_SECONDS / 60)
        cache[path] = (duration, indices, scheme, blocks)
    return cache[path]


def _template_bytes(indices, duration_min, name):
    """Build a COPUS coding sheet for these windows and return it as bytes."""
    tmp = os.path.join(tempfile.mkdtemp(prefix="copus_tpl_"), name)
    write_template(tmp, indices, WINDOW_SECONDS / 60, duration_min)
    with open(tmp, "rb") as f:
        return f.read()


def _save_upload(uploaded, dest_dir, name):
    """Persist an UploadedFile to disk; returns the saved path."""
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, name)
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    return path


def _style_kappa(df):
    """Color κ cells: green ≥0.7, yellow 0.5–0.7, red <0.5, gray for N/A."""
    def color(v):
        # N/A (no label variation) reads back from CSV as NaN — keep it neutral,
        # not red, so it isn't mistaken for a poor score.
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "background-color: #e0e0e0; color: #555"
        try:
            x = float(v)
        except (TypeError, ValueError):
            return "background-color: #e0e0e0; color: #555"
        if x >= 0.7:
            return "background-color: #b7e4c7; color: #14532d"
        if x >= 0.5:
            return "background-color: #ffe8a3; color: #7a5b00"
        return "background-color: #f5b7b1; color: #7b241c"

    cols = [c for c in KAPPA_COLS if c in df.columns]
    # Only per-code rows are agreement values; the trailing codes_clearing_* rows
    # are counts and must not be colored as if 3 were a kappa of 3.
    code_rows = df.index[~df["code"].isin(SUMMARY_LABELS)]
    return (df.style.map(color, subset=(code_rows, cols))
            .format(na_rep="N/A", subset=cols))


def _zip_output(output_dir):
    """Zip the output folder, excluding raw chunk media (IRB: no AV in exports)."""
    buf = io.BytesIO()
    parent = os.path.dirname(output_dir)
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(output_dir):
            if "chunks" in dirs:
                dirs.remove("chunks")  # skip raw video/audio chunks
            for fn in files:
                fp = os.path.join(root, fn)
                z.write(fp, os.path.relpath(fp, parent))
    buf.seek(0)
    return buf.getvalue()


def _try_generate_pdf(output_dir, meta, data_dir=None):
    """Build the Faculty Feedback Report PDF; returns its path, or None on failure."""
    try:
        pdf_path = os.path.join(output_dir, "faculty_feedback_report.pdf")
        generate_report(output_dir, pdf_path, meta, data_dir=data_dir)
        return pdf_path
    except Exception as e:
        st.warning(f"PDF generation failed: {e}")
        return None


def _lecture_dirs(output_dir):
    """Per-lecture output folders: any subfolder holding a results_*.csv."""
    return sorted(
        d for d in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, d))
        and any(f.startswith("results_") and f.endswith(".csv")
                for f in os.listdir(os.path.join(output_dir, d)))
    )


def _embed_html(path, height):
    if os.path.exists(path):
        with open(path, "r") as f:
            components.html(f.read(), height=height, scrolling=True)
        return True
    return False


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="COPUS Pipeline", page_icon="🎓", layout="wide")

ss = st.session_state
ss.setdefault("stage", "idle")   # idle | running | done | cancelled | error
ss.setdefault("logs", "")
ss.setdefault("runner", None)
ss.setdefault("meta", {})


def reset_run():
    """Tear down any finished run and its temp dir so the user can start over."""
    tmp = ss.get("temp_dir")
    if tmp and os.path.isdir(tmp):
        shutil.rmtree(tmp, ignore_errors=True)
    for k in ("runner", "temp_dir", "output_dir", "data_dir", "pdf_path"):
        ss.pop(k, None)
    ss.stage = "idle"
    ss.logs = ""


# ---- Sidebar: professor info ----
with st.sidebar:
    st.header("Professor Info")
    professor = st.text_input("Professor name", value=ss.meta.get("professor", ""))
    course = st.text_input("Course name", placeholder="EGN 3000 - Engineering Analysis",
                           value=ss.meta.get("course", ""))
    semester = st.text_input("Semester", placeholder="Summer 2026",
                             value=ss.meta.get("semester", ""))
    st.divider()
    arm = st.selectbox("Classification arm", list(ARM_CONFIG) + ["all"], index=0,
                       help="'all' runs the full 4-arm ablation study.")
    model = st.selectbox("Gemini model", VALID_MODELS,
                         index=VALID_MODELS.index(DEFAULT_MODEL),
                         help="Flash for dev, Pro for final validation.")
    cap = st.number_input("Max windows per lecture (0 = all)", min_value=0, value=0, step=1,
                          help="Cap 2-min windows per lecture for a quick test run. "
                               "Ignored for lectures that have a window set.")
    workers = st.number_input(
        "Parallel requests", min_value=1, max_value=32, value=DEFAULT_WORKERS, step=1,
        help="How many Gemini requests run at once across all lectures and arms. "
             "Higher is faster; lower it if the log shows 429 / rate-limit retries. "
             "1 = fully sequential.")

st.title("🎓 COPUS Classroom Analytics Pipeline")
st.caption("Runs locally — no video or data leaves this machine.")


# =========================== RUNNING ===========================
if ss.stage == "running":
    runner = ss.runner
    st.subheader("Running pipeline…")

    # Drain any new log lines.
    chunks = []
    try:
        while True:
            chunks.append(runner.log_queue.get_nowait())
    except queue.Empty:
        pass
    if chunks:
        ss.logs += "".join(chunks)

    p, status = runner.progress()
    st.progress(min(max(p, 0.0), 1.0), text=status)

    if st.button("🛑 Cancel", type="secondary"):
        runner.cancel_event.set()
        st.info("Cancelling — requests already in flight will finish, "
                "nothing new starts…")

    with st.expander("Live log", expanded=True):
        st.code(ss.logs[-6000:] or "…", language="text")

    if runner.done_event.is_set():
        # Flush remaining logs.
        try:
            while True:
                ss.logs += runner.log_queue.get_nowait()
        except queue.Empty:
            pass
        if runner.error:
            ss.stage = "error"
        elif runner.cancelled:
            ss.stage = "cancelled"
        else:
            ss.stage = "done"
            ss.output_dir = runner.result["output_dir"]
        st.rerun()
    else:
        time.sleep(0.6)
        st.rerun()


# =========================== IDLE / SETUP ===========================
elif ss.stage == "idle":
    st.subheader("1 · Lecture Videos")
    st.caption(
        "Add a lecture and the app reads its duration, works out which windows to code, "
        "and builds the coding sheet for you — no terminal needed. Paste a file path "
        "(best for large recordings) **and press Enter**, or upload the file."
    )
    # Two ways in. Uploading pushes the whole file through the browser and buffers
    # it server-side, which is painful past a couple of GB — and pointless, since
    # the app runs locally and the file is already on this disk. So a path is the
    # default for real lecture recordings.
    # One block per lecture: its inputs AND its plan render together, so the
    # coding sheet appears directly under the lecture it belongs to.
    lecture_files, lecture_paths, lecture_windows, lecture_ids = {}, {}, {}, {}
    lecture_durations = {}
    for i in range(3):
        st.markdown(f"### Lecture {i + 1}")
        typed = (st.text_input(
            f"Path to lecture {i + 1} on this machine — press Enter after pasting",
            key=f"lecpath_{i}", placeholder="/Users/you/Lectures/lecture_004.mov",
            help="Recommended for large recordings — nothing is copied or uploaded. "
                 "Streamlit only reads the box once you press Enter (or click away).")
            or "").strip().strip('"').strip("'")
        f = st.file_uploader(
            f"…or upload lecture {i + 1}", type=VIDEO_TYPES, key=f"lec_{i}")
        lecture_files[i] = f
        if typed:
            path = os.path.expanduser(typed)
            if not os.path.isfile(path):
                st.error(f"Lecture {i + 1}: no file at `{path}`")
                continue
            f = _PathFile(path)
        elif f is not None:
            with st.spinner(f"Saving lecture {i + 1}…"):
                path = _persist_upload(f, f"lec_{i}")
        else:
            st.caption("Paste a path above and **press Enter** — or upload a file — "
                       "to get this lecture's coding sheet.")
            st.divider()
            continue
        try:
            with st.spinner(f"Reading lecture {i + 1}…"):
                duration, indices, scheme, blocks = _lecture_plan(path)
        except Exception as e:
            st.error(f"Lecture {i + 1}: could not read this file ({e}). "
                     f"Is it a video ffmpeg can open?")
            st.divider()
            continue
        # Only a readable lecture counts as present -- the run needs its duration.
        lecture_paths[i] = path
        lecture_durations[i] = duration

        st.success(f"**{f.name}** · {f.size / 1e9:.2f} GB")
        c1, c2, c3 = st.columns(3)
        c1.metric("Duration", f"{duration:.1f} min")
        c2.metric("Sampling", scheme)
        c3.metric("Windows to code", f"{len(indices)}")

        with st.expander(f"Blocks to watch — Lecture {i + 1}", expanded=False):
            rows = []
            for label, first_chunk, last_chunk in blocks:
                start = first_chunk * WINDOW_SECONDS / 60
                end = min((last_chunk + 1) * WINDOW_SECONDS / 60, duration)
                rows.append({"Block": label,
                             "Minutes": f"{start:g} – {end:g}",
                             "Chunks": f"{first_chunk}–{last_chunk}"})
            st.table(pd.DataFrame(rows))

        stem = os.path.splitext(f.name)[0]
        suggested = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_") or f"lecture_{i + 1}"
        lecture_ids[i] = st.text_input(
            f"Lecture ID — Lecture {i + 1}", value=suggested, key=f"lid_{i}",
            help="Names this lecture's output folder, its rows in every results CSV, "
                 "and its coding sheet. Must be unique across the run — otherwise "
                 "results from different lectures overwrite each other.").strip()

        default = ",".join(str(x) for x in indices)
        edited = st.text_input(
            f"Windows — Lecture {i + 1}", value=default, key=f"win_{i}",
            help="Computed from the lecture's duration. Edit only if you coded a "
                 "different set. Blank = classify the whole lecture.")
        try:
            lecture_windows[i] = (sorted({int(w) for w in edited.split(",") if w.strip()})
                                  or None)
        except ValueError:
            st.error(f"Lecture {i + 1}: windows must be comma-separated integers.")
            lecture_windows[i] = indices

        st.download_button(
            f"⬇️ Coding sheet — Lecture {i + 1} (.xlsx)",
            data=_template_bytes(lecture_windows[i] or list(range(
                math.ceil(duration / (WINDOW_SECONDS / 60)))), duration,
                f"{lecture_ids[i] or f'lecture_{i + 1}'}_coding.xlsx"),
            file_name=f"{lecture_ids[i] or f'lecture_{i + 1}'}_coding.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"tpl_{i}",
        )
        st.divider()

    st.subheader("2 · Manual COPUS Coding — Sofia")
    st.caption(
        "Download the sheet above, code each row top to bottom, then upload it here. "
        "Every row needs at least one code — a blank row is read as \"nothing happened\", "
        "not \"not coded\"."
    )
    sofia_files = [
        st.file_uploader(f"Sofia Coding — Lecture {i + 1}", type=["xlsx"], key=f"sofia_{i}")
        for i in range(3)
    ]

    st.subheader("3 · CUCEI Workbook (optional)")
    st.caption(
        "The professor's filled CUCEI_Scoring_Tool workbook — one per professor, "
        "covering all the lectures above. Its scores go into the Faculty Feedback Report."
    )
    cucei_file = st.file_uploader("CUCEI workbook", type=["xlsm", "xlsx"], key="cucei")
    cucei_scores, cucei_ok = None, True
    if cucei_file is not None:
        # The ID links the scores to this run's lectures in the report. Any
        # consistent value works; prefer the real professor number when there is one.
        folder_id = next((parse_professor_folder(p)[0] for p in lecture_paths.values()
                          if parse_professor_folder(p)[0]), None)
        cucei_pid = (professor_id_from_filename(cucei_file.name) or folder_id
                     or professor_slug(professor) or "professor")
        try:
            cucei_scores = load_workbook_scores(
                _persist_upload(cucei_file, "cucei"), cucei_pid)
        except Exception as e:
            st.error(f"**{cucei_file.name}**: {e}")
            cucei_ok = False
        else:
            n = pd.to_numeric(cucei_scores["n_respondents"], errors="coerce").max()
            who = f" · {int(n)} respondents" if pd.notna(n) else ""
            st.success(f"**{cucei_file.name}**{who} · scores match the raw answers")
            st.dataframe(cucei_scores.drop(columns="professor_id"), hide_index=True,
                         use_container_width=True)
            if folder_id and folder_id != cucei_pid:
                st.warning(f"This workbook is for `{cucei_pid}`, but the lecture "
                           f"folder says `{folder_id}`. The run will use `{cucei_pid}`.")

    # A lecture is runnable only if it has BOTH a readable video and Sofia's
    # coding. Keyed on lecture_paths, not the uploader — a lecture given by path
    # has no uploaded file object.
    runnable = [
        i for i in range(3)
        if lecture_paths.get(i) and sofia_files[i] is not None
    ]
    can_run = len(runnable) >= 1

    chosen_ids = [lecture_ids.get(i, "") for i in runnable]
    if len(set(chosen_ids)) != len(chosen_ids):
        dupes = sorted({x for x in chosen_ids if chosen_ids.count(x) > 1})
        st.error(f"Lecture IDs must be unique — {dupes} is used more than once. "
                 f"Two lectures sharing an ID write to the same folder and the "
                 f"second overwrites the first.")
        can_run = False
    if any(not x for x in chosen_ids):
        st.error("Every lecture needs a non-empty Lecture ID.")
        can_run = False

    if not cucei_ok:
        st.error("Fix or remove the CUCEI workbook above before running.")
        can_run = False

    if not can_run:
        st.info("Add at least **Lecture 1** (path or upload) and its **Sofia coding** "
                "to enable the run.")
    else:
        skipped = [
            i + 1 for i in range(3)
            if lecture_paths.get(i) and sofia_files[i] is None
        ]
        if skipped:
            st.warning(f"Lecture(s) {skipped} have a video but no Sofia coding — they will be skipped.")

    if st.button("🚀 Run Pipeline", type="primary", disabled=not can_run):
        # Persist uploads to a temp dir, then hand off to the worker thread.
        temp_dir = tempfile.mkdtemp(prefix="copus_")
        inputs = os.path.join(temp_dir, "inputs")
        output_dir = os.path.join(temp_dir, "output", _slug(professor))

        lectures = []
        for i in runnable:
            lid = lecture_ids.get(i) or f"lecture_{i + 1}"
            # Already on disk from the duration probe — don't copy it again.
            lec_path = lecture_paths[i]
            sofia_path = _save_upload(sofia_files[i], inputs, f"{lid}_sofia.xlsx")
            windows = lecture_windows.get(i)
            if windows:
                n_windows = len(windows)
            else:
                n_windows = math.ceil(lecture_durations[i] / (WINDOW_SECONDS / 60))
                if cap:
                    n_windows = min(n_windows, int(cap))
            lectures.append((lec_path, lid, sofia_path, windows, n_windows))

        data_dir = None
        if cucei_scores is not None:
            data_dir = stage_data_dir(cucei_scores, os.path.join(temp_dir, "data"))

        arms_to_run = list(ARM_CONFIG) if arm == "all" else [arm]
        primary_arm = "multimodal" if "multimodal" in arms_to_run else arms_to_run[0]

        config = {
            "lectures": lectures,
            "output_dir": output_dir,
            "arms_to_run": arms_to_run,
            "primary_arm": primary_arm,
            "max_chunks": int(cap) or None,
            "model": model,
            "workers": int(workers),
            "professor": professor,
            # Stamped on every lecture so the report finds the uploaded scores;
            # None keeps the usual folder / name-slug resolution.
            "professor_id": (cucei_scores["professor_id"].iloc[0]
                             if cucei_scores is not None else None),
            "course": course,
        }
        ss.meta = {
            "professor": professor, "course": course, "semester": semester,
            "arm": arm, "primary_arm": primary_arm, "model": model,
        }
        ss.temp_dir = temp_dir
        ss.output_dir = output_dir
        ss.data_dir = data_dir
        ss.logs = ""
        runner = PipelineRunner(config)
        runner.start()
        ss.runner = runner
        ss.stage = "running"
        st.rerun()


# =========================== RESULTS / TERMINAL STATES ===========================
else:
    if ss.stage == "error":
        st.error("Pipeline failed. See the log below.")
        with st.expander("Log", expanded=True):
            st.code(ss.logs[-8000:] or "…", language="text")
    elif ss.stage == "cancelled":
        st.warning("Pipeline cancelled. Partial outputs (if any) are below.")

    output_dir = ss.get("output_dir")
    meta = ss.meta

    if ss.stage in ("done", "cancelled") and output_dir and os.path.isdir(output_dir):
        if ss.stage == "done":
            st.success(f"Pipeline complete for **{meta.get('professor') or 'professor'}**.")

        tab_k, tab_t = st.tabs(["κ Scores", "Behavioral Timelines"])

        # --- Tab 1: kappa ---
        with tab_k:
            combined = os.path.join(output_dir, "combined_kappa.csv")
            if os.path.exists(combined):
                df = pd.read_csv(combined)
                st.markdown("**Combined (pooled across lectures)**")
                st.dataframe(_style_kappa(df), use_container_width=True)
                st.caption("🟩 κ / AC1 ≥ 0.7  ·  🟨 0.5–0.7  ·  🟥 < 0.5  ·  ⬜ N/A (no label variation)")
            else:
                st.info("No combined κ table found.")

            for lid_dir in _lecture_dirs(output_dir):
                fp = os.path.join(output_dir, lid_dir, "kappa_ai_vs_sofia.csv")
                if os.path.exists(fp):
                    with st.expander(f"{lid_dir} — AI vs Sofia κ"):
                        st.dataframe(pd.read_csv(fp), use_container_width=True)

        # --- Tab 2: timelines ---
        with tab_t:
            for lid_dir in _lecture_dirs(output_dir):
                st.markdown(f"**{lid_dir}**")
                if not _embed_html(os.path.join(output_dir, lid_dir, "dashboard.html"), 520):
                    st.info("No timeline for this lecture.")
            st.divider()
            st.markdown("**Professor-level combined profile**")
            _embed_html(os.path.join(output_dir, "professor_dashboard.html"), 520)

        # --- Downloads ---
        st.divider()
        st.subheader("Download")
        c1, c2 = st.columns(2)
        with c1:
            # Built once per run: every widget click reruns this page, and the
            # report calls Gemini for its narratives.
            if "pdf_path" not in ss:
                with st.spinner("Building the Faculty Feedback Report…"):
                    ss.pdf_path = _try_generate_pdf(output_dir, meta, ss.get("data_dir"))
            pdf_path = ss.pdf_path
            if pdf_path and os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "📥 Faculty Feedback Report (PDF)", f.read(),
                        file_name="faculty_feedback_report.pdf", mime="application/pdf",
                    )
            else:
                st.button("📥 Faculty Feedback Report (PDF)", disabled=True,
                          help="The report could not be built — see the warning above.")
        with c2:
            st.download_button(
                "📥 Raw Results (ZIP)", _zip_output(output_dir),
                file_name=f"{_slug(meta.get('professor'))}_results.zip", mime="application/zip",
                help="Excludes raw video/audio chunks.",
            )

    st.divider()
    st.button("↩️ Start over", on_click=reset_run)
