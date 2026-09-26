"""
Phase 8 — Streamlit UI for the COPUS pipeline.

A local web front end (localhost:8501) wrapping the Phase 7 CLI. It saves
uploaded files to a temp dir, then calls the SAME functions the CLI uses
(`process_lecture`, `build_combined_kappa`, `build_professor_dashboard`,
`analyze_survey` from run.py) — no pipeline logic is re-implemented here.

Run with:  uv run streamlit run app.py

The pipeline runs in a background thread so the UI can stream log output,
show a live progress bar, and offer a Cancel button while it works.
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

from classify.classifier import DEFAULT_MODEL, VALID_MODELS
from validate.validator import SUMMARY_LABELS
from utils.cucei import load_workbook_scores, professor_id_from_filename, stage_data_dir
from utils.sparse_windows import plan_sparse_windows, probe_duration_minutes, write_template
from run import (
    ARM_CONFIG,
    process_lecture,
    build_combined_kappa,
    build_professor_dashboard,
    analyze_survey,
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

    def _run(self):
        writer = _QueueWriter(self.log_queue)
        try:
            with contextlib.redirect_stdout(writer):
                self._pipeline()
        except Exception as e:  # never crash the UI
            self.error = f"{type(e).__name__}: {e}"
            self.log_queue.put(f"\n[FATAL] {self.error}\n")
        finally:
            self.done_event.set()

    def _pipeline(self):
        cfg = self.config
        lectures = cfg["lectures"]  # list of (path, id, sofia, windows)
        n = len(lectures)
        total_steps = n + 1  # per-lecture work + professor-level rollup

        lecture_infos = []
        for i, (path, lid, sofia, windows) in enumerate(lectures):
            if self.cancel_event.is_set():
                self.cancelled = True
                self.log_queue.put(f"\n[CANCELLED] stopped before {lid}\n")
                return
            self._set(i / total_steps, f"Processing {lid} ({i + 1}/{n})…")
            info = process_lecture(
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
            )
            lecture_infos.append(info)

        if self.cancel_event.is_set():
            self.cancelled = True
            self.log_queue.put("\n[CANCELLED] stopped before rollup\n")
            return

        self._set(n / total_steps, "Building professor-level outputs…")
        build_combined_kappa(
            lecture_infos, os.path.join(cfg["output_dir"], "combined_kappa.csv")
        )
        build_professor_dashboard(
            lecture_infos,
            os.path.join(cfg["output_dir"], "professor_dashboard.html"),
            WINDOW_SECONDS,
        )
        if cfg["survey"]:
            analyze_survey(
                cfg["survey"], os.path.join(cfg["output_dir"], "survey_analysis.csv")
            )

        self._set(1.0, "Done")
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
    return (df.style.applymap(color, subset=(code_rows, cols))
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
    """Phase 9 hook. Returns a PDF path if the report module exists, else None."""
    try:
        from report.faculty_report import generate_report  # built in Phase 9
    except Exception:
        return None
    try:
        pdf_path = os.path.join(output_dir, "faculty_feedback_report.pdf")
        generate_report(output_dir, pdf_path, meta, data_dir=data_dir)
        return pdf_path
    except Exception as e:
        st.warning(f"PDF generation failed: {e}")
        return None


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
    for k in ("runner", "temp_dir", "output_dir", "data_dir"):
        ss.pop(k, None)
    ss.stage = "idle"
    ss.logs = ""


# ---- Sidebar: professor info ----
with st.sidebar:
    st.header("Professor Info")
    professor = st.text_input("Professor name", value=ss.meta.get("professor", ""))
    professor_id = st.text_input(
        "Professor ID (optional)", value=ss.meta.get("professor_id", ""),
        placeholder="professor_1",
        help="Links this run to the professor's CUCEI scores. Leave blank to take it "
             "from the uploaded CUCEI workbook's name, or from a 'PROFESSOR N' folder "
             "in the lecture path.").strip()
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
        st.info("Cancelling after the current lecture finishes…")

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
        lecture_paths[i] = path
        try:
            with st.spinner(f"Reading lecture {i + 1}…"):
                duration, indices, scheme, blocks = _lecture_plan(path)
        except Exception as e:
            st.error(f"Lecture {i + 1}: could not read this file ({e}). "
                     f"Is it a video ffmpeg can open?")
            st.divider()
            continue

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
        "The filled CUCEI_Scoring_Tool workbook, named with the professor number "
        "(e.g. `CUCEI PROFESSOR 1.xlsm`). Its scores go into the Faculty Feedback "
        "Report. Add other professors' workbooks too to get a study average."
    )
    cucei_files = st.file_uploader(
        "CUCEI workbook(s)", type=["xlsm", "xlsx"], key="cucei",
        accept_multiple_files=True) or []
    cucei_ok, cucei_ids = True, []
    for j, cf in enumerate(cucei_files):
        pid = professor_id_from_filename(cf.name)
        try:
            if pid in cucei_ids:
                raise ValueError(f"a second workbook for {pid}.")
            scores = load_workbook_scores(_persist_upload(cf, f"cucei_{j}"), pid)
        except Exception as e:
            st.error(f"**{cf.name}**: {e}")
            cucei_ok = False
            continue
        cucei_ids.append(pid)
        with st.expander(f"✅ {cf.name} → {pid}", expanded=len(cucei_files) == 1):
            st.dataframe(scores.drop(columns="professor_id"), hide_index=True,
                         use_container_width=True)

    # Which professor's CUCEI scores this report shows: the sidebar ID, else the
    # only uploaded workbook. Otherwise it falls back to the lecture's folder.
    run_professor_id = professor_id or (cucei_ids[0] if len(cucei_ids) == 1 else "")
    if cucei_ids and run_professor_id and run_professor_id not in cucei_ids:
        st.warning(f"Professor ID is `{run_professor_id}`, but the uploaded "
                   f"workbooks are for {cucei_ids} — this report may show no CUCEI "
                   f"scores unless data/ has them.")
    elif len(cucei_ids) > 1 and not professor_id:
        st.warning("Several CUCEI workbooks uploaded — set **Professor ID** in the "
                   "sidebar so the report knows which one is this professor's.")

    st.subheader("4 · Student Survey (optional)")
    survey_file = st.file_uploader(
        "Other student survey — (optional, generic summary only)", type=["csv"], key="survey",
        help="Not for CUCEI — upload the CUCEI workbook in section 3."
    )

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
        st.error("Fix or remove the CUCEI workbook(s) above before running.")
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
            lectures.append((lec_path, lid, sofia_path, lecture_windows.get(i)))

        survey_path = (
            _save_upload(survey_file, inputs, "survey.csv") if survey_file else None
        )

        data_dir = None
        if cucei_files:
            cucei_inputs = os.path.join(inputs, "cucei")
            wb_paths = [_save_upload(cf, cucei_inputs,
                                     re.sub(r"[^A-Za-z0-9._ ()-]", "_", cf.name))
                        for cf in cucei_files]
            data_dir = stage_data_dir(wb_paths, os.path.join(temp_dir, "data"))

        arms_to_run = list(ARM_CONFIG) if arm == "all" else [arm]
        primary_arm = "multimodal" if "multimodal" in arms_to_run else arms_to_run[0]

        config = {
            "lectures": lectures,
            "output_dir": output_dir,
            "arms_to_run": arms_to_run,
            "primary_arm": primary_arm,
            "max_chunks": int(cap) or None,
            "model": model,
            "survey": survey_path,
            "professor": professor,
            "professor_id": run_professor_id,
            "course": course,
        }
        ss.meta = {
            "professor": professor, "professor_id": professor_id,
            "course": course, "semester": semester,
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

        tab_k, tab_t, tab_s = st.tabs(["κ Scores", "Behavioral Timelines", "Survey Analysis"])

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

            for lid_dir in sorted(
                d for d in os.listdir(output_dir)
                if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("lecture_")
            ):
                ld = os.path.join(output_dir, lid_dir)
                with st.expander(f"{lid_dir} — per-comparison κ"):
                    for label, fn in (
                        ("AI vs Sofia", "kappa_ai_vs_sofia.csv"),
                    ):
                        fp = os.path.join(ld, fn)
                        if os.path.exists(fp):
                            st.markdown(f"*{label}*")
                            st.dataframe(pd.read_csv(fp), use_container_width=True)

        # --- Tab 2: timelines ---
        with tab_t:
            for lid_dir in sorted(
                d for d in os.listdir(output_dir)
                if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("lecture_")
            ):
                st.markdown(f"**{lid_dir}**")
                if not _embed_html(os.path.join(output_dir, lid_dir, "dashboard.html"), 520):
                    st.info("No timeline for this lecture.")
            st.divider()
            st.markdown("**Professor-level combined profile**")
            _embed_html(os.path.join(output_dir, "professor_dashboard.html"), 520)

        # --- Tab 3: survey ---
        with tab_s:
            survey_csv = os.path.join(output_dir, "survey_analysis.csv")
            if os.path.exists(survey_csv):
                st.dataframe(pd.read_csv(survey_csv), use_container_width=True)
            else:
                st.info("No survey uploaded.")
            st.caption("AI-generated feedback narrative arrives with Phase 9.")

        # --- Downloads ---
        st.divider()
        st.subheader("Download")
        c1, c2 = st.columns(2)
        with c1:
            pdf_path = _try_generate_pdf(output_dir, meta, ss.get("data_dir"))
            if pdf_path and os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "📥 Faculty Feedback Report (PDF)", f.read(),
                        file_name="faculty_feedback_report.pdf", mime="application/pdf",
                    )
            else:
                st.button("📥 Faculty Feedback Report (PDF)", disabled=True,
                          help="Available once Phase 9 (PDF report) is built.")
        with c2:
            st.download_button(
                "📥 Raw Results (ZIP)", _zip_output(output_dir),
                file_name=f"{_slug(meta.get('professor'))}_results.zip", mime="application/zip",
                help="Excludes raw video/audio chunks.",
            )

    st.divider()
    st.button("↩️ Start over", on_click=reset_run)
