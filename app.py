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

from run import (
    ARM_CONFIG,
    process_lecture,
    build_combined_kappa,
    build_professor_dashboard,
    analyze_survey,
)

WINDOW_SECONDS = 120
KAPPA_COLS = ["ai_vs_sofia_kappa", "ai_vs_kaw_kappa", "sofia_vs_kaw_kappa"]


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
        lectures = cfg["lectures"]  # list of (path, id, sofia, kaw)
        n = len(lectures)
        total_steps = n + 1  # per-lecture work + professor-level rollup

        lecture_infos = []
        for i, (path, lid, sofia, kaw) in enumerate(lectures):
            if self.cancel_event.is_set():
                self.cancelled = True
                self.log_queue.put(f"\n[CANCELLED] stopped before {lid}\n")
                return
            self._set(i / total_steps, f"Processing {lid} ({i + 1}/{n})…")
            info = process_lecture(
                lecture_path=path,
                lecture_id=lid,
                sofia_xlsx=sofia,
                kaw_xlsx=kaw,
                lecture_dir=os.path.join(cfg["output_dir"], lid),
                arms_to_run=cfg["arms_to_run"],
                primary_arm=cfg["primary_arm"],
                max_chunks=cfg["max_chunks"],
                window_seconds=WINDOW_SECONDS,
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
    return df.style.applymap(color, subset=cols).format(na_rep="N/A", subset=cols)


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


def _try_generate_pdf(output_dir, meta):
    """Phase 9 hook. Returns a PDF path if the report module exists, else None."""
    try:
        from report.faculty_report import generate_report  # built in Phase 9
    except Exception:
        return None
    try:
        pdf_path = os.path.join(output_dir, "faculty_feedback_report.pdf")
        generate_report(output_dir, pdf_path, meta)
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
    for k in ("runner", "temp_dir", "output_dir"):
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
    cap = st.number_input("Max windows per lecture (0 = all)", min_value=0, value=0, step=1,
                          help="Cap 2-min windows per lecture for a quick test run.")

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
    lecture_files = [
        st.file_uploader(f"Lecture {i + 1}", type=["mp4"], key=f"lec_{i}")
        for i in range(3)
    ]
    for f in lecture_files:
        if f:
            st.caption(f"✓ {f.name} — {f.size / 1e6:.1f} MB")

    st.subheader("2 · Manual COPUS Coding — Sofia")
    sofia_files = [
        st.file_uploader(f"Sofia Coding — Lecture {i + 1}", type=["xlsx"], key=f"sofia_{i}")
        for i in range(3)
    ]

    st.subheader("3 · Manual COPUS Coding — Dr. Kaw")
    kaw_files = [
        st.file_uploader(f"Dr. Kaw Coding — Lecture {i + 1}", type=["xlsx"], key=f"kaw_{i}")
        for i in range(3)
    ]

    st.subheader("4 · Student Survey (optional)")
    survey_file = st.file_uploader(
        "Student Engagement Survey (SCCCEI/CUCEI) — (optional)", type=["csv"], key="survey"
    )

    # A lecture is runnable only if it has BOTH a video and Sofia's coding.
    runnable = [
        i for i in range(3)
        if lecture_files[i] is not None and sofia_files[i] is not None
    ]
    can_run = len(runnable) >= 1

    if not can_run:
        st.info("Upload at least **Lecture 1** and its **Sofia coding** to enable the run.")
    else:
        skipped = [
            i + 1 for i in range(3)
            if lecture_files[i] is not None and sofia_files[i] is None
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
            lid = f"lecture_{i + 1}"
            lec_path = _save_upload(lecture_files[i], inputs, f"{lid}.mp4")
            sofia_path = _save_upload(sofia_files[i], inputs, f"{lid}_sofia.xlsx")
            kaw_path = (
                _save_upload(kaw_files[i], inputs, f"{lid}_kaw.xlsx")
                if kaw_files[i] is not None else None
            )
            lectures.append((lec_path, lid, sofia_path, kaw_path))

        survey_path = (
            _save_upload(survey_file, inputs, "survey.csv") if survey_file else None
        )

        arms_to_run = list(ARM_CONFIG) if arm == "all" else [arm]
        primary_arm = "multimodal" if "multimodal" in arms_to_run else arms_to_run[0]

        config = {
            "lectures": lectures,
            "output_dir": output_dir,
            "arms_to_run": arms_to_run,
            "primary_arm": primary_arm,
            "max_chunks": int(cap) or None,
            "survey": survey_path,
        }
        ss.meta = {
            "professor": professor, "course": course, "semester": semester,
            "arm": arm, "primary_arm": primary_arm,
        }
        ss.temp_dir = temp_dir
        ss.output_dir = output_dir
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
                st.caption("🟩 κ ≥ 0.7  ·  🟨 0.5–0.7  ·  🟥 < 0.5  ·  ⬜ N/A (no label variation)")
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
                        ("AI vs Dr. Kaw", "kappa_ai_vs_kaw.csv"),
                        ("Sofia vs Dr. Kaw", "kappa_sofia_vs_kaw.csv"),
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
            pdf_path = _try_generate_pdf(output_dir, meta)
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
