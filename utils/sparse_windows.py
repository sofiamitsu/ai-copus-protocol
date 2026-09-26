"""
Sparse-window calculator for COPUS manual-coding blocks.

Sofia hand-codes fixed blocks per lecture rather than the whole lecture, so the
AI has to classify the SAME chunk indices or kappa compares mismatched windows.
Lectures run from ~60 min to ~2 hrs, so the block schedule is computed from the
lecture's duration instead of being hardcoded.

A block is 8 chunks x 2 min = 16 min (approximates the "15 min block" target).

  duration >= 75 min   4 blocks: first + 2 middle at equal gaps + last  (32 chunks)
  48 <= duration < 75  3 blocks: first + centered middle + last         (24 chunks)
  duration < 48 min    fallback: every chunk, with a printed warning

Partial chunks at the end of a lecture are INCLUDED rather than truncated, so
the chunk count is always ceil(duration / window_size) and the final window may
be shorter than window_size.

Unsampled gaps are BY DESIGN, not dropped data. Chunk indices keep their real
position in the lecture, so every chunk between blocks is simply absent from
the human sheet, the results CSVs, and kappa. For a 48-49.99 min lecture (25
chunks) the blocks are 0-7, 8-15, 17-24 and the only gap is chunk 16 -- the
"missing window 16" seen in the lecture_001/Mehran runs. Other durations leave
other gaps (50 min: 8; 60 min: 8-10 and 19-21). See NOTES.md, "Window 16".

Rounding: this module never uses the built-in round(). Python rounds half to
even -- round(12.5) == 12 but round(13.5) == 14 -- which for a 50-minute lecture
picks a different middle block than a human reading "round(duration / 4)" would
expect. _round_half_up() makes the schedule deterministic and reproducible
across all 9 lectures.

CLI:
    uv run python -m utils.sparse_windows --video lecture.mp4
    uv run python -m utils.sparse_windows --duration 1:15:30
    uv run python -m utils.sparse_windows --duration 75 --make-template sheet.xlsx
"""
import argparse
import math
import os

# A block is 8 chunks. At the default 2-min window that is a 16-min block.
BLOCK_CHUNKS = 8

# Scheme thresholds, in absolute minutes (from the sampling design, not derived
# from window size).
FOUR_BLOCK_MIN_MINUTES = 75.0
THREE_BLOCK_MIN_MINUTES = 48.0


def _round_half_up(value):
    """Round half away from zero -- see the module docstring on banker's rounding."""
    return int(math.floor(value + 0.5))


def _block_at(start_chunk, total_chunks):
    """A BLOCK_CHUNKS-long run of indices starting at start_chunk, clamped in range."""
    start = max(0, min(start_chunk, total_chunks - BLOCK_CHUNKS))
    return list(range(start, start + BLOCK_CHUNKS))


def plan_sparse_windows(duration_min, window_size_min=2.0):
    """
    Like compute_sparse_windows, but also returns the scheme name and the block
    layout so a caller (the CLI) can show a human-verifiable breakdown.

    Returns (indices, scheme, blocks) where blocks is a list of
    (label, first_chunk, last_chunk).
    """
    if duration_min <= 0:
        raise ValueError(f"duration_min must be positive, got {duration_min}")
    if window_size_min <= 0:
        raise ValueError(f"window_size_min must be positive, got {window_size_min}")

    # Include the trailing partial window rather than truncating it.
    total_chunks = math.ceil(duration_min / window_size_min)
    block_min = BLOCK_CHUNKS * window_size_min

    # Too short to sample sparsely -- code the whole thing.
    if duration_min < THREE_BLOCK_MIN_MINUTES or total_chunks <= BLOCK_CHUNKS:
        print(f"[warn] lecture is {duration_min:.2f} min "
              f"(< {THREE_BLOCK_MIN_MINUTES:.0f} min): too short to sample in blocks, "
              f"using all {total_chunks} chunks (full lecture).")
        indices = list(range(total_chunks))
        return indices, "full", [("full lecture", 0, total_chunks - 1)]

    first = _block_at(0, total_chunks)
    last = _block_at(total_chunks - BLOCK_CHUNKS, total_chunks)

    if duration_min >= FOUR_BLOCK_MIN_MINUTES:
        # Two middle blocks separated from each other and from the end blocks by
        # three equal gaps across the span between the first and last blocks.
        gap_min = (duration_min - 4 * block_min) / 3
        mid1 = _block_at(_round_half_up((block_min + gap_min) / window_size_min),
                         total_chunks)
        mid2 = _block_at(_round_half_up((2 * block_min + 2 * gap_min) / window_size_min),
                         total_chunks)
        scheme = "4-block"
        blocks = [("first", first[0], first[-1]),
                  ("middle 1", mid1[0], mid1[-1]),
                  ("middle 2", mid2[0], mid2[-1]),
                  ("last", last[0], last[-1])]
        chosen = first + mid1 + mid2 + last
    else:
        # One block centered on the midpoint of the lecture.
        center = _round_half_up(duration_min / (2 * window_size_min))
        mid = _block_at(center - BLOCK_CHUNKS // 2, total_chunks)
        scheme = "3-block"
        blocks = [("first", first[0], first[-1]),
                  ("middle", mid[0], mid[-1]),
                  ("last", last[0], last[-1])]
        chosen = first + mid + last

    # Blocks can abut on short lectures; they must never overlap or repeat.
    indices = sorted(set(chosen))
    assert all(b < a for a, b in zip(indices[1:], indices)), \
        f"sparse windows not strictly increasing: {indices}"
    assert indices[-1] < total_chunks, \
        f"chunk index {indices[-1]} past end of lecture ({total_chunks} chunks)"
    return indices, scheme, blocks


def compute_sparse_windows(duration_min, window_size_min=2.0):
    """
    Chunk indices to classify for a lecture of duration_min minutes.

    >>> len(compute_sparse_windows(120, 2))
    32
    >>> len(compute_sparse_windows(60, 2))
    24
    """
    indices, _scheme, _blocks = plan_sparse_windows(duration_min, window_size_min)
    return indices


def parse_duration_minutes(text):
    """
    A lecture's running time, in minutes.

    Accepts what a recording actually reports:
      "48.33"    -> 48.33 minutes
      "75:30"    -> 75 min 30 sec  = 75.5
      "1:15:30"  -> 1 hr 15 min 30 sec = 75.5
    """
    text = str(text).strip()
    if ":" not in text:
        return float(text)
    parts = [float(p) for p in text.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        hours = 0.0
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        raise ValueError(f"can't read '{text}' as a duration "
                         f"(use MIN, MM:SS, or HH:MM:SS)")
    return hours * 60 + minutes + seconds / 60


def probe_duration_minutes(video_path):
    """Read a video's duration with ffprobe (same probe the chunker uses)."""
    import ffmpeg
    probe = ffmpeg.probe(video_path)
    return float(probe["format"]["duration"]) / 60


def write_template(xlsx_path, indices, window_size_min, duration_min):
    """
    Write a COPUS coding sheet containing exactly the sampled windows.

    One row per sampled chunk, in order, with that chunk's real minute range in
    column A -- which is what convert_copus_sheet.convert() cross-checks. Sheet
    rows stay contiguous while chunk indices jump, so a generic 30-row template
    cannot be used for sparse sampling.
    """
    import openpyxl
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

    # Imported here so this module stays importable without the validate package.
    from validate.convert_copus_sheet import (
        DATA_START_ROW, INSTRUCTOR_CODES, INSTRUCTOR_START_COL, LABEL_COL,
    )

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "COPUS Coding Sheet"

    ws.cell(row=1, column=1, value="Date:")
    ws.cell(row=1, column=7, value="Class:")
    ws.cell(row=1, column=12, value="Instructor:")
    ws.cell(row=2, column=2, value="2. Instructor Doing").font = Font(bold=True)

    head = Font(bold=True, color="FFFFFF")
    fill = PatternFill("solid", fgColor="003366")
    thin = Side(style="thin", color="CCCCCC")
    box = Border(left=thin, right=thin, top=thin, bottom=thin)

    ws.cell(row=3, column=LABEL_COL, value="min").font = head
    ws.cell(row=3, column=LABEL_COL).fill = fill
    for j, code in enumerate(INSTRUCTOR_CODES):
        c = ws.cell(row=3, column=INSTRUCTOR_START_COL + j, value=code)
        c.font, c.fill, c.alignment = head, fill, Alignment(horizontal="center")
    note_col = INSTRUCTOR_START_COL + len(INSTRUCTOR_CODES)
    ws.cell(row=3, column=note_col, value="chunk").font = Font(bold=True, italic=True)
    ws.cell(row=3, column=note_col + 1, value="note").font = Font(bold=True, italic=True)

    for i, chunk in enumerate(indices):
        row = DATA_START_ROW + i
        start_min = int(chunk * window_size_min)
        end_min = int((chunk + 1) * window_size_min)
        ws.cell(row=row, column=LABEL_COL, value=f"{start_min}-{end_min}").border = box
        for j in range(len(INSTRUCTOR_CODES)):
            ws.cell(row=row, column=INSTRUCTOR_START_COL + j).border = box
        # Reference columns -- convert() never reads past the code columns.
        ws.cell(row=row, column=note_col, value=chunk)
        if (chunk + 1) * window_size_min > duration_min:
            ws.cell(row=row, column=note_col + 1,
                    value=f"PARTIAL - video ends at {duration_min:.2f} min")
        elif i > 0 and chunk != indices[i - 1] + 1:
            ws.cell(row=row, column=note_col + 1, value="^ gap - new block")

    ws.column_dimensions["A"].width = 10
    for j in range(len(INSTRUCTOR_CODES)):
        ws.column_dimensions[ws.cell(row=3, column=INSTRUCTOR_START_COL + j)
                             .column_letter].width = 5
    ws.column_dimensions[ws.cell(row=3, column=note_col).column_letter].width = 7
    ws.column_dimensions[ws.cell(row=3, column=note_col + 1).column_letter].width = 34
    ws.freeze_panes = ws.cell(row=DATA_START_ROW, column=INSTRUCTOR_START_COL)

    os.makedirs(os.path.dirname(xlsx_path) or ".", exist_ok=True)
    wb.save(xlsx_path)
    return xlsx_path


def main():
    parser = argparse.ArgumentParser(
        description="Compute COPUS sparse-sampling chunk indices for one lecture"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--duration",
                     help="Lecture running time: MIN (48.33), MM:SS (75:30), "
                          "or HH:MM:SS (1:15:30)")
    src.add_argument("--video", help="Lecture .mp4 -- duration is read from the file")
    parser.add_argument("--window-size", type=float, default=2.0,
                        help="Window size in minutes (default: 2.0)")
    parser.add_argument("--make-template", metavar="OUT.xlsx", default=None,
                        help="Also write a COPUS coding sheet holding exactly these "
                             "windows, with the correct minute labels")
    args = parser.parse_args()

    if args.video:
        duration = probe_duration_minutes(args.video)
        print(f"\nProbed {args.video}: {duration * 60:.1f}s")
    else:
        duration = parse_duration_minutes(args.duration)
    args.duration = duration

    indices, scheme, blocks = plan_sparse_windows(args.duration, args.window_size)
    total_chunks = math.ceil(args.duration / args.window_size)

    print(f"\nDuration:  {args.duration:g} min")
    print(f"Window:    {args.window_size:g} min  ->  {total_chunks} chunks total")
    print(f"Scheme:    {scheme}")
    print(f"\n{'Block':<12}{'Chunks':>12}{'Minutes':>18}")
    print("-" * 42)
    for label, first_chunk, last_chunk in blocks:
        start_min = first_chunk * args.window_size
        end_min = min((last_chunk + 1) * args.window_size, args.duration)
        partial = " (partial)" if (last_chunk + 1) * args.window_size > args.duration else ""
        print(f"{label:<12}{f'{first_chunk}-{last_chunk}':>12}"
              f"{f'{start_min:g}-{end_min:g}':>18}{partial}")
    print(f"\nSampled: {len(indices)} chunks "
          f"= {len(indices) * args.window_size:g} min of {args.duration:g} min")
    print("\nChunk indices (paste into --windows):\n")
    print(",".join(str(i) for i in indices))

    if args.make_template:
        path = write_template(args.make_template, indices,
                              args.window_size, args.duration)
        print(f"\nCoding sheet written -> {path}")
        print(f"  {len(indices)} rows, one per sampled window, "
              f"labelled with each window's real minute range.")
        print("  Code it top to bottom; the row order IS the --windows order.")


if __name__ == "__main__":
    main()
