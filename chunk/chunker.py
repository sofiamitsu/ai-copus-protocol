import ffmpeg
import math
import os


def expected_chunk_count(duration_seconds, window_seconds=120):
    """
    How many chunks a lecture of this length produces.

        N = ceil(duration_seconds / window_seconds)

    The final chunk may be shorter than window_seconds -- partial windows are
    kept, not truncated. This is the number Sofia should be able to predict from
    a lecture's duration alone and see match the AI output.
    """
    return math.ceil(duration_seconds / window_seconds)


def _write_chunk(input_path, start, end, out_path, copy_kwargs, encode_kwargs):
    """
    Cut one chunk, preferring a fast stream copy and falling back to a re-encode.

    `c="copy"` is fast but only works when the source codec can live in the target
    container: a ProRes or HEVC .mov copied into .mp4 fails outright, and ffmpeg
    leaves a 0-byte file behind when it does. That empty file then looks "already
    chunked" to the idempotency check, so classification later fails on an empty
    input. Re-encoding a single 2-minute window is cheap, unlike re-encoding a
    multi-GB lecture.
    """
    last_error = None
    for mode, kwargs in (("copy", copy_kwargs), ("re-encode", encode_kwargs)):
        try:
            (
                ffmpeg
                .input(input_path, ss=start, to=end)
                .output(out_path, **kwargs)
                .overwrite_output()
                .run(quiet=True)
            )
            if os.path.getsize(out_path) > 0:
                return mode
            raise RuntimeError("ffmpeg wrote an empty file")
        except Exception as e:
            last_error = e
            # Never leave a partial or empty file behind.
            if os.path.exists(out_path):
                os.remove(out_path)
    raise RuntimeError(f"could not write {os.path.basename(out_path)}: {last_error}")


def chunk_video(input_path, output_dir, window_seconds=120, max_chunks=None,
                chunk_indices=None):
    """
    Takes a lecture .mp4 and splits it into 2-minute windows.
    For each window, produces three files:
      - chunk_NNN_full.mp4   (video + audio)
      - chunk_NNN_muted.mp4  (video only, no audio)
      - chunk_NNN_audio.mp3  (audio only)

    chunk_indices restricts the cut to those windows only (sparse sampling) --
    on a 2-hr lecture that is 32 windows instead of 60, times three variants
    each. Indices past the end of the video are ignored.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: probe the video to get its total duration in seconds
    probe = ffmpeg.probe(input_path)
    duration = float(probe["format"]["duration"])
    total_chunks = expected_chunk_count(duration, window_seconds)

    print(f"Video duration: {duration:.1f}s")
    print(f"Window size: {window_seconds}s")
    print(f"Expected chunks: {total_chunks}")

    if chunk_indices is None:
        wanted = list(range(total_chunks))
    else:
        wanted = sorted({i for i in chunk_indices if 0 <= i < total_chunks})
        dropped = sorted(set(chunk_indices) - set(wanted))
        if dropped:
            print(f"[warn] ignoring {len(dropped)} requested chunk(s) past the end "
                  f"of the video (only {total_chunks} exist): {dropped}")
        print(f"Sparse cut: {len(wanted)} of {total_chunks} chunks")
    if max_chunks is not None:
        wanted = wanted[:max_chunks]

    reencoded = False
    for chunk_index in wanted:
        start = chunk_index * window_seconds
        end = min(start + window_seconds, duration)
        chunk_name = f"chunk_{chunk_index:03d}"  # e.g. chunk_000, chunk_001
        is_short = (end - start) < window_seconds

        print(f"\nChunk {chunk_index}: {start:.1f}s -> {end:.1f}s", end="")
        if is_short:
            print(" [SHORT -- last window]", end="")
        print()

        # Full video + audio
        mode = _write_chunk(
            input_path, start, end,
            os.path.join(output_dir, f"{chunk_name}_full.mp4"),
            copy_kwargs={"c": "copy"},  # don't re-encode, just cut -- fast
            encode_kwargs={"vcodec": "libx264", "acodec": "aac", "preset": "veryfast"},
        )
        if mode == "re-encode" and not reencoded:
            print(f"  [note] source codec can't be copied into .mp4 — re-encoding "
                  f"chunks (slower, but only per 2-min window)")
            reencoded = True

        # Muted video (video stream only, no audio stream)
        _write_chunk(
            input_path, start, end,
            os.path.join(output_dir, f"{chunk_name}_muted.mp4"),
            copy_kwargs={"an": None, "c": "copy"},  # "an" = audio none
            encode_kwargs={"an": None, "vcodec": "libx264", "preset": "veryfast"},
        )

        # Audio only as mp3 (always re-encoded -- mp3 is never a copy target here)
        _write_chunk(
            input_path, start, end,
            os.path.join(output_dir, f"{chunk_name}_audio.mp3"),
            copy_kwargs={"vn": None},   # "vn" = video none = strip video
            encode_kwargs={"vn": None, "acodec": "libmp3lame"},
        )

    print(f"\nDone. {len(wanted)} chunks saved to: {output_dir}")
