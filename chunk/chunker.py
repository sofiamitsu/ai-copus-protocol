import ffmpeg
import os

def chunk_video(input_path, output_dir, window_seconds=120, max_chunks=None):
    """
    Takes a lecture .mp4 and splits it into 2-minute windows.
    For each window, produces three files:
      - chunk_NNN_full.mp4   (video + audio)
      - chunk_NNN_muted.mp4  (video only, no audio)
      - chunk_NNN_audio.mp3  (audio only)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: probe the video to get its total duration in seconds
    probe = ffmpeg.probe(input_path)
    duration = float(probe["format"]["duration"])

    print(f"Video duration: {duration:.1f}s")
    print(f"Window size: {window_seconds}s")
    print(f"Expected chunks: {int(duration // window_seconds) + 1}")

    chunk_index = 0
    start = 0.0

    while start < duration and (max_chunks is None or chunk_index < max_chunks):
        end = min(start + window_seconds, duration)
        chunk_name = f"chunk_{chunk_index:03d}"  # e.g. chunk_000, chunk_001
        is_short = (end - start) < window_seconds

        print(f"\nChunk {chunk_index}: {start:.1f}s → {end:.1f}s", end="")
        if is_short:
            print(" [SHORT — last window]", end="")
        print()

        # Full video + audio
        (
            ffmpeg
            .input(input_path, ss=start, to=end)
            .output(os.path.join(output_dir, f"{chunk_name}_full.mp4"),
                    c="copy")  # "copy" means don't re-encode, just cut — fast
            .overwrite_output()
            .run(quiet=True)
        )

        # Muted video (video stream only, no audio stream)
        (
            ffmpeg
            .input(input_path, ss=start, to=end)
            .output(os.path.join(output_dir, f"{chunk_name}_muted.mp4"),
                    an=None,   # "an" = audio none = strip audio
                    c="copy")
            .overwrite_output()
            .run(quiet=True)
        )

        # Audio only as mp3
        (
            ffmpeg
            .input(input_path, ss=start, to=end)
            .output(os.path.join(output_dir, f"{chunk_name}_audio.mp3"),
                    vn=None)   # "vn" = video none = strip video
            .overwrite_output()
            .run(quiet=True)
        )

        chunk_index += 1
        start = end

    print(f"\nDone. {chunk_index} chunks saved to: {output_dir}")