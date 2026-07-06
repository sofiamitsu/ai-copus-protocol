from chunk.chunker import chunk_video

chunk_video(
    input_path="videoplayback.mp4",  # replace with your actual filename
    output_dir="output/test_chunks",
    window_seconds=120,
    max_chunks=5  # for testing, limit to first 5 chunks
)