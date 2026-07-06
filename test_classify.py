import os
from dotenv import load_dotenv

load_dotenv()  # loads your .env file

from classify.classifier import classify_chunk_multimodal

# Test on just the first chunk
classify_chunk_multimodal(
    video_path="output/test_chunks/chunk_001_full.mp4",
    chunk_index=1
)