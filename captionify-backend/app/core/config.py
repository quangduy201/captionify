import os
from pathlib import Path

from dotenv import load_dotenv

# Environment variables
load_dotenv()

FRONTEND_ORIGINS = [
    origin.strip()
    for origin in os.getenv("FRONTEND_ORIGINS", "").split(",")
    if origin.strip()
]
MODEL_ID = os.getenv("MODEL_ID", "quangduy201/image-captioning/pyTorch/checkpoint")


# Absolute paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent

ALL_DATASETS_DIR = BASE_DIR / "training" / "datasets"
FLICKR8K_DIR = ALL_DATASETS_DIR / "flickr8k"
ANNOTATION_FILE = FLICKR8K_DIR / "flickr8k_annotations" / "Flickr8k.token.txt"
IMAGE_DIR = FLICKR8K_DIR / "flickr8k_images"
TEST_IMAGE_DIR = ALL_DATASETS_DIR / "test_images"

CACHE_DIR = Path.home() / ".cache"
MODEL_CACHE_DIR = CACHE_DIR / "kagglehub" / "models" / MODEL_ID

OUTPUT_DIR = BASE_DIR / "training" / "output"
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint.pth.tar"
VOCAB_PATH = OUTPUT_DIR / "vocabulary.pkl"
