"""Project configuration and paths."""

import os
from pathlib import Path

# Load .env file if present
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().strip().splitlines():
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_IMAGES_DIR = DATA_DIR / "sample_images"
CACHE_DIR = DATA_DIR / "cache"

# Results paths
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"

# Model config
MODEL_NAME = "Salesforce/blip2-opt-2.7b"
DEVICE = "cuda"

# VQAv2 dataset config
VQAV2_DATASET = "lmms-lab/VQAv2"

# Fine-tuning config
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 3
MAX_LENGTH = 128

# RAG config
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
RAG_INDEX_DIR = DATA_DIR / "rag_index"
RAG_KB_SIZE = 5000
RAG_TOP_K = 3
RAG_ALPHA = 0.5
RAG_CANDIDATE_K = 30
RAG_TAU = 0.3

# Create directories
for d in [DATA_DIR, SAMPLE_IMAGES_DIR, CACHE_DIR, RESULTS_DIR, CHECKPOINTS_DIR,
          RAG_INDEX_DIR]:
    d.mkdir(parents=True, exist_ok=True)
