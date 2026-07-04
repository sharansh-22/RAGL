"""
RAGL — A0 Reference Implementation Configuration
===================================================
All constants for the A0 experiment in one place.

A0 is the baseline against which every future experiment is benchmarked.
Nothing here should be modified without creating a new experiment (A1, A2, ...).
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to RAGL root)
# ---------------------------------------------------------------------------
RAGL_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = str(RAGL_ROOT / "data")
INDEX_DIR = str(RAGL_ROOT / "A" / "A0" / "artifacts")

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE = 512       # approximate tokens per chunk
CHUNK_OVERLAP = 64     # approximate token overlap between chunks

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"   # 384-dim, GPU via ONNX

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K = 5

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
LLM_MODEL = "llama3:8b"

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
EXPERIMENT_NAME = "A0"
EXPERIMENT_STATUS = "Frozen"
VERSION = "0.1"
EVAL_DATASETS_DIR = str(RAGL_ROOT / "evaluation" / "datasets")
EVAL_REPORTS_DIR = str(RAGL_ROOT / "evaluation" / "reports")
EVAL_PLOTS_DIR = str(RAGL_ROOT / "evaluation" / "plots" / "A0")
EVAL_CACHE_DIR = str(RAGL_ROOT / "evaluation" / "cache" / "A0")

# ---------------------------------------------------------------------------
# Evaluation Models (NLI + Cross-Encoder)
# ---------------------------------------------------------------------------
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
