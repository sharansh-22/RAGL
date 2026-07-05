"""
RAGL — A2 Experiment Configuration (Chunking Strategy Selection)
==============================================================
This configuration reads the active chunking strategy from the A2_CHUNKER_STRATEGY
environment variable to benchmark multiple strategies without modifying
the core runner.
"""

import os
from pathlib import Path

CHUNKER_STRATEGY = os.environ.get("A2_CHUNKER_STRATEGY", "sentence")

# ---------------------------------------------------------------------------
# Paths (relative to RAGL root)
# ---------------------------------------------------------------------------
RAGL_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = str(RAGL_ROOT / "data")

# IMPORTANT: Independent artifacts per strategy
INDEX_DIR = str(RAGL_ROOT / "A" / "A2" / "artifacts" / CHUNKER_STRATEGY)

# ---------------------------------------------------------------------------
# Embedding (Frozen to A0 Baseline)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE = 512       # approximate tokens per chunk
CHUNK_OVERLAP = 64     # approximate token overlap between chunks

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
EXPERIMENT_NAME = f"A2_{CHUNKER_STRATEGY}"
EXPERIMENT_STATUS = "Active"
VERSION = "0.1"
EVAL_DATASETS_DIR = str(RAGL_ROOT / "evaluation" / "datasets")
EVAL_REPORTS_DIR = str(RAGL_ROOT / "evaluation" / "reports")
EVAL_PLOTS_DIR = str(RAGL_ROOT / "evaluation" / "plots" / "A2" / CHUNKER_STRATEGY)
EVAL_CACHE_DIR = str(RAGL_ROOT / "evaluation" / "cache" / "A2" / CHUNKER_STRATEGY)

# ---------------------------------------------------------------------------
# Evaluation Models (NLI + Cross-Encoder)
# ---------------------------------------------------------------------------
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
