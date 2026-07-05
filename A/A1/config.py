"""
RAGL — A1 Experiment Configuration (Embedding Model Selection)
==============================================================
This configuration reads the active embedding model from the A1_MODEL
environment variable to benchmark multiple models without modifying
the core runner.
"""

import os
from pathlib import Path

MODEL_ALIASES = {
    "BAAI/bge-small-en-v1.5": "bge_small",
    "BAAI/bge-base-en-v1.5": "bge_base",
    "BAAI/bge-large-en-v1.5": "bge_large",
    "intfloat/e5-base-v2": "e5_base",
    "intfloat/e5-large-v2": "e5_large",
    "nomic-ai/nomic-embed-text-v1.5": "nomic",
    "Snowflake/snowflake-arctic-embed-m-v1.5": "arctic"
}

EMBEDDING_MODEL = os.environ.get("A1_MODEL", "BAAI/bge-small-en-v1.5")
MODEL_ALIAS = MODEL_ALIASES.get(EMBEDDING_MODEL, "unknown")

# ---------------------------------------------------------------------------
# Paths (relative to RAGL root)
# ---------------------------------------------------------------------------
RAGL_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = str(RAGL_ROOT / "data")

# IMPORTANT: Independent artifacts per model
INDEX_DIR = str(RAGL_ROOT / "A" / "A1" / "artifacts" / MODEL_ALIAS)

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
EXPERIMENT_NAME = f"A1_{MODEL_ALIAS}"
EXPERIMENT_STATUS = "Active"
VERSION = "0.1"
EVAL_DATASETS_DIR = str(RAGL_ROOT / "evaluation" / "datasets")
EVAL_REPORTS_DIR = str(RAGL_ROOT / "evaluation" / "reports")
EVAL_PLOTS_DIR = str(RAGL_ROOT / "evaluation" / "plots" / "A1" / MODEL_ALIAS)
EVAL_CACHE_DIR = str(RAGL_ROOT / "evaluation" / "cache" / "A1" / MODEL_ALIAS)

# ---------------------------------------------------------------------------
# Evaluation Models (NLI + Cross-Encoder)
# ---------------------------------------------------------------------------
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
