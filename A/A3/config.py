import os

# Base configuration (inherited from A0)
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "llama3:8b"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K = 5

# Reranker specific configuration
RETRIEVE_TOP_K = 20

# Reusing A0 indices absolutely!
DATA_DIR = "data"
INDEX_DIR = "A/A0/artifacts"

# Evaluation config
EVAL_DATASETS_DIR = "evaluation/datasets"
EVAL_CACHE_DIR = "evaluation/cache/A3"
EVAL_REPORTS_DIR = "evaluation/reports"
EVAL_PLOTS_DIR = "evaluation/plots/A3"

# Reranker candidates
CANDIDATES = {
    "A0_No_Reranker": {
        "strategy": "none"
    },
    "FlashRank_MiniLM": {
        "strategy": "flashrank",
        "model_name": "ms-marco-MiniLM-L-12-v2"
    },
    "BGE_Reranker_Base": {
        "strategy": "cross_encoder",
        "model_name": "BAAI/bge-reranker-base"
    },
    "CrossEncoder_MiniLM": {
        "strategy": "cross_encoder",
        "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2"
    }
}
