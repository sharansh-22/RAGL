"""
RAGL — Indexer
===============
Responsible for:
  - Building a FAISS index from embeddings
  - Saving an index to disk
  - Loading an index from disk

Not responsible for querying, retrieval, or embedding.
"""

import logging
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS IndexFlatIP (inner product) index.

    BGE embeddings are L2-normalized, so inner product = cosine similarity.

    Args:
        embeddings: np.ndarray of shape (n, dimension), dtype float32.

    Returns:
        A populated FAISS index.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {embeddings.shape}")

    n, dimension = embeddings.shape
    logger.info(f"Building FAISS IndexFlatIP: {n} vectors, dim={dimension}")

    # Normalize embeddings for cosine similarity via inner product
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    logger.info(f"Index built: {index.ntotal} vectors indexed")
    return index


def save_index(index: faiss.Index, path: str) -> None:
    """
    Save a FAISS index to disk.

    Args:
        index: The FAISS index to save.
        path: File path to save to (e.g., "A/A0/index/faiss.index").
    """
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(save_path))
    logger.info(f"Index saved to: {save_path} ({index.ntotal} vectors)")


def load_index(path: str) -> faiss.Index:
    """
    Load a FAISS index from disk.

    Args:
        path: File path to load from.

    Returns:
        The loaded FAISS index.
    """
    load_path = Path(path)
    if not load_path.exists():
        raise FileNotFoundError(f"Index not found: {load_path}")

    index = faiss.read_index(str(load_path))
    logger.info(f"Index loaded from: {load_path} ({index.ntotal} vectors)")
    return index
