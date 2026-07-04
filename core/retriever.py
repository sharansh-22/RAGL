"""
RAGL — Retriever
=================
Responsible for:
  - Performing similarity search against a FAISS index
  - Returning Top-k chunks with scores

Not responsible for prompt construction, generation, or embedding.
"""

import logging

import faiss
import numpy as np

logger = logging.getLogger(__name__)


def retrieve(
    query_embedding: np.ndarray,
    index: faiss.Index,
    chunks: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Retrieve the top-k most similar chunks for a query embedding.

    Args:
        query_embedding: np.ndarray of shape (1, dimension), dtype float32.
        index: A populated FAISS index.
        chunks: The list of chunk dicts (parallel to the index vectors).
        top_k: Number of results to return.

    Returns:
        List of dicts, each containing the chunk fields plus:
          - score: similarity score from FAISS
          - rank: 1-based rank
    """
    if query_embedding.ndim != 2 or query_embedding.shape[0] != 1:
        raise ValueError(
            f"Expected shape (1, dim), got {query_embedding.shape}"
        )

    # Normalize query for cosine similarity via inner product
    faiss.normalize_L2(query_embedding)

    # Clamp top_k to available vectors
    k = min(top_k, index.ntotal)
    scores, indices = index.search(query_embedding, k)

    results = []
    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]), start=1
    ):
        if idx == -1:
            # FAISS returns -1 for unfilled slots
            continue

        result = {
            **chunks[idx],
            "score": float(score),
            "rank": rank,
        }
        results.append(result)

    logger.debug(f"Retrieved {len(results)} chunks (top_k={top_k})")
    return results
