"""
RAGL — Retrieval Metrics
=========================
Pure functions for evaluating retrieval quality.

All functions compare retrieved chunk sources against expected sources
from the gold dataset.

Metrics:
  - Recall@k
  - Precision@k
  - MRR (Mean Reciprocal Rank)
  - NDCG (Normalized Discounted Cumulative Gain)
  - Hit Rate

No side effects. No model dependencies. No I/O.
"""

import math
from typing import Optional


def _get_retrieved_sources(retrieved_chunks: list[dict]) -> list[str]:
    """Extract source filenames from retrieved chunks, preserving order."""
    return [chunk.get("source", "") for chunk in retrieved_chunks]


def recall_at_k(
    retrieved_chunks: list[dict],
    expected_sources: list[str],
    k: Optional[int] = None,
) -> float:
    """
    Recall@k: fraction of expected sources found in the top-k retrieved chunks.

    Args:
        retrieved_chunks: List of chunk dicts with 'source' field.
        expected_sources: List of expected source filenames.
        k: Cutoff (defaults to len(retrieved_chunks)).

    Returns:
        Float in [0.0, 1.0].
    """
    if not expected_sources:
        return 0.0

    sources = _get_retrieved_sources(retrieved_chunks)
    if k is not None:
        sources = sources[:k]

    retrieved_set = set(sources)
    expected_set = set(expected_sources)
    hits = retrieved_set & expected_set

    return len(hits) / len(expected_set)


def precision_at_k(
    retrieved_chunks: list[dict],
    expected_sources: list[str],
    k: Optional[int] = None,
) -> float:
    """
    Precision@k: fraction of top-k retrieved chunks from expected sources.

    Args:
        retrieved_chunks: List of chunk dicts with 'source' field.
        expected_sources: List of expected source filenames.
        k: Cutoff (defaults to len(retrieved_chunks)).

    Returns:
        Float in [0.0, 1.0].
    """
    sources = _get_retrieved_sources(retrieved_chunks)
    if k is not None:
        sources = sources[:k]

    if not sources:
        return 0.0

    expected_set = set(expected_sources)
    relevant = sum(1 for s in sources if s in expected_set)

    return relevant / len(sources)


def mrr(
    retrieved_chunks: list[dict],
    expected_sources: list[str],
) -> float:
    """
    Mean Reciprocal Rank: 1 / rank of the first relevant result.

    Args:
        retrieved_chunks: List of chunk dicts with 'source' field.
        expected_sources: List of expected source filenames.

    Returns:
        Float in [0.0, 1.0]. Returns 0.0 if no relevant result found.
    """
    expected_set = set(expected_sources)
    sources = _get_retrieved_sources(retrieved_chunks)

    for rank, source in enumerate(sources, start=1):
        if source in expected_set:
            return 1.0 / rank

    return 0.0


def ndcg(
    retrieved_chunks: list[dict],
    expected_sources: list[str],
    k: Optional[int] = None,
) -> float:
    """
    Normalized Discounted Cumulative Gain at k.

    Binary relevance: 1 if source is in expected_sources, 0 otherwise.

    Args:
        retrieved_chunks: List of chunk dicts with 'source' field.
        expected_sources: List of expected source filenames.
        k: Cutoff (defaults to len(retrieved_chunks)).

    Returns:
        Float in [0.0, 1.0].
    """
    sources = _get_retrieved_sources(retrieved_chunks)
    if k is not None:
        sources = sources[:k]

    if not sources or not expected_sources:
        return 0.0

    expected_set = set(expected_sources)

    # DCG
    dcg = 0.0
    for i, source in enumerate(sources):
        rel = 1.0 if source in expected_set else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 because rank is 1-indexed

    # Ideal DCG: all relevant results at the top
    n_relevant = min(len(expected_set), len(sources))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_relevant))

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def hit_rate(
    retrieved_chunks: list[dict],
    expected_sources: list[str],
) -> float:
    """
    Hit Rate: 1.0 if any expected source appears in retrieved chunks, else 0.0.

    Args:
        retrieved_chunks: List of chunk dicts with 'source' field.
        expected_sources: List of expected source filenames.

    Returns:
        Float: 1.0 or 0.0.
    """
    if not expected_sources:
        return 0.0

    expected_set = set(expected_sources)
    sources = _get_retrieved_sources(retrieved_chunks)

    for source in sources:
        if source in expected_set:
            return 1.0

    return 0.0


def compute_all_retrieval_metrics(
    retrieved_chunks: list[dict],
    expected_sources: list[str],
    k: Optional[int] = None,
) -> dict:
    """
    Compute all retrieval metrics for a single query.

    Returns:
        Dict with keys: recall_at_k, precision_at_k, mrr, ndcg, hit_rate.
    """
    return {
        "recall_at_k": recall_at_k(retrieved_chunks, expected_sources, k),
        "precision_at_k": precision_at_k(retrieved_chunks, expected_sources, k),
        "mrr": mrr(retrieved_chunks, expected_sources),
        "ndcg": ndcg(retrieved_chunks, expected_sources, k),
        "hit_rate": hit_rate(retrieved_chunks, expected_sources),
    }
