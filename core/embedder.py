"""
RAGL — Embedder
================
Responsible for:
  - Loading an embedding model (FastEmbed with GPU acceleration)
  - Generating embeddings for texts and queries

Not responsible for indexing, retrieval, or generation.
"""

import logging
import numpy as np
from fastembed import TextEmbedding

logger = logging.getLogger(__name__)


class Embedder:
    """
    Wraps FastEmbed TextEmbedding with GPU acceleration.

    Uses CUDAExecutionProvider when available, falls back to CPU.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """
        Initialize the embedding model.

        Args:
            model_name: HuggingFace model identifier for FastEmbed.
        """
        self.model_name = model_name
        logger.info(f"Loading embedding model: {model_name}")

        try:
            self._model = TextEmbedding(
                model_name=model_name,
                providers=["CUDAExecutionProvider"],
            )
            logger.info("Embedding model loaded with CUDA acceleration")
        except Exception:
            logger.warning("CUDA unavailable, falling back to CPU")
            self._model = TextEmbedding(model_name=model_name)

        self.dimension = self._get_dimension()
        logger.info(f"Embedding dimension: {self.dimension}")

    def _get_dimension(self) -> int:
        """Determine embedding dimension from a probe embedding."""
        probe = list(self._model.embed(["test"]))[0]
        return len(probe)

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Batch embed a list of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            np.ndarray of shape (len(texts), dimension), dtype float32.
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.dimension)

        embeddings = list(self._model.embed(texts))
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.

        Args:
            query: The query text.

        Returns:
            np.ndarray of shape (1, dimension), dtype float32.
        """
        embedding = list(self._model.embed([query]))[0]
        return np.array([embedding], dtype=np.float32)
