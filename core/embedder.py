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
    Automatically applies recommended query/passage prefixes based on model family.
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
            if "nomic" in model_name.lower() or "snowflake" in model_name.lower():
                raise ValueError("is not supported (forced fallback)")
            
            self._model = TextEmbedding(
                model_name=model_name,
                providers=["CUDAExecutionProvider"],
            )
            self._is_st = False
            logger.info("Embedding model loaded with CUDA acceleration via FastEmbed")
        except ValueError as e:
            if "is not supported" in str(e):
                logger.warning(f"FastEmbed unsupported model '{model_name}'. Falling back to SentenceTransformers.")
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(model_name, device="cuda", trust_remote_code=True)
                self._is_st = True
                logger.info("Embedding model loaded with CUDA via SentenceTransformers")
            else:
                raise
        except Exception:
            logger.warning("CUDA unavailable, falling back to CPU FastEmbed")
            self._model = TextEmbedding(model_name=model_name)
            self._is_st = False

        self.dimension = self._get_dimension()
        logger.info(f"Embedding dimension: {self.dimension}")

        # Configure prefixes
        self._doc_prefix = ""
        self._query_prefix = ""

        lower_name = model_name.lower()
        if "e5" in lower_name:
            self._doc_prefix = "passage: "
            self._query_prefix = "query: "
        elif "nomic" in lower_name:
            self._doc_prefix = "search_document: "
            self._query_prefix = "search_query: "
        elif "arctic" in lower_name:
            self._query_prefix = "Represent this sentence for searching relevant passages: "
        elif "bge" in lower_name:
            self._query_prefix = "Represent this sentence for searching relevant passages: "

        if self._doc_prefix or self._query_prefix:
            logger.info(f"Configured model prefixes -> Query: '{self._query_prefix}', Doc: '{self._doc_prefix}'")

    def _get_dimension(self) -> int:
        """Determine embedding dimension from a probe embedding."""
        if self._is_st:
            return self._model.get_sentence_embedding_dimension()
        probe = list(self._model.embed(["test"]))[0]
        return len(probe)

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Batch embed a list of texts (documents).

        Args:
            texts: List of strings to embed.

        Returns:
            np.ndarray of shape (len(texts), dimension), dtype float32.
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.dimension)

        prefixed_texts = [f"{self._doc_prefix}{text}" for text in texts]
        if self._is_st:
            return self._model.encode(prefixed_texts, batch_size=8, show_progress_bar=False).astype(np.float32)
        else:
            embeddings = list(self._model.embed(prefixed_texts, batch_size=2))
            return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.

        Args:
            query: The query text.

        Returns:
            np.ndarray of shape (1, dimension), dtype float32.
        """
        prefixed_query = f"{self._query_prefix}{query}"
        if self._is_st:
            return self._model.encode([prefixed_query], show_progress_bar=False).astype(np.float32)
        else:
            embedding = list(self._model.embed([prefixed_query]))[0]
            return np.array([embedding], dtype=np.float32)
