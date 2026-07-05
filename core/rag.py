"""
RAGL — RAG Pipeline
====================
Responsible for:
  - Orchestrating the full RAG pipeline: Retrieve → Build Prompt → Generate
  - Timing each stage

Not responsible for individual component logic.
Each component (embedder, retriever, generator) is injected.
"""

import logging
import time

import faiss

from core.embedder import Embedder
from core.retriever import retrieve
from core.generator import build_prompt, generate

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Minimal RAG pipeline for A0 Reference Implementation.

    Orchestration only:
        1. Embed query
        2. Retrieve top-k chunks
        3. Build prompt
        4. Generate answer
    """

    def __init__(
        self,
        embedder: Embedder,
        index: faiss.Index,
        chunks: list[dict],
        model_name: str = "llama3:8b",
        top_k: int = 5,
    ):
        """
        Initialize the RAG pipeline.

        Args:
            embedder: An Embedder instance for query embedding.
            index: A populated FAISS index.
            chunks: The chunk list (parallel to index vectors).
            model_name: Ollama model identifier for generation.
            top_k: Number of chunks to retrieve.
        """
        self.embedder = embedder
        self.index = index
        self.chunks = chunks
        self.model_name = model_name
        self.top_k = top_k

    def query(self, question: str) -> dict:
        """
        Run the full RAG pipeline for a question.

        Args:
            question: The user's question.

        Returns:
            Dict containing:
                - query: the original question
                - answer: generated answer
                - retrieved_chunks: list of retrieved chunk dicts
                - prompt: the constructed prompt
                - latency: dict with retrieval_ms, generation_ms, total_ms
        """
        total_start = time.perf_counter()

        # 1. Retrieve
        retrieval_start = time.perf_counter()
        query_embedding = self.embedder.embed_query(question)
        retrieved = retrieve(
            query_embedding, self.index, self.chunks, self.top_k
        )
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        # 2. Build Prompt
        prompt = build_prompt(question, retrieved)

        # 3. Generate
        generation_start = time.perf_counter()
        answer = generate(prompt, self.model_name)
        generation_ms = (time.perf_counter() - generation_start) * 1000

        total_ms = (time.perf_counter() - total_start) * 1000

        return {
            "query": question,
            "answer": answer,
            "retrieved_chunks": retrieved,
            "prompt": prompt,
            "latency": {
                "retrieval_ms": round(retrieval_ms, 2),
                "generation_ms": round(generation_ms, 2),
                "total_ms": round(total_ms, 2),
            },
        }
