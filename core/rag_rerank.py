"""
RAGL — Reranking RAG Pipeline
=============================
Alternative RAG pipeline that inserts a reranking stage between retrieval and generation.
Maintains all original A0 retrieval logic but modifies the ordering of the candidate pool.
"""

import logging
import time

from core.embedder import Embedder
from core.retriever import retrieve
from core.generator import build_prompt, generate
from core.rerankers import Reranker

logger = logging.getLogger(__name__)

class RerankRAGPipeline:
    def __init__(
        self,
        embedder: Embedder,
        index: faiss.Index,
        chunks: list[dict],
        model_name: str = "llama3:8b",
        retrieve_top_k: int = 20,
        final_top_k: int = 5,
        reranker: Reranker = None,
    ):
        """
        Initialize the Reranking RAG pipeline.

        Args:
            embedder: An Embedder instance for query embedding.
            index: A populated FAISS index.
            chunks: The chunk list (parallel to index vectors).
            model_name: Ollama model identifier for generation.
            retrieve_top_k: Number of chunks to initially retrieve from FAISS (Candidate Pool).
            final_top_k: Number of chunks to send to the LLM after reranking.
            reranker: An instance of Reranker protocol. If None, just truncates retrieval.
        """
        self.embedder = embedder
        self.index = index
        self.chunks = chunks
        self.model_name = model_name
        self.retrieve_top_k = retrieve_top_k
        self.final_top_k = final_top_k
        self.reranker = reranker

    def query(self, question: str) -> dict:
        total_start = time.perf_counter()

        # 1. Base Retrieval
        retrieval_start = time.perf_counter()
        query_embedding = self.embedder.embed_query(question)
        candidate_chunks = retrieve(
            query_embedding, self.index, self.chunks, self.retrieve_top_k
        )
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        # Track the "before" state (what A0 would have seen)
        # We assign a rank for clarity in the output
        retrieved_before = []
        for i, c in enumerate(candidate_chunks[:self.final_top_k]):
            retrieved_before.append({
                "rank": i + 1,
                "source": c.get("source"),
                "score": c.get("score"),
                "id": c.get("id", i)
            })

        # 2. Reranking
        reranking_start = time.perf_counter()
        if self.reranker and candidate_chunks:
            reranked_chunks = self.reranker.rerank(question, candidate_chunks)
            final_chunks = reranked_chunks[:self.final_top_k]
        else:
            final_chunks = candidate_chunks[:self.final_top_k]
        reranking_ms = (time.perf_counter() - reranking_start) * 1000

        # Track the "after" state
        retrieved_after = []
        for i, c in enumerate(final_chunks):
            retrieved_after.append({
                "rank": i + 1,
                "source": c.get("source"),
                "rerank_score": c.get("rerank_score", c.get("score")),
                "id": c.get("id", i)
            })

        # 3. Build Prompt
        prompt = build_prompt(question, final_chunks)

        # 4. Generate
        generation_start = time.perf_counter()
        answer = generate(prompt, self.model_name)
        generation_ms = (time.perf_counter() - generation_start) * 1000

        total_ms = (time.perf_counter() - total_start) * 1000

        return {
            "query": question,
            "answer": answer,
            "retrieved_chunks": final_chunks,  # Sent to metric evaluator
            "retrieved_before": retrieved_before,
            "retrieved_after": retrieved_after,
            "prompt": prompt,
            "latency": {
                "retrieval_ms": round(retrieval_ms, 2),
                "reranking_ms": round(reranking_ms, 2),
                "generation_ms": round(generation_ms, 2),
                "total_ms": round(total_ms, 2),
            },
        }
