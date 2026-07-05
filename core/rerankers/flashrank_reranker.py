from typing import List, Dict
import os
import logging
from .base import Reranker

logger = logging.getLogger(__name__)

class FlashRankReranker(Reranker):
    """
    Reranker using FlashRank (ONNX backend).
    Optimized for CPU/lightweight GPU inference.
    """
    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2", cache_dir: str = "./cache/flashrank"):
        from flashrank import Ranker
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Initializing FlashRank model: {model_name}")
        self.ranker = Ranker(model_name=model_name, cache_dir=cache_dir)

    def rerank(self, query: str, chunks: List[Dict]) -> List[Dict]:
        from flashrank import RerankRequest
        
        # Prepare passages for FlashRank format
        passages = []
        for idx, chunk in enumerate(chunks):
            passages.append({
                "id": idx,
                "text": chunk["text"],
                # Keep original chunk info so we can map it back
                "meta": chunk
            })
            
        req = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(req)
        
        # FlashRank returns a sorted list of dicts with 'score', 'id', 'text', 'meta'
        reranked_chunks = []
        for res in results:
            # Map back the original chunk
            orig_chunk = res["meta"].copy()
            # Add the reranker score
            orig_chunk["rerank_score"] = float(res["score"])
            reranked_chunks.append(orig_chunk)
            
        return reranked_chunks
