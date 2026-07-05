from typing import List, Dict
import logging
import torch
from .base import Reranker

logger = logging.getLogger(__name__)

class CrossEncoderReranker(Reranker):
    """
    Reranker using HuggingFace sentence-transformers CrossEncoder.
    Can be used for BAAI/bge-reranker-base and cross-encoder/ms-marco-MiniLM-L-6-v2.
    """
    def __init__(self, model_name: str, device: str = None):
        from sentence_transformers import CrossEncoder
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        logger.info(f"Initializing CrossEncoder: {model_name} on {device}")
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, chunks: List[Dict]) -> List[Dict]:
        if not chunks:
            return []
            
        # CrossEncoder expects pairs of (query, document)
        pairs = [[query, chunk["text"]] for chunk in chunks]
        
        # Predict scores
        scores = self.model.predict(pairs)
        
        # Zip chunks with their scores
        scored_chunks = []
        for chunk, score in zip(chunks, scores):
            c = chunk.copy()
            c["rerank_score"] = float(score)
            scored_chunks.append(c)
            
        # Sort by rerank_score descending
        scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_chunks
