from typing import Protocol, List, Dict

class Reranker(Protocol):
    """Protocol for reranking models."""
    def rerank(self, query: str, chunks: List[Dict]) -> List[Dict]:
        """
        Rerank a list of candidate chunks for a given query.
        
        Args:
            query: The user query string
            chunks: List of chunk dictionaries. Each chunk must contain 'text'.
                   May optionally contain 'source', 'id', 'score', etc.
                   
        Returns:
            A new list of chunks, sorted descending by reranker score.
            Each returned chunk should include a 'rerank_score' key.
        """
        pass
