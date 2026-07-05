from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class BaseChunker(ABC):
    """
    Abstract base class for all RAGL document chunking strategies.
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def chunk_documents(self, documents: list[dict]) -> list[dict]:
        """
        Splits a list of documents into chunks.
        
        Args:
            documents: List of dicts `[{"text": "...", "source": "...", "category": "..."}]`
            
        Returns:
            List of dicts: `[{"text": "...", "source": "...", "category": "...", "chunk_index": i, "doc_index": j}]`
        """
        pass

    def _approximate_token_count(self, text: str) -> int:
        """
        Approximate token count using whitespace splitting.
        """
        return len(text.split())
