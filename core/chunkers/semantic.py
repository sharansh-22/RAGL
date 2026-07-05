import re
import numpy as np
from core.chunkers.base import BaseChunker, logger

class SemanticChunker(BaseChunker):
    """
    Semantic chunking strategy.
    Splits text into sentences, embeds each sentence using a fast local model,
    and forms chunks based on semantic similarity boundaries (cosine similarity).
    If a chunk exceeds chunk_size, it splits regardless.
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64, similarity_threshold: float = 0.5):
        super().__init__(chunk_size, chunk_overlap)
        self.similarity_threshold = similarity_threshold
        
        self._SENTENCE_PATTERN = re.compile(
            r'(?<=[.!?])\s+(?=[A-Z])'
            r'|(?:\n\s*\n)'
        )
        self._model = None

    def _get_model(self):
        if self._model is None:
            logger.info("Loading lightweight embedder for SemanticChunker (all-MiniLM-L6-v2)...")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    def _split_into_sentences(self, text: str) -> list[str]:
        sentences = self._SENTENCE_PATTERN.split(text)
        return [s.strip() for s in sentences if s and s.strip()]

    def chunk_documents(self, documents: list[dict]) -> list[dict]:
        all_chunks = []
        model = self._get_model()
        
        for doc_idx, doc in enumerate(documents):
            sentences = self._split_into_sentences(doc["text"])
            if not sentences:
                continue

            # Embed all sentences
            embeddings = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
            
            final_chunks = []
            current_sentences = []
            current_tokens = 0
            
            for i in range(len(sentences)):
                sentence = sentences[i]
                sentence_tokens = self._approximate_token_count(sentence)
                
                if current_tokens + sentence_tokens > self.chunk_size and current_sentences:
                    # Force split if it gets too large
                    final_chunks.append(" ".join(current_sentences))
                    current_sentences = [sentence]
                    current_tokens = sentence_tokens
                    continue
                    
                if not current_sentences:
                    current_sentences.append(sentence)
                    current_tokens += sentence_tokens
                    continue
                    
                # Compute similarity with previous sentence
                prev_embedding = embeddings[i-1]
                curr_embedding = embeddings[i]
                similarity = np.dot(prev_embedding, curr_embedding)
                
                if similarity < self.similarity_threshold:
                    # Semantic shift detected, break chunk here
                    final_chunks.append(" ".join(current_sentences))
                    # Note: we do not use overlap in pure semantic boundary splitting,
                    # as the boundary itself represents a topic shift.
                    current_sentences = [sentence]
                    current_tokens = sentence_tokens
                else:
                    current_sentences.append(sentence)
                    current_tokens += sentence_tokens
                    
            if current_sentences:
                final_chunks.append(" ".join(current_sentences))
                
            for chunk_idx, chunk_text in enumerate(final_chunks):
                all_chunks.append({
                    "text": chunk_text,
                    "source": doc["source"],
                    "category": doc["category"],
                    "chunk_index": chunk_idx,
                    "doc_index": doc_idx,
                })
            
            logger.info(f"  {doc['source']}: {len(final_chunks)} chunks (Semantic)")
            
        logger.info(f"Total chunks created by SemanticChunker: {len(all_chunks)}")
        return all_chunks
