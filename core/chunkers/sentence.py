import re
from core.chunkers.base import BaseChunker, logger

class SentenceChunker(BaseChunker):
    """
    Sentence-aware recursive chunking.
    Preserves natural sentence boundaries.
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        super().__init__(chunk_size, chunk_overlap)
        self._SENTENCE_PATTERN = re.compile(
            r'(?<=[.!?])\s+(?=[A-Z])'   # sentence boundary
            r'|(?:\n\s*\n)',             # paragraph boundary (double newline)
        )

    def _split_into_sentences(self, text: str) -> list[str]:
        sentences = self._SENTENCE_PATTERN.split(text)
        return [s.strip() for s in sentences if s and s.strip()]

    def _build_chunks_from_sentences(self, sentences: list[str]) -> list[str]:
        chunks = []
        current_sentences = []
        current_tokens = 0

        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_tokens = self._approximate_token_count(sentence)

            # If a single sentence exceeds chunk_size, it becomes its own chunk
            if sentence_tokens > self.chunk_size:
                if current_sentences:
                    chunks.append(" ".join(current_sentences))
                chunks.append(sentence)
                current_sentences = []
                current_tokens = 0
                i += 1
                continue

            # If adding this sentence would exceed the limit, finalize the chunk
            if current_tokens + sentence_tokens > self.chunk_size and current_sentences:
                chunks.append(" ".join(current_sentences))

                # Rewind for overlap
                overlap_tokens = 0
                rewind_count = 0
                for j in range(len(current_sentences) - 1, -1, -1):
                    s_tokens = self._approximate_token_count(current_sentences[j])
                    if overlap_tokens + s_tokens > self.chunk_overlap:
                        break
                    if overlap_tokens + s_tokens + sentence_tokens > self.chunk_size:
                        break
                    overlap_tokens += s_tokens
                    rewind_count += 1

                if rewind_count > 0:
                    current_sentences = current_sentences[-rewind_count:]
                    current_tokens = sum(self._approximate_token_count(s) for s in current_sentences)
                else:
                    current_sentences = []
                    current_tokens = 0

            current_sentences.append(sentence)
            current_tokens += sentence_tokens
            i += 1

        if current_sentences:
            chunks.append(" ".join(current_sentences))

        return chunks

    def chunk_documents(self, documents: list[dict]) -> list[dict]:
        all_chunks = []

        for doc_idx, doc in enumerate(documents):
            sentences = self._split_into_sentences(doc["text"])

            if not sentences:
                logger.warning(f"No sentences extracted from: {doc['source']}")
                continue

            chunks = self._build_chunks_from_sentences(sentences)

            for chunk_idx, chunk_text in enumerate(chunks):
                all_chunks.append({
                    "text": chunk_text,
                    "source": doc["source"],
                    "category": doc["category"],
                    "chunk_index": chunk_idx,
                    "doc_index": doc_idx,
                })

            logger.info(f"  {doc['source']}: {len(chunks)} chunks from {len(sentences)} sentences")

        logger.info(f"Total chunks created by SentenceChunker: {len(all_chunks)}")
        return all_chunks
