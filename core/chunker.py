"""
RAGL — Chunker
===============
Responsible for:
  - Loading documents from a directory (PDF via pymupdf4llm)
  - Sentence-aware recursive chunking with metadata preservation

Not responsible for embedding, indexing, or retrieval.
"""

import os
import re
import logging
from pathlib import Path

import pymupdf4llm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Document Loading
# ---------------------------------------------------------------------------

def load_documents(data_dir: str) -> list[dict]:
    """
    Recursively walk `data_dir`, extract text from every PDF.

    Returns:
        List of dicts: [{text, source, category}]
        - source: filename (e.g., "1706.03762v7.pdf")
        - category: immediate parent directory name (e.g., "Machine learning")
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    documents = []
    pdf_files = sorted(data_path.rglob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {data_dir}")

    logger.info(f"Found {len(pdf_files)} PDF files in {data_dir}")

    for pdf_path in pdf_files:
        logger.info(f"Extracting text from: {pdf_path.name}")
        try:
            # pymupdf4llm returns markdown-formatted text
            text = pymupdf4llm.to_markdown(str(pdf_path))

            if not text or not text.strip():
                logger.warning(f"Empty text extracted from: {pdf_path.name}")
                continue

            documents.append({
                "text": text,
                "source": pdf_path.name,
                "category": pdf_path.parent.name,
            })
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path.name}: {e}")
            continue

    logger.info(f"Successfully loaded {len(documents)} documents")
    return documents


# ---------------------------------------------------------------------------
# Sentence-Aware Recursive Chunking
# ---------------------------------------------------------------------------

# Sentence boundary pattern: split on period/question/exclamation followed by
# whitespace and an uppercase letter, or on newlines.
_SENTENCE_PATTERN = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z])'   # sentence boundary
    r'|(?:\n\s*\n)',             # paragraph boundary (double newline)
)


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentence-like segments preserving boundaries."""
    # Use regex split but keep the separators by using a lookahead/lookbehind
    sentences = _SENTENCE_PATTERN.split(text)
    # Filter out empty strings
    return [s.strip() for s in sentences if s and s.strip()]


def _approximate_token_count(text: str) -> int:
    """
    Approximate token count using whitespace splitting.
    This is a rough estimate (~1.3 tokens per word for English).
    Good enough for chunking purposes.
    """
    return len(text.split())


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[dict]:
    """
    Sentence-aware recursive chunking.

    Strategy:
        1. Split document into sentences.
        2. Accumulate sentences until approximate token count reaches chunk_size.
        3. When a chunk is full, create a new chunk starting from an overlap
           point (approximately chunk_overlap tokens back from the end).
        4. Preserve metadata (source, category) for every chunk.

    Args:
        documents: List of dicts from load_documents().
        chunk_size: Target chunk size in approximate tokens.
        chunk_overlap: Overlap between consecutive chunks in approximate tokens.

    Returns:
        List of dicts: [{text, source, category, chunk_index, doc_index}]
    """
    all_chunks = []

    for doc_idx, doc in enumerate(documents):
        sentences = _split_into_sentences(doc["text"])

        if not sentences:
            logger.warning(f"No sentences extracted from: {doc['source']}")
            continue

        chunks = _build_chunks_from_sentences(
            sentences, chunk_size, chunk_overlap
        )

        for chunk_idx, chunk_text in enumerate(chunks):
            all_chunks.append({
                "text": chunk_text,
                "source": doc["source"],
                "category": doc["category"],
                "chunk_index": chunk_idx,
                "doc_index": doc_idx,
            })

        logger.info(
            f"  {doc['source']}: {len(chunks)} chunks "
            f"from {len(sentences)} sentences"
        )

    logger.info(f"Total chunks created: {len(all_chunks)}")
    return all_chunks


def _build_chunks_from_sentences(
    sentences: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """
    Accumulate sentences into chunks respecting approximate token limits.

    When a chunk exceeds chunk_size tokens, finalize it and start a new chunk
    with overlap by rewinding approximately chunk_overlap tokens worth of
    sentences.
    """
    chunks = []
    current_sentences = []
    current_tokens = 0

    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        sentence_tokens = _approximate_token_count(sentence)

        # If a single sentence exceeds chunk_size, it becomes its own chunk
        if sentence_tokens > chunk_size:
            if current_sentences:
                chunks.append(" ".join(current_sentences))
            chunks.append(sentence)
            current_sentences = []
            current_tokens = 0
            i += 1
            continue

        # If adding this sentence would exceed the limit, finalize the chunk
        if current_tokens + sentence_tokens > chunk_size and current_sentences:
            chunks.append(" ".join(current_sentences))

            # Rewind for overlap: find the starting sentence for the next chunk
            overlap_tokens = 0
            rewind_count = 0
            for j in range(len(current_sentences) - 1, -1, -1):
                s_tokens = _approximate_token_count(current_sentences[j])
                if overlap_tokens + s_tokens > chunk_overlap:
                    break
                if overlap_tokens + s_tokens + sentence_tokens > chunk_size:
                    break
                overlap_tokens += s_tokens
                rewind_count += 1

            # Start new chunk with overlapping sentences
            if rewind_count > 0:
                current_sentences = current_sentences[-rewind_count:]
                current_tokens = sum(
                    _approximate_token_count(s) for s in current_sentences
                )
            else:
                current_sentences = []
                current_tokens = 0

        current_sentences.append(sentence)
        current_tokens += sentence_tokens
        i += 1

    # Finalize the last chunk
    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks
