"""
RAGL — Chunker Factory
======================
Delegates chunking to a specific strategy module in `core/chunkers/`.
Defaults to `sentence` for backward compatibility with A0.
"""

import logging
import os
from pathlib import Path
import pymupdf4llm

from core.chunkers import RecursiveChunker, SentenceChunker, StructureChunker, SemanticChunker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Document Loading
# ---------------------------------------------------------------------------

def load_documents(data_dir: str) -> list[dict]:
    """
    Recursively walk `data_dir`, extract text from every PDF.
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
# Factory Delegation
# ---------------------------------------------------------------------------

def chunk_documents(
    documents: list[dict],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    strategy: str = "sentence"
) -> list[dict]:
    """
    Factory that delegates chunking to a specific strategy.
    
    Strategies:
      - "sentence": Sentence-aware recursive chunking (A0 default)
      - "recursive": Recursive character chunking
      - "structure": Markdown heading chunking
      - "semantic": Semantic similarity boundary chunking
    """
    logger.info(f"Using chunking strategy: {strategy}")
    
    if strategy == "recursive":
        chunker = RecursiveChunker(chunk_size, chunk_overlap)
    elif strategy == "structure":
        chunker = StructureChunker(chunk_size, chunk_overlap)
    elif strategy == "semantic":
        chunker = SemanticChunker(chunk_size, chunk_overlap)
    else:
        # Default for backward compatibility with A0 and A1
        chunker = SentenceChunker(chunk_size, chunk_overlap)
        
    return chunker.chunk_documents(documents)
