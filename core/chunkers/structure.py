import re
from core.chunkers.base import BaseChunker, logger
from core.chunkers.recursive import RecursiveChunker

class StructureChunker(BaseChunker):
    """
    Hierarchical chunker that utilizes Markdown heading structure.
    Splits by headings (#, ##, ###). If a section exceeds chunk_size,
    it falls back to RecursiveChunker for that section.
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        super().__init__(chunk_size, chunk_overlap)
        self.recursive_fallback = RecursiveChunker(chunk_size, chunk_overlap)
        
        # Regex to match markdown headings: `# Heading`
        self.heading_pattern = re.compile(r'^(#{1,6})\s+(.*)', re.MULTILINE)

    def _split_by_headings(self, text: str) -> list[str]:
        """
        Split text by headings, keeping the heading as part of the chunk.
        """
        matches = list(self.heading_pattern.finditer(text))
        if not matches:
            return [text]
            
        sections = []
        
        # Text before the first heading
        first_match = matches[0]
        if first_match.start() > 0:
            prefix = text[:first_match.start()].strip()
            if prefix:
                sections.append(prefix)
                
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i+1].start() if i + 1 < len(matches) else len(text)
            section = text[start:end].strip()
            if section:
                sections.append(section)
                
        return sections

    def chunk_documents(self, documents: list[dict]) -> list[dict]:
        all_chunks = []
        
        for doc_idx, doc in enumerate(documents):
            text = doc["text"]
            if not text.strip():
                continue
                
            sections = self._split_by_headings(text)
            final_doc_chunks = []
            
            for section in sections:
                section_len = self._approximate_token_count(section)
                if section_len <= self.chunk_size:
                    final_doc_chunks.append(section)
                else:
                    # Fall back to recursive chunker for this large section
                    sub_chunks = self.recursive_fallback._split_text(section, self.recursive_fallback.separators)
                    final_doc_chunks.extend(sub_chunks)
                    
            for chunk_idx, chunk_text in enumerate(final_doc_chunks):
                all_chunks.append({
                    "text": chunk_text,
                    "source": doc["source"],
                    "category": doc["category"],
                    "chunk_index": chunk_idx,
                    "doc_index": doc_idx,
                })
            
            logger.info(f"  {doc['source']}: {len(final_doc_chunks)} chunks (Markdown heading split)")
            
        logger.info(f"Total chunks created by StructureChunker: {len(all_chunks)}")
        return all_chunks
