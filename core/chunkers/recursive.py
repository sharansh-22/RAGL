from core.chunkers.base import BaseChunker, logger

class RecursiveChunker(BaseChunker):
    """
    Langchain-style Recursive Character Chunker.
    Attempts to split by double newlines, then newlines, then spaces.
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        super().__init__(chunk_size, chunk_overlap)
        self.separators = ["\n\n", "\n", " ", ""]

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        final_chunks = []
        separator = separators[0]
        has_separators = False
        
        for sep in separators:
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                has_separators = True
                break
                
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        good_splits = []
        for s in splits:
            if s:
                good_splits.append(s)

        current_doc = []
        current_len = 0
        
        for s in good_splits:
            s_len = self._approximate_token_count(s)
            
            if current_len + s_len > self.chunk_size and current_doc:
                final_chunks.append(separator.join(current_doc))
                
                # Overlap logic
                overlap_len = 0
                rewind = []
                for prev in reversed(current_doc):
                    prev_len = self._approximate_token_count(prev)
                    if overlap_len + prev_len > self.chunk_overlap:
                        break
                    overlap_len += prev_len
                    rewind.insert(0, prev)
                
                current_doc = rewind
                current_len = overlap_len
                
            # If a single piece is too large, recurse
            if s_len > self.chunk_size and has_separators:
                # Add existing chunks
                if current_doc:
                    final_chunks.append(separator.join(current_doc))
                    current_doc = []
                    current_len = 0
                
                next_separators = separators[separators.index(separator) + 1:]
                if next_separators:
                    recursed = self._split_text(s, next_separators)
                    final_chunks.extend(recursed)
                else:
                    final_chunks.append(s)
            else:
                current_doc.append(s)
                current_len += s_len
                
        if current_doc:
            final_chunks.append(separator.join(current_doc))
            
        return final_chunks

    def chunk_documents(self, documents: list[dict]) -> list[dict]:
        all_chunks = []
        
        for doc_idx, doc in enumerate(documents):
            text = doc["text"]
            if not text.strip():
                continue
                
            chunks = self._split_text(text, self.separators)
            
            for chunk_idx, chunk_text in enumerate(chunks):
                all_chunks.append({
                    "text": chunk_text,
                    "source": doc["source"],
                    "category": doc["category"],
                    "chunk_index": chunk_idx,
                    "doc_index": doc_idx,
                })
            
            logger.info(f"  {doc['source']}: {len(chunks)} chunks")
            
        logger.info(f"Total chunks created by RecursiveChunker: {len(all_chunks)}")
        return all_chunks
