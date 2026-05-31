from typing import List, Dict, Any


class RecursiveLegalTextSplitter:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100, separators: List[str] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\nĐiều", "\nKhoản", "\nĐiểm", "\n\n", "\n", " ", ""]

    def split_text(self, text: str) -> List[str]:
        def recurse(text_to_split: str, separator_idx: int) -> List[str]:
            if len(text_to_split) <= self.chunk_size:
                return [text_to_split]
            if separator_idx >= len(self.separators):
                # No more separators, fallback to character slices
                return [text_to_split[i:i + self.chunk_size] for i in range(0, len(text_to_split), self.chunk_size - self.chunk_overlap)]
            
            separator = self.separators[separator_idx]
            if separator == "":
                return [text_to_split[i:i + self.chunk_size] for i in range(0, len(text_to_split), self.chunk_size - self.chunk_overlap)]
                
            if separator not in text_to_split:
                return recurse(text_to_split, separator_idx + 1)
                
            # Split and keep the separator for legal structure visibility
            parts = text_to_split.split(separator)
            reconstructed = []
            for i, part in enumerate(parts):
                if i == 0:
                    if part:
                        reconstructed.append(part)
                else:
                    reconstructed.append(separator + part)
                    
            final_chunks = []
            for part in reconstructed:
                if len(part) > self.chunk_size:
                    final_chunks.extend(recurse(part, separator_idx + 1))
                else:
                    final_chunks.append(part)
                    
            return final_chunks

        raw_splits = recurse(text, 0)
        
        # Merge small chunks to reach chunk_size while keeping overlap
        merged_chunks = []
        current_chunk = ""
        
        for split in raw_splits:
            if not split.strip():
                continue
            if not current_chunk:
                current_chunk = split
            elif len(current_chunk) + len(split) <= self.chunk_size:
                current_chunk += split
            else:
                merged_chunks.append(current_chunk)
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                current_chunk = overlap_text + split
                
        if current_chunk:
            merged_chunks.append(current_chunk)
            
        return merged_chunks


def chunk_document(doc: Dict[str, Any], chunk_size: int = 800, chunk_overlap: int = 100) -> List[Dict[str, Any]]:
    """Split a legal document's content into smaller, semantically intact chunks

    using structural legal separators ("\nĐiều", "\nKhoản", "\nĐiểm").
    """
    splitter = RecursiveLegalTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    raw_content = doc.get("raw_content", "")
    split_texts = splitter.split_text(raw_content)
    
    chunks = []
    for idx, text in enumerate(split_texts):
        text_stripped = text.strip()
        if not text_stripped:
            continue
            
        # Enrich the vector text with document context so embeddings capture the source metadata
        enriched_text = f"Văn bản: {doc['title']} ({doc['doc_id']})\n{text_stripped}"
        
        chunks.append({
            "chunk_id": f"{doc['doc_id']}_chunk_{idx}",
            "doc_id": doc["doc_id"],
            "text": enriched_text,          # Enriched text for embedding/dense retrieval
            "raw_text": text_stripped,       # Raw text for exact matching or display
            "metadata": {
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "doc_type": doc["doc_type"],
                "issue_date": doc["issue_date"],
                "status": doc["status"],
                "chunk_index": idx
            }
        })
        
    return chunks


def chunk_documents(documents: List[Dict[str, Any]], chunk_size: int = 800, chunk_overlap: int = 100) -> List[Dict[str, Any]]:
    """Chunk a list of legal documents and return a flat list of enriched chunks."""
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc, chunk_size, chunk_overlap))
    return all_chunks
