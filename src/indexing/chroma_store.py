"""Vector store indexing and querying module using ChromaDB."""
import os
from typing import List, Dict, Any
from langchain_chroma import Chroma
from src.indexing.embeddings import get_embedding_model

# Default storage directory
CHROMA_DIR = os.path.join("data", "chroma")

# Module-level singleton: embeddings + store are initialized only ONCE per process
_chroma_store_cache: dict = {}


def get_chroma_store(persist_directory: str = CHROMA_DIR) -> Chroma:
    """Retrieve the Chroma vector store instance (singleton per directory).

    Avoids re-initializing the embedding model and reconnecting to ChromaDB
    on every query, which was the primary source of latency overhead.
    """
    if persist_directory not in _chroma_store_cache:
        embeddings = get_embedding_model()
        _chroma_store_cache[persist_directory] = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_name="vietnamese_laws_v1"
        )
    return _chroma_store_cache[persist_directory]


def invalidate_store_cache():
    """Invalidate the singleton cache (call after re-ingestion to pick up new data)."""
    _chroma_store_cache.clear()


def add_chunks_to_vector_store(chunks: List[Dict[str, Any]], persist_directory: str = CHROMA_DIR) -> Chroma:
    """Add document chunks to the Chroma vector store."""
    store = get_chroma_store(persist_directory)
    
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    ids = [chunk["chunk_id"] for chunk in chunks]
    
    import time
    # Batch addition to prevent memory or API issues
    batch_size = 30  # Smaller batch size is much safer for Gemini Free Tier
    for i in range(0, len(chunks), batch_size):
        retries = 5
        while retries > 0:
            try:
                store.add_texts(
                    texts=texts[i : i + batch_size],
                    metadatas=metadatas[i : i + batch_size],
                    ids=ids[i : i + batch_size]
                )
                print(f"👉 Indexed batch {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1}...")
                # Add a brief pause between successful batches to respect Free Tier rate limits
                time.sleep(2)
                break
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower() or "resource_exhausted" in str(e).lower():
                    print(f"⚠️ Rate limit hit (429 Resource Exhausted). Sleeping for 35s before retry... (Retries left: {retries})")
                    time.sleep(35)
                    retries -= 1
                else:
                    raise e
        else:
            raise RuntimeError("Failed to index chunks due to persistent Google rate limit (429). Please wait a minute and try again.")
        
    print(f"Successfully indexed {len(chunks)} chunks in ChromaDB vector store.")
    return store


def query_vector_store(query: str, k: int = 5, persist_directory: str = CHROMA_DIR) -> List[Dict[str, Any]]:
    """Query the Chroma vector store for similar chunks."""
    store = get_chroma_store(persist_directory)
    results = store.similarity_search_with_relevance_scores(query, k=k)
    
    formatted_results = []
    for doc, score in results:
        # Normalize score to 0-1 range
        formatted_results.append({
            "chunk_id": doc.metadata.get("chunk_id", f"{doc.metadata.get('doc_id')}_chunk_{doc.metadata.get('chunk_index')}"),
            "doc_id": doc.metadata.get("doc_id"),
            "text": doc.page_content,
            "score": float(score),
            "metadata": doc.metadata
        })
    return formatted_results
