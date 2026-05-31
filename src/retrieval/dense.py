"""Dense retrieval module querying ChromaDB for semantic search."""
from typing import List, Dict, Any
from src.indexing.chroma_store import query_vector_store


def retrieve_dense(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Query the ChromaDB vector store using dense semantic similarity.

    Returns a list of matching chunks with similarity scores.
    """
    print(f"🔍 Running Dense Retrieval for: '{query}' (k={k})")
    try:
        return query_vector_store(query, k=k)
    except Exception as e:
        print(f"Error in Dense Retrieval: {e}. Returning empty list.")
        return []
