"""BM25 lexical retrieval module."""
from typing import List, Dict, Any
from src.indexing.bm25_index import query_bm25


def retrieve_bm25(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Query the BM25 index using exact keyword matching.

    Returns a list of matching chunks with normalized BM25 scores.
    """
    print(f"Running BM25 Retrieval for: '{query}' (k={k})")
    try:
        return query_bm25(query, k=k)
    except Exception as e:
        print(f"Error in BM25 Retrieval: {e}. Returning empty list.")
        return []
