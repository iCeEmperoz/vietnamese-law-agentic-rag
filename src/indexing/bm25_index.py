"""Lexical index module using BM25 for keyword-based search with Vietnamese tokenization."""
import os
import re
import pickle
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

# Default storage paths
BM25_DIR = os.path.join("data", "bm25")
BM25_PATH = os.path.join(BM25_DIR, "bm25_index.pkl")

# Module-level singleton: BM25 index loaded from disk only once per process
_bm25_cache: dict = {}


def tokenize_vietnamese(text: str) -> List[str]:
    """Clean and tokenize Vietnamese text.

    Splits by whitespace, lowercases, and removes punctuation.
    """
    if not text:
        return []
    # Lowercase text
    text = text.lower()
    # Replace non-word and non-spacing characters with space
    text = re.sub(r"[^\w\s\n]", " ", text)
    # Tokenize by splitting whitespace
    tokens = [t.strip() for t in text.split() if t.strip()]
    return tokens


def build_bm25_index(chunks: List[Dict[str, Any]], persist_path: str = BM25_PATH) -> BM25Okapi:
    """Build a BM25 index on enriched document chunks and serialize it to disk."""
    os.makedirs(os.path.dirname(persist_path), exist_ok=True)
    
    # Tokenize corpus (use enriched chunk text containing document title/id for keyword coverage)
    tokenized_corpus = [tokenize_vietnamese(chunk["text"]) for chunk in chunks]
    
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Save the BM25 model, chunks data, and corpus for quick deserialization
    data_to_save = {
        "bm25": bm25,
        "chunks": chunks,
        "tokenized_corpus": tokenized_corpus
    }
    
    with open(persist_path, "wb") as f:
        pickle.dump(data_to_save, f)
        
    print(f"Successfully built BM25 index with {len(chunks)} chunks at {persist_path}.")
    return bm25


def query_bm25(query: str, k: int = 5, persist_path: str = BM25_PATH) -> List[Dict[str, Any]]:
    """Search the BM25 index for keyword matches and return the top k results with normalized scores.

    The index is loaded from disk only on first call and cached in memory thereafter.
    """
    if not os.path.exists(persist_path):
        print(f"Warning: BM25 index not found at {persist_path}. Returning empty list.")
        return []

    # Load from cache; deserialize from disk only on first call
    if persist_path not in _bm25_cache:
        with open(persist_path, "rb") as f:
            _bm25_cache[persist_path] = pickle.load(f)

    data = _bm25_cache[persist_path]
    bm25 = data["bm25"]
    chunks = data["chunks"]

    tokenized_query = tokenize_vietnamese(query)
    scores = bm25.get_scores(tokenized_query)

    # Pair scores with their original chunks, keep only positive-scoring ones
    chunk_scores = [(chunks[idx], score) for idx, score in enumerate(scores) if score > 0]
    chunk_scores.sort(key=lambda x: x[1], reverse=True)

    top_k = chunk_scores[:k]
    max_score = top_k[0][1] if top_k else 0.0

    formatted_results = []
    for chunk, score in top_k:
        normalized_score = float(score / max_score) if max_score > 0 else 0.0
        formatted_results.append({
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "text": chunk["text"],
            "score": normalized_score,
            "raw_score": float(score),
            "metadata": chunk["metadata"]
        })

    return formatted_results


def invalidate_bm25_cache():
    """Invalidate the BM25 in-memory cache (call after re-ingestion)."""
    _bm25_cache.clear()
