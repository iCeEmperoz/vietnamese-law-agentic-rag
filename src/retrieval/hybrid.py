"""Hybrid retrieval module combining Dense and Lexical search via Reciprocal Rank Fusion (RRF)."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from src.retrieval.dense import retrieve_dense
from src.retrieval.bm25 import retrieve_bm25
from configs.setting import SETTINGS


def retrieve_hybrid(query: str, k: int = 5, rrf_k: int = None) -> List[Dict[str, Any]]:
    """Retrieve chunks using hybrid search combining Dense and BM25 with Reciprocal Rank Fusion (RRF).

    Dense and BM25 retrievals run concurrently to reduce end-to-end latency.
    """
    if rrf_k is None:
        rrf_k = SETTINGS.get("retrieval", {}).get("rrf_k", 60)

    print(f"Running Hybrid RRF Retrieval (k={k}, rrf_k={rrf_k})")

    # Fetch a wider candidate pool from both strategies — run in PARALLEL
    pool_size = k * 3
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_dense = executor.submit(retrieve_dense, query, pool_size)
        future_bm25 = executor.submit(retrieve_bm25, query, pool_size)
        dense_results = future_dense.result()
        bm25_results = future_bm25.result()
    
    rrf_scores = {}
    chunk_lookup = {}
    
    # 1. Process dense rankings
    for rank, chunk in enumerate(dense_results):
        chunk_id = chunk["chunk_id"]
        chunk_lookup[chunk_id] = chunk
        # RRF formula: 1 / (rrf_k + rank) where rank is 1-indexed
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + (rank + 1)))
        
    # 2. Process BM25 rankings
    for rank, chunk in enumerate(bm25_results):
        chunk_id = chunk["chunk_id"]
        if chunk_id not in chunk_lookup:
            chunk_lookup[chunk_id] = chunk
        # Add RRF reciprocal score
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + (rank + 1)))
        
    # 3. Sort chunk keys by accumulated fusion score descending
    sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    
    # 4. Form final top-k hybrid candidates
    hybrid_results = []
    for chunk_id in sorted_chunk_ids[:k]:
        chunk_copy = chunk_lookup[chunk_id].copy()
        chunk_copy["rrf_score"] = float(rrf_scores[chunk_id])
        chunk_copy["score"] = float(rrf_scores[chunk_id])  # standard search score API
        hybrid_results.append(chunk_copy)
        
    print(f"Hybrid RRF fusion selected top {len(hybrid_results)} candidates out of {len(rrf_scores)} unique pool items.")
    return hybrid_results
