"""Reranker module to refine candidate ranking using Cross-Encoder or smart LLM-based scoring."""
import os
import re
import json
from typing import List, Dict, Any
from configs.setting import SETTINGS
from src.llm import get_llm

_cross_encoder = None


def get_cross_encoder():
    """Lazy initializer for the local CrossEncoder to save memory on startup."""
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            # Using an extremely lightweight and fast 80MB cross-encoder
            model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            print(f"Loading local CrossEncoder model: {model_name}...")
            _cross_encoder = CrossEncoder(model_name, max_length=512)
            print("CrossEncoder loaded successfully.")
        except Exception as e:
            print(f"Warning: Failed to load local CrossEncoder: {e}.")
            _cross_encoder = "FAILED"
    return _cross_encoder


def llm_rerank_fallback(query: str, candidates: List[Dict[str, Any]]) -> List[float]:
    """Fallback reranker: score all candidates in a single LLM call instead of one per chunk.

    Returns a list of floats in [0.0, 1.0] aligned with the input candidate list.
    """
    print("Using LLM-based batch scoring fallback for reranking...")
    llm = get_llm(temperature=0.0)

    # Build a single batch prompt listing all candidates at once
    numbered_texts = "\n\n".join(
        f"[{i+1}] {item['text'][:400]}"  # truncate to 400 chars to stay within token limits
        for i, item in enumerate(candidates)
    )
    prompt = (
        f"Câu hỏi: \"{query}\"\n\n"
        "Dưới đây là các đoạn văn bản luật (đánh số từ 1). "
        f"Hãy trả về một mảng JSON gồm {len(candidates)} số thực từ 0.0 đến 1.0 "
        "thể hiện mức độ liên quan của từng đoạn với câu hỏi (theo đúng thứ tự). "
        "Chỉ trả về mảng JSON, không giải thích.\n\n"
        f"{numbered_texts}\n\n"
        "Ví dụ định dạng trả về: [0.9, 0.4, 0.7]"
    )
    try:
        raw = llm.invoke(prompt).content
        # Normalize: newer Gemini models may return a list of content parts
        if isinstance(raw, list):
            response = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in raw)
        else:
            response = str(raw)
        match = re.search(r"\[.*?\]", response, re.DOTALL)
        if match:
            scores = json.loads(match.group(0))
            if isinstance(scores, list) and len(scores) == len(candidates):
                return [max(0.0, min(1.0, float(s))) for s in scores]
    except Exception as e:
        print(f"Error in batch LLM reranking: {e}")

    # Fallback: neutral scores
    return [0.5] * len(candidates)


def rerank_candidates(query: str, candidates: List[Dict[str, Any]], top_n: int = None) -> List[Dict[str, Any]]:
    """Rerank candidates using LLM-based relevance scoring.

    CrossEncoder is skipped to avoid HuggingFace download overhead.
    Uses LLM batch scorer directly for fast, reliable reranking.
    """
    if not candidates:
        return []

    if top_n is None:
        top_n = SETTINGS.get("retrieval", {}).get("rerank_top_n", 3)

    print(f"Running LLM Reranker (input: {len(candidates)} candidates, top_n: {top_n})")

    try:
        scores = llm_rerank_fallback(query, candidates)
        scored_candidates = []
        for idx, score in enumerate(scores):
            item = candidates[idx].copy()
            item["rerank_score"] = float(score)
            item["score"] = float(score)
            scored_candidates.append(item)
    except Exception as e:
        print(f"LLM reranking failed: {e}. Retaining original ranks.")
        scored_candidates = [dict(c, rerank_score=c.get("score", 0.5)) for c in candidates]

    scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    final_results = scored_candidates[:top_n]
    print(f"Reranker complete. Selected top {len(final_results)} chunks.")
    return final_results

