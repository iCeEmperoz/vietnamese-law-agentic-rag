# Settings module for aio-agentic-rag
import os
from pathlib import Path
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = BASE_DIR / "configs" / "config.yaml"


def load_settings():
    if not CONFIG_FILE.exists():
        return {
            "app_name": "aio-agentic-rag",
            "version": "0.1.0",
            "storage": {"type": "local", "path": "data"},
            "model": {"name": "gemini-1.5-flash", "temperature": 0.0},
            "ingestion": {"chunk_size": 800, "chunk_overlap": 100},
            "retrieval": {"k": 5, "graph_max_hops": 2, "rrf_k": 60, "rerank_top_n": 3}
        }
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        # Fallback in case of errors
        return {
            "app_name": "aio-agentic-rag",
            "version": "0.1.0",
            "storage": {"type": "local", "path": "data"},
            "model": {"name": "gemini-1.5-flash", "temperature": 0.0},
            "ingestion": {"chunk_size": 800, "chunk_overlap": 100},
            "retrieval": {"k": 5, "graph_max_hops": 2, "rrf_k": 60, "rerank_top_n": 3}
        }


SETTINGS = load_settings()
