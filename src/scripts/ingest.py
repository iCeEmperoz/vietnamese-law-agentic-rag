"""Orchestrator script for the Vietnamese Law Ingestion & Indexing Pipeline (Phase 1)."""
import sys
import os
import argparse
from pathlib import Path

# Add project root directory to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.ingestion.loader import load_documents
from src.ingestion.cleaner import clean_documents_parallel
from src.ingestion.chunker import chunk_documents
from src.indexing.chroma_store import add_chunks_to_vector_store
from src.indexing.bm25_index import build_bm25_index
from src.indexing.graph_index import build_graph_index


def run_ingestion(sample_size: int = None, data_path: str = None):
    """Run the end-to-end ingestion pipeline."""
    print("=" * 60)
    print("STARTING VIETNAMESE LAW AGENTIC RAG INGESTION PIPELINE")
    print("=" * 60)
    
    # 1. Load raw documents
    print("\n[Step 1/6] Loading legal documents...")
    docs = load_documents(data_path)
    
    if sample_size and sample_size < len(docs):
        docs = docs[:sample_size]
        print(f"Sampled down to {sample_size} documents as requested.")
    else:
        print(f"Loaded {len(docs)} documents.")
        
    # 2. Parallel cleaning (strips HTML and normalizes spaces)
    print("\n[Step 2/6] Cleaning text content (multiprocessing enabled)...")
    cleaned_docs = clean_documents_parallel(docs)
    print("Text cleaning completed.")
    
    # 3. Structural legal-based chunking
    print("\n[Step 3/6] Splitting documents into structural legal chunks...")
    chunks = chunk_documents(cleaned_docs)
    print(f"Generated {len(chunks)} structural chunks.")
    
    # 4. Dense Vector store indexing
    print("\n[Step 4/6] Creating Vector Index in ChromaDB (text-embedding-004)...")
    add_chunks_to_vector_store(chunks)
    
    # 5. Lexical BM25 index serialization
    print("\n[Step 5/6] Building Lexical BM25 keyword index...")
    build_bm25_index(chunks)
    
    # 6. Graph relationship mapping
    print("\n[Step 6/6] Establishing Document Relationship Graph (NetworkX)...")
    build_graph_index(docs)
    
    print("\n" + "=" * 60)
    print("INGESTION PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Indexed documents: {len(docs)}")
    print(f"Indexed chunks:    {len(chunks)}")
    print(f"Vector Database:   ChromaDB (local)")
    print(f"Lexical Index:     BM25 Okapi (.pkl)")
    print(f"Relation Graph:    NetworkX DiGraph (.json)")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vietnamese Law Ingestion Pipeline")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Limit the number of documents processed (e.g. --sample 1000)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Custom path to raw Vietnamese Law document files"
    )
    
    args = parser.parse_args()
    run_ingestion(sample_size=args.sample, data_path=args.data_path)
