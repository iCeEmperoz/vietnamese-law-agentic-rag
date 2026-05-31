"""Graph-based retrieval module using relationship traversal to find related laws."""
import os
import json
from typing import List, Dict, Any
from configs.setting import SETTINGS
from src.indexing.graph_index import get_document_relations

# Path to persisted graph JSON (written by ingest pipeline)
_GRAPH_JSON_PATH = os.path.join("data", "graph", "graph_index.json")

# Module-level node metadata cache: loaded from graph JSON once
_node_meta_cache: dict | None = None


def _get_node_meta_cache() -> dict:
    """Load node metadata from the persisted graph JSON (once per process)."""
    global _node_meta_cache
    if _node_meta_cache is None:
        if os.path.exists(_GRAPH_JSON_PATH):
            with open(_GRAPH_JSON_PATH, "r", encoding="utf-8") as f:
                graph_data = json.load(f)
            _node_meta_cache = graph_data.get("nodes", {})
        else:
            _node_meta_cache = {}
    return _node_meta_cache


def get_doc_by_id(doc_id: str) -> Dict[str, Any]:
    """Retrieve document metadata by ID from the cached graph node index."""
    return _get_node_meta_cache().get(doc_id)



def retrieve_graph_relations(seed_doc_ids: List[str], max_hops: int = None) -> Dict[str, Any]:
    """Traverse relationships starting from seed doc IDs and pull in related documents.

    Returns structured related document metadata and relationship context strings.
    """
    if max_hops is None:
        max_hops = SETTINGS.get("retrieval", {}).get("graph_max_hops", 2)
        
    print(f"🕸️ Running Graph Relation Retrieval for seeds: {seed_doc_ids} (max_hops={max_hops})")
    
    all_related_nodes = {}
    all_related_edges = []
    
    for doc_id in seed_doc_ids:
        relations = get_document_relations(doc_id, max_hops=max_hops)
        # Merge nodes
        for nid, nmeta in relations.get("nodes", {}).items():
            if nid not in all_related_nodes:
                all_related_nodes[nid] = nmeta
        # Merge edges
        for edge in relations.get("edges", []):
            if edge not in all_related_edges:
                all_related_edges.append(edge)
                
    # Format relations into a readable context block for the LLM
    relationship_context = []
    fetched_contents = {}
    
    # 1. Format nodes information
    for nid, node_meta in all_related_nodes.items():
        # Load full raw content of related document if not in seeds
        if nid not in seed_doc_ids:
            doc_data = get_doc_by_id(nid)
            if doc_data:
                fetched_contents[nid] = doc_data.get("raw_content", "")
                
    # 2. Format edges (relations) to explain how documents link
    relationship_descriptions = []
    for edge in all_related_edges:
        source_title = all_related_nodes.get(edge["source"], {}).get("title", edge["source"])
        target_title = all_related_nodes.get(edge["target"], {}).get("title", edge["target"])
        relation_type = edge["relation"]
        
        if relation_type == "amends":
            desc = f"- Văn bản '{source_title}' ({edge['source']}) SỬA ĐỔI, BỔ SUNG cho văn bản '{target_title}' ({edge['target']})."
        elif relation_type == "replaces":
            desc = f"- Văn bản '{source_title}' ({edge['source']}) THAY THẾ HOÀN TOÀN cho văn bản '{target_title}' ({edge['target']})."
        else:
            desc = f"- Văn bản '{source_title}' ({edge['source']}) LIÊN KẾT/THAM CHIẾU tới văn bản '{target_title}' ({edge['target']})."
        
        relationship_descriptions.append(desc)
        
    return {
        "nodes": all_related_nodes,
        "edges": all_related_edges,
        "descriptions": relationship_descriptions,
        "contents": fetched_contents
    }
