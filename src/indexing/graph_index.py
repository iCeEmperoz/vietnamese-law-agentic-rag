"""Graph index module using NetworkX to map and traverse legal document relationships."""
import os
import json
import networkx as nx
from typing import List, Dict, Any

# Default storage paths
GRAPH_DIR = os.path.join("data", "graph")
GRAPH_PATH = os.path.join(GRAPH_DIR, "graph_index.json")


def build_graph_index(documents: List[Dict[str, Any]], persist_path: str = GRAPH_PATH) -> nx.DiGraph:
    """Build a directed graph of relationships (amends, replaces) using NetworkX and persist as JSON."""
    os.makedirs(os.path.dirname(persist_path), exist_ok=True)
    
    # Directed Graph: Edges flow from newer documents pointing to target documents
    G = nx.DiGraph()
    
    # 1. Add all documents as nodes with metadata
    for doc in documents:
        G.add_node(
            doc["doc_id"],
            title=doc["title"],
            doc_type=doc["doc_type"],
            issue_date=doc["issue_date"],
            effective_date=doc["effective_date"],
            status=doc["status"]
        )
        
    # 2. Add edges representing legal linkages
    for doc in documents:
        source_id = doc["doc_id"]
        
        # Connect to documents that this document AMENDS
        for target_id in doc.get("amends", []):
            if G.has_node(target_id):
                G.add_edge(source_id, target_id, relation="amends")
                
        # Connect to documents that this document REPLACES
        for target_id in doc.get("replaces", []):
            if G.has_node(target_id):
                G.add_edge(source_id, target_id, relation="replaces")
                
        # Handle reverse relationships if explicitly stated in standard loader
        for target_id in doc.get("amended_by", []):
            if G.has_node(target_id):
                G.add_edge(target_id, source_id, relation="amends")
                
        for target_id in doc.get("replaced_by", []):
            if G.has_node(target_id):
                G.add_edge(target_id, source_id, relation="replaces")
                
    # Serialize directed graph using node-link format
    graph_data = nx.node_link_data(G)
    
    with open(persist_path, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)
        
    print(f"Successfully built relationship graph index with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges at {persist_path}.")
    return G


def load_graph_index(persist_path: str = GRAPH_PATH) -> nx.DiGraph:
    """Load and reconstruct the NetworkX directed graph index from the JSON file."""
    if not os.path.exists(persist_path):
        print(f"Warning: Graph index not found at {persist_path}. Creating a blank graph.")
        return nx.DiGraph()
        
    with open(persist_path, "r", encoding="utf-8") as f:
        graph_data = json.load(f)
        
    return nx.node_link_graph(graph_data)


def get_document_relations(doc_id: str, max_hops: int = 2, persist_path: str = GRAPH_PATH) -> Dict[str, Any]:
    """Traverse the graph in both directions (in-edges and out-edges) up to max_hops

    starting from a specific seed document, and return all related node metadata and edges.
    """
    G = load_graph_index(persist_path)
    if not G.has_node(doc_id):
        return {"nodes": {}, "edges": []}
        
    visited_nodes = {doc_id}
    queue = [(doc_id, 0)]
    edges = []
    
    # BFS traversal to find all connected nodes within max_hops (undirected traversal of a directed graph)
    while queue:
        current_node, depth = queue.pop(0)
        if depth >= max_hops:
            continue
            
        # 1. Check out-edges (newer/amending documents that this document points to)
        for neighbor in G.successors(current_node):
            relation_data = G.get_edge_data(current_node, neighbor)
            edges.append({
                "source": current_node,
                "target": neighbor,
                "relation": relation_data.get("relation", "references")
            })
            if neighbor not in visited_nodes:
                visited_nodes.add(neighbor)
                queue.append((neighbor, depth + 1))
                
        # 2. Check in-edges (older/target documents that point to this document)
        for neighbor in G.predecessors(current_node):
            relation_data = G.get_edge_data(neighbor, current_node)
            edges.append({
                "source": neighbor,
                "target": current_node,
                "relation": relation_data.get("relation", "references")
            })
            if neighbor not in visited_nodes:
                visited_nodes.add(neighbor)
                queue.append((neighbor, depth + 1))
                
    # Format and collect node metadata for all visited nodes
    nodes_metadata = {}
    for node in visited_nodes:
        if G.has_node(node):
            nodes_metadata[node] = dict(G.nodes[node])
            nodes_metadata[node]["doc_id"] = node
            
    # Deduplicate edges
    unique_edges = []
    seen_edges = set()
    for edge in edges:
        edge_key = (edge["source"], edge["target"], edge["relation"])
        if edge_key not in seen_edges:
            seen_edges.add(edge_key)
            unique_edges.append(edge)
            
    return {
        "nodes": nodes_metadata,
        "edges": unique_edges
    }
