"""Orchestrator module coordinating the Agentic Loop and other search strategies."""
import time
from typing import List, Dict, Any
from src.retrieval.dense import retrieve_dense
from src.retrieval.bm25 import retrieve_bm25
from src.retrieval.hybrid import retrieve_hybrid
from src.retrieval.reranker import rerank_candidates
from src.agents.rag_agent import classify_query, grade_retrieved_context, synthesize_legal_answer
from src.retrieval.graph import retrieve_graph_relations

# Global storage settings
from configs.setting import SETTINGS


def execute_rag_flow(query: str, strategy: str = "agentic") -> Dict[str, Any]:
    """Execute the Legal RAG pipeline using the specified strategy.

    Strategies:
    - 'dense': Dense Vector search only + Reranker + LLM Synthesis
    - 'bm25': Lexical search only + Reranker + LLM Synthesis
    - 'hybrid': Hybrid (RRF) search + Reranker + LLM Synthesis
    - 'agentic': Reason-Act-Observe Loop with Automatic Graph Self-Correction
    """
    start_time = time.time()
    logs = []
    retrieved_chunks = []
    relations = None
    
    logs.append(f"⏱️ Bắt đầu xử lý câu hỏi với chiến lược: '{strategy}'")
    
    # Extract config limits
    k = SETTINGS.get("retrieval", {}).get("k", 5)
    
    # -------------------------------------------------------------
    # CASE 1: Non-agentic Direct Search Strategies
    # -------------------------------------------------------------
    if strategy != "agentic":
        logs.append(f"🔍 Thực hiện tìm kiếm trực tiếp ({strategy.upper()})...")
        if strategy == "dense":
            chunks = retrieve_dense(query, k=k)
        elif strategy == "bm25":
            chunks = retrieve_bm25(query, k=k)
        else:
            # Default to hybrid
            chunks = retrieve_hybrid(query, k=k)
            
        logs.append(f"📥 Tìm thấy {len(chunks)} phân đoạn.")
        
        # Apply reranking
        logs.append("🎯 Tiến hành chấm điểm sâu bằng Reranker...")
        reranked_chunks = rerank_candidates(query, chunks, top_n=3)
        
        # LLM Synthesis
        logs.append("✍️ Đang tổng hợp câu trả lời chi tiết...")
        answer = synthesize_legal_answer(query, reranked_chunks)
        
        elapsed = time.time() - start_time
        return {
            "query": query,
            "strategy": strategy,
            "answer": answer,
            "retrieved_chunks": reranked_chunks,
            "relations": None,
            "logs": logs,
            "execution_time_seconds": round(elapsed, 2)
        }
        
    # -------------------------------------------------------------
    # CASE 2: Agentic Reason-Act-Observe Loop
    # -------------------------------------------------------------
    logs.append("🧠 [Vòng lặp 1 - REASON] Phân tích đặc trưng câu hỏi để chọn chiến lược tìm kiếm...")
    initial_strategy = classify_query(query)
    logs.append(f"👉 LLM đề xuất chiến lược tìm kiếm ban đầu: '{initial_strategy.upper()}'")
    
    # Round 1: ACT
    logs.append(f"⚡ [Vòng lặp 1 - ACT] Gọi công cụ tìm kiếm '{initial_strategy.upper()}' để thu thập cơ sở pháp lý...")
    if initial_strategy == "dense":
        initial_chunks = retrieve_dense(query, k=k)
    elif initial_strategy == "bm25":
        initial_chunks = retrieve_bm25(query, k=k)
    else:
        initial_chunks = retrieve_hybrid(query, k=k)
        
    logs.append(f"📥 Đã lấy được {len(initial_chunks)} phân đoạn pháp luật.")
    
    # Round 1: OBSERVE & GRADE
    logs.append("👁️ [Vòng lặp 1 - OBSERVE] Sub-agent đang kiểm tra tính đầy đủ và trạng thái hiệu lực của văn bản...")
    grader_decision, seed_docs = grade_retrieved_context(query, initial_chunks)
    
    logs.append(f"👉 Quyết định của Grader: '{grader_decision.upper()}' | Đề xuất văn bản hạt giống: {seed_docs}")
    
    # Condition: If expired/amended documents found or more detail is needed
    if grader_decision == "needs_graph" and seed_docs:
        logs.append("⚠️ [CẢNH BÁO] Phát hiện văn bản liên quan có thể bị sửa đổi, hết hiệu lực hoặc cần liên kết sâu rộng!")
        logs.append("🧠 [Vòng lặp 2 - REASON] Tác tử quyết định kích hoạt Đồ thị liên kết (Graph Tool) để tra cứu lịch sử sửa đổi...")
        
        # Round 2: ACT (Graph traversal)
        logs.append(f"🕸️ [Vòng lặp 2 - ACT] Duyệt đồ thị quan hệ qua {len(seed_docs)} văn bản hạt giống...")
        relations = retrieve_graph_relations(seed_doc_ids=seed_docs)
        
        descriptions = relations.get("descriptions", [])
        logs.append(f"👉 Đã phát hiện {len(relations.get('nodes', {}))} nút liên quan và {len(relations.get('edges', []))} cạnh kết nối.")
        for desc in descriptions:
            logs.append(f"   ℹ️ {desc}")
            
        # Round 2: OBSERVE (Merge context)
        logs.append("👁️ [Vòng lặp 2 - OBSERVE] Tích hợp thêm nội dung các luật sửa đổi/bổ sung vào bộ nhớ chung (State)...")
        # We preserve initial chunks but will let the final LLM know about the graph relations
        retrieved_chunks = initial_chunks
    else:
        logs.append("✅ [OBSERVE] Tài liệu thu thập đã đầy đủ, có hiệu lực hiện hành và không phát hiện sửa đổi. Không cần duyệt đồ thị.")
        retrieved_chunks = initial_chunks
        
    # Phase 3: RERANK
    logs.append("🎯 [RERANK] Tiến hành chuẩn hóa và chấm điểm sâu các ứng viên bằng CrossEncoder để đưa điều khoản đúng nhất lên đầu...")
    final_chunks = rerank_candidates(query, retrieved_chunks, top_n=3)
    
    # Phase 4: SYNTHESIS
    logs.append("✍️ [SYNTHESIS] Luật sư AI đang tổng hợp câu trả lời chính xác nhất kèm cảnh báo hiệu lực và trích dẫn nguồn...")
    answer = synthesize_legal_answer(query, final_chunks, relations)
    
    elapsed = time.time() - start_time
    logs.append(f"⏱️ Hoàn thành Agentic RAG Flow sau {round(elapsed, 2)} giây.")
    
    return {
        "query": query,
        "strategy": "agentic",
        "answer": answer,
        "retrieved_chunks": final_chunks,
        "relations": relations,
        "logs": logs,
        "execution_time_seconds": round(elapsed, 2)
    }
