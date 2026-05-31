"""Legal reasoning and grading prompts module for the Agentic RAG Agent."""
import json
from typing import List, Dict, Any, Tuple
from src.llm import get_llm


def _extract_text(content) -> str:
    """Normalize LLM response content to a plain string.

    Newer Gemini models (3.x, 3.5-flash) may return .content as a list of
    typed dicts like [{'type': 'text', 'text': '...'}] instead of a plain string.
    This helper handles both cases safely.
    """
    if isinstance(content, str):
        return content
    # List of content parts (e.g. [{'type': 'text', 'text': 'Hello'}])
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text", ""))
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)


def classify_query(query: str) -> str:
    """Analyze the legal query to decide the initial retrieval strategy:

    - 'bm25': if query contains exact document codes, article numbers, decree numbers
    - 'dense': if query is a general concept/meaning search
    - 'hybrid': default general search
    """
    llm = get_llm(temperature=0.0)
    prompt = (
        "Bạn là bộ não điều phối của hệ thống trợ lý pháp luật Việt Nam.\n"
        f"Hãy phân tích câu hỏi người dùng: \"{query}\"\n\n"
        "Quyết định xem chiến lược tìm kiếm ban đầu nào là phù hợp nhất:\n"
        "- Trả về 'bm25' nếu câu hỏi chứa đích danh số hiệu văn bản pháp luật, điều luật cụ thể (ví dụ: 'Nghị định 15/2020', 'Điều 101', 'Luật An ninh mạng 2018').\n"
        "- Trả về 'dense' nếu câu hỏi mang tính khái niệm chung, tìm kiếm ý nghĩa (ví dụ: 'các hành vi bị cấm trên mạng là gì', 'trách nhiệm của doanh nghiệp nước ngoài').\n"
        "- Trả về 'hybrid' nếu câu hỏi kết hợp cả hai yếu tố hoặc không rõ ràng.\n\n"
        "CHỈ TRẢ VỀ một trong ba từ: 'bm25', 'dense', hoặc 'hybrid'. Không viết thêm bất cứ giải thích nào."
    )
    try:
        response = _extract_text(llm.invoke(prompt).content)
        decision = response.strip().lower()
        if decision in ["bm25", "dense", "hybrid"]:
            return decision
    except Exception as e:
        print(f"Error classifying query: {e}")
        
    return "hybrid"  # Safe default


def grade_retrieved_context(query: str, chunks: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    """Sub-agent Grader: Evaluates the retrieved chunks to see if they are sufficient

    to answer the query, and checks if any document is expired, amended, or has related documents in the graph.
    
    Returns:
    - decision: 'sufficient' (done) or 'needs_graph' (requires graph expansion)
    - seed_doc_ids: list of document IDs to traverse in the graph
    """
    if not chunks:
        return "needs_graph", []
        
    llm = get_llm(temperature=0.0)
    
    # Format chunks for the LLM to inspect
    formatted_chunks = []
    for idx, c in enumerate(chunks):
        status_str = c["metadata"].get("status", "effective")
        formatted_chunks.append(
            f"[{idx+1}] ID văn bản: {c['doc_id']}\n"
            f"Tiêu đề: {c['metadata'].get('title')}\n"
            f"Trạng thái hiệu lực: {status_str}\n"
            f"Nội dung: {c['text']}\n"
        )
    chunks_str = "\n".join(formatted_chunks)
    
    prompt = (
        "Bạn là Trợ lý Giám sát Pháp luật (Grader Sub-Agent).\n"
        f"Câu hỏi của người dùng: \"{query}\"\n\n"
        f"Dưới đây là các tài liệu vừa thu thập được:\n{chunks_str}\n\n"
        "Nhiệm vụ của bạn là phân tích và chấm điểm độ tin cậy của tài liệu:\n"
        "1. Các tài liệu thu thập được có đủ thông tin chi tiết và tin cậy để trả lời câu hỏi chưa?\n"
        "2. Có phát hiện tài liệu nào có trạng thái hiệu lực bị SỬA ĐỔI ('amended') hoặc HẾT HIỆU LỰC ('expired') không?\n"
        "3. Nếu tài liệu bị sửa đổi/hết hiệu lực, hoặc thiếu văn bản hướng dẫn chi tiết, chúng ta có cần gọi công cụ Đồ thị (Graph Tool) "
        "để tìm các văn bản sửa đổi bổ sung/thay thế không?\n\n"
        "Hãy phản hồi dưới dạng JSON hợp lệ với cấu trúc sau:\n"
        "{\n"
        "  \"decision\": \"sufficient\" hoặc \"needs_graph\",\n"
        "  \"reason\": \"Giải thích ngắn gọn lý do tại đây\",\n"
        "  \"seed_docs\": [\"danh_sách_id_văn_bản_cần_tra_cứu_đồ_thị_nếu_có\"]\n"
        "}\n\n"
        "Lưu ý chỉ trả về duy nhất chuỗi JSON, không viết lời mở đầu hay kết thúc."
    )
    
    try:
        response = _extract_text(llm.invoke(prompt).content)
        # Clean JSON markdown if present
        json_str = response.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]
        json_str = json_str.strip()
        
        data = json.loads(json_str)
        decision = data.get("decision", "sufficient")
        seed_docs = data.get("seed_docs", [])
        
        # If the JSON returned some seeds but didn't choose 'needs_graph', override it
        if seed_docs and decision == "sufficient":
            # If any retrieved document status is 'amended' or 'expired', we should query the graph
            for c in chunks:
                if c["metadata"].get("status") in ["amended", "expired"] and c["doc_id"] in seed_docs:
                    decision = "needs_graph"
                    break
                    
        return decision, seed_docs
    except Exception as e:
        print(f"Error in context grading: {e}. Defaulting to hybrid/graph merge.")
        # Fail-safe: extract doc_ids from chunks to check graph
        doc_ids = list(set([c["doc_id"] for c in chunks]))
        return "needs_graph", doc_ids


def synthesize_legal_answer(query: str, chunks: List[Dict[str, Any]], relations: Dict[str, Any] = None) -> str:
    """Synthesize the final legal answer with citations and status markers using Gemini 1.5 Flash."""
    llm = get_llm(temperature=0.0)
    
    # 1. Format core chunks
    formatted_chunks = []
    for idx, c in enumerate(chunks):
        formatted_chunks.append(
            f"Nguồn [{idx+1}]: {c['metadata'].get('title')} ({c['doc_id']})\n"
            f"Trạng thái hiệu lực: {c['metadata'].get('status', 'effective')}\n"
            f"Nội dung quy định:\n{c['text']}\n"
        )
    chunks_context = "\n".join(formatted_chunks)
    
    # 2. Format graph relations if available
    graph_context = ""
    if relations and relations.get("nodes"):
        desc_list = relations.get("descriptions", [])
        content_dict = relations.get("contents", {})
        
        desc_str = "\n".join(desc_list)
        
        content_list = []
        for nid, ncontent in content_dict.items():
            node_meta = relations["nodes"].get(nid, {})
            content_list.append(
                f"Văn bản liên quan qua Đồ thị: {node_meta.get('title')} ({nid})\n"
                f"Trạng thái: {node_meta.get('status')}\n"
                f"Nội dung tóm tắt:\n{ncontent[:1000]}...\n"
            )
        contents_str = "\n".join(content_list)
        
        graph_context = (
            "\n=== CÁC MỐI QUAN HỆ VÀ VĂN BẢN ĐƯỢC DUYỆT TỪ ĐỒ THỊ LIÊN KẾT LUẬT ===\n"
            f"{desc_str}\n\n"
            "Nội dung của văn bản liên quan được bổ sung:\n"
            f"{contents_str}\n"
        )
        
    prompt = (
        "Bạn là Luật sư Trợ lý Pháp luật Việt Nam thông thái và chính xác.\n"
        f"Hãy trả lời câu hỏi sau của người dùng: \"{query}\"\n\n"
        "Dưới đây là các tài liệu cơ sở pháp lý thu thập được:\n"
        f"{chunks_context}\n"
        f"{graph_context}\n\n"
        "HƯỚNG DẪN BIÊN SOẠN CÂU TRẢ LỜI:\n"
        "1. Câu trả lời phải cực kỳ chính xác về mặt luật pháp, trình bày mạch lạc, trang nghiêm và rõ ràng.\n"
        "2. ĐẶC BIỆT CHÚ Ý về tình trạng hiệu lực: \n"
        "   - Nếu một điều khoản thuộc văn bản bị SỬA ĐỔI hoặc HẾT HIỆU LỰC, bạn phải nêu rõ quy định mới nhất thay thế hoặc sửa đổi nó "
        "(ví dụ: 'Quy định gốc tại Nghị định 15/2020 đã bị sửa đổi/bổ sung bởi Nghị định 14/2022 như sau...').\n"
        "   - Tuyệt đối không tư vấn áp dụng các quy định đã hết hiệu lực mà không cảnh báo.\n"
        "3. Sử dụng các ký hiệu biểu tượng (emoji) để người dùng dễ quan sát:\n"
        "   - 🟢 cho văn bản đang có hiệu lực (effective)\n"
        "   - 🟡 cho văn bản bị sửa đổi/bổ sung một phần (amended)\n"
        "   - 🔴 cho văn bản hết hiệu lực hoặc bị thay thế (expired)\n"
        "4. TRÌNH BÀY PHÁP LUẬT ĐỂ DỄ DÀNG ĐỌC (TRÁNH BỨC TƯỜNG CHỮ - WALL OF TEXT):\n"
        "   - Tuyệt đối không viết các đoạn văn dài dằng dặc. Chia nhỏ thông tin thành nhiều đoạn ngắn.\n"
        "   - Khi so sánh hoặc liệt kê các mức phạt cho nhiều đối tượng khác nhau (ví dụ: Ô tô và Xe máy, Cá nhân và Tổ chức), "
        "bắt buộc sử dụng Định dạng Bảng (Markdown Table) hoặc Danh sách gạch đầu dòng có cấu trúc thụt lề rõ ràng.\n"
        "   - Hãy BÔI ĐẬM các thông tin quan trọng nhất để người dùng dễ quét mắt: số tiền phạt (ví dụ: `**phạt tiền từ 6.000.000 đồng đến 8.000.000 đồng**`), "
        "hình phạt bổ sung (ví dụ: `**tước quyền sử dụng Giấy phép lái xe từ 22 đến 24 tháng**`), và hành vi vi phạm cụ thể.\n"
        "5. ĐỊNH DẠNG CĂN CỨ PHÁP LÝ BẰNG INLINE CODE:\n"
        "   - Đối với tất cả các dẫn chiếu đến Điều, Khoản, Điểm hoặc số hiệu văn bản cụ thể (ví dụ: `Khoản 3 Điều 5`, `Nghị định 100/2019/NĐ-CP`), "
        "hãy bao quanh chúng bằng dấu backtick (ví dụ: `Khoản 3 Điều 5`, `Nghị định 100/2019/NĐ-CP`) để hệ thống làm nổi bật bằng màu sắc trực quan.\n"
        "6. Cuối câu trả lời, hãy tạo một mục riêng mang tên '📚 Nguồn dẫn chiếu chi tiết' để liệt kê rõ ràng:\n"
        "   - Tên văn bản chính thức, số hiệu, ngày ban hành và tình trạng hiệu lực cụ thể.\n"
        "   - Các mối quan hệ liên kết luật đã duyệt qua (ví dụ: 'Nghị định 53/2022/NĐ-CP hướng dẫn chi tiết cho Luật An ninh mạng 2018').\n\n"
        "Hãy viết câu trả lời hoàn toàn bằng tiếng Việt, chuyên nghiệp và có chiều sâu."
    )
    
    try:
        return _extract_text(llm.invoke(prompt).content)
    except Exception as e:
        return f"Lỗi tổng hợp câu trả lời từ LLM: {e}. Vui lòng kiểm tra API key."
