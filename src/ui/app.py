"""Modern chatbot-style Streamlit UI for the Vietnamese Law Agentic RAG System."""
import streamlit as st
import streamlit.components.v1 as components
import os

# ─────────────────────────────────────────────
# Pre-load heavy resources once at startup using Streamlit cache.
# Without this, ChromaDB + BM25 + LLM are re-initialized on every user interaction.
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="🔄 Đang khởi tạo Vector DB...")
def _load_chroma_store():
    from src.indexing.chroma_store import get_chroma_store
    return get_chroma_store()

@st.cache_resource(show_spinner="🔄 Đang nạp chỉ mục BM25...")
def _load_bm25_index():
    # Pre-warm the BM25 cache by loading the index file into memory
    from src.indexing.bm25_index import query_bm25
    query_bm25("khởi động")  # triggers disk load and caches in _bm25_cache
    return True

@st.cache_resource(show_spinner="🔄 Đang kết nối LLM Gemini...")
def _load_llm():
    from src.llm import get_llm
    return get_llm()

# Trigger pre-loading immediately on startup (results are cached after first run)
_load_chroma_store()
_load_bm25_index()
_load_llm()

from src.agents.orchestrator import execute_rag_flow


# ─────────────────────────────────────────────
# 1. Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Trợ Lý Pháp Luật Việt Nam",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# 2. Global CSS – premium dark chat theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Be+Vietnam+Pro:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', 'Be Vietnam Pro', sans-serif;
    background-color: #0F1117;
    color: #E5E7EB;
}

/* ── Hide Streamlit default chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 0rem !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b27 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.07);
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: #D1D5DB !important;
}

/* ── Chat container ── */
.chat-wrapper {
    max-width: 860px;
    margin: 0 auto;
}

/* ── User bubble ── */
.bubble-user {
    display: flex;
    justify-content: flex-end;
    margin: 0.75rem 0;
}
.bubble-user .bubble-content {
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
    color: #fff;
    padding: 0.85rem 1.2rem;
    border-radius: 18px 18px 4px 18px;
    max-width: 72%;
    font-size: 0.97rem;
    line-height: 1.6;
    box-shadow: 0 4px 20px rgba(37, 99, 235, 0.35);
}

/* ── Assistant bubble ── */
.bubble-ai {
    display: flex;
    justify-content: flex-start;
    margin: 0.75rem 0;
    gap: 0.7rem;
    align-items: flex-start;
}
.bubble-ai .avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: linear-gradient(135deg, #10B981, #059669);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
    box-shadow: 0 2px 8px rgba(16,185,129,0.4);
}
.bubble-ai .bubble-content {
    background: rgba(31, 41, 55, 0.75);
    border: 1px solid rgba(255,255,255,0.08);
    color: #E5E7EB;
    padding: 0.9rem 1.2rem;
    border-radius: 4px 18px 18px 18px;
    max-width: 80%;
    font-size: 0.97rem;
    line-height: 1.7;
    backdrop-filter: blur(8px);
}
/* Style legal citations/inline code inside the AI bubble to pop out with an elegant amber glow */
.bubble-ai .bubble-content code {
    color: #FBBF24 !important;
    background-color: rgba(245, 158, 11, 0.08) !important;
    border: 1px solid rgba(245, 158, 11, 0.25) !important;
    padding: 0.15rem 0.4rem !important;
    border-radius: 6px !important;
    font-family: 'Courier New', monospace !important;
    font-size: 0.9em !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.15) !important;
}


/* ── Citation cards ── */
.cite-card {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    background: rgba(17, 24, 39, 0.8);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.85rem;
}
.cite-card .doc-title { color: #D1D5DB; font-weight: 500; }
.cite-card .doc-id   { color: #6B7280; font-size: 0.78rem; margin-top: 2px; }

.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    white-space: nowrap;
    flex-shrink: 0;
}
.badge-green  { background: rgba(16,185,129,0.15); color: #34D399; border: 1px solid rgba(16,185,129,0.3); }
.badge-yellow { background: rgba(245,158,11,0.15);  color: #FBBF24; border: 1px solid rgba(245,158,11,0.3); }
.badge-red    { background: rgba(239, 68,68,0.15);  color: #F87171; border: 1px solid rgba(239,68,68,0.3); }

/* ── Section divider ── */
.section-title {
    font-size: 0.78rem;
    font-weight: 600;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 1.2rem 0 0.5rem;
}

/* ── Log lines ── */
.log-line {
    font-family: 'Courier New', monospace;
    font-size: 0.82rem;
    color: #9CA3AF;
    padding: 0.3rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.03);
}
.log-line:last-child { border-bottom: none; }

/* ── Chat input override ── */
[data-testid="stChatInput"] {
    border-radius: 14px !important;
    background-color: transparent !important;
}
[data-testid="stChatInput"] textarea {
    background-color: rgba(17, 24, 39, 0.85) !important;
    border: 1px solid rgba(255, 255, 255, 0.16) !important;
    border-radius: 12px !important;
    color: #F3F4F6 !important;
    font-size: 0.98rem !important;
    padding: 0.75rem 1rem !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: rgba(37, 99, 235, 0.85) !important;
    box-shadow: 0 0 15px rgba(37, 99, 235, 0.3) !important;
    background-color: rgba(17, 24, 39, 0.95) !important;
}
/* Send button styling */
[data-testid="stChatInput"] button {
    background-color: rgba(37, 99, 235, 0.2) !important;
    border: 1px solid rgba(37, 99, 235, 0.4) !important;
    color: #3B82F6 !important;
    transition: all 0.2s ease !important;
}
[data-testid="stChatInput"] button:hover {
    background-color: rgba(37, 99, 235, 0.8) !important;
    color: #ffffff !important;
    box-shadow: 0 0 10px rgba(37, 99, 235, 0.4) !important;
}


/* Welcome card */
.welcome-card {
    text-align: center;
    padding: 3rem 1rem 2rem;
    opacity: 0.85;
}
.welcome-icon {
    font-size: 3.5rem;
    margin-bottom: 0.8rem;
    display: block;
}
.welcome-title {
    font-family: 'Be Vietnam Pro', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #F3F4F6;
    margin-bottom: 0.5rem;
}
.welcome-sub {
    font-size: 1rem;
    color: #9CA3AF;
    line-height: 1.6;
    max-width: 520px;
    margin: 0 auto 2rem;
}
.example-chip {
    display: inline-block;
    background: rgba(31,41,55,0.8);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 999px;
    padding: 0.45rem 1rem;
    font-size: 0.85rem;
    color: #D1D5DB;
    margin: 0.25rem;
    cursor: pointer;
}

/* Scrollable messages area */
.messages-area {
    min-height: 200px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 3. Session state
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []   # list of {role, content, logs, relations, citations}

# ─────────────────────────────────────────────
# 4. Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## ⚖️ Trợ Lý Pháp Luật")
    st.markdown("<p style='color:#6B7280; font-size:0.82rem;'>Hệ thống Agentic RAG 4 tầng</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("#### 🧭 Chiến lược tìm kiếm")
    strategy = st.selectbox(
        "Chiến lược",
        options=["agentic", "hybrid", "dense", "bm25"],
        format_func=lambda x: {
            "agentic": "🧠 Agentic Loop",
            "hybrid":  "🔀 Hybrid (Dense + BM25)",
            "dense":   "🔍 Dense (Vector DB)",
            "bm25":    "📝 Lexical (BM25)"
        }[x],
        label_visibility="collapsed"
    )
    # Persist so it's available during pending_query processing after st.rerun()
    st.session_state.strategy = strategy

    st.markdown("""
<div style='margin-top:0.5rem; font-size:0.8rem; color:#6B7280; line-height:1.5;'>
    <b style='color:#9CA3AF;'>Agentic Loop</b> tự động phát hiện và cảnh báo các văn bản luật bị sửa đổi hoặc hết hiệu lực thông qua đồ thị NetworkX.
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📦 Trạng thái hệ thống")

    # Check index files exist
    chroma_ok = os.path.exists(os.path.join("data", "chroma"))
    bm25_ok   = os.path.exists(os.path.join("data", "bm25", "bm25_index.pkl"))
    graph_ok  = os.path.exists(os.path.join("data", "graph", "graph_index.json"))

    def status_row(label, ok):
        icon  = "🟢" if ok else "🔴"
        color = "#34D399" if ok else "#F87171"
        note  = "Sẵn sàng" if ok else "Chưa nạp"
        st.markdown(
            f"<div style='display:flex; justify-content:space-between; font-size:0.83rem; margin:0.25rem 0;'>"
            f"<span style='color:#D1D5DB;'>{icon} {label}</span>"
            f"<span style='color:{color};'>{note}</span></div>",
            unsafe_allow_html=True
        )

    status_row("Vector DB", chroma_ok)
    status_row("BM25 Index", bm25_ok)
    status_row("Graph Index", graph_ok)

    st.markdown("---")
    if st.button("🗑️ Xóa lịch sử trò chuyện", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.75rem; color:#374151; text-align:center;'>Agentic RAG · Gemini 2.0 Flash Lite</div>",
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────
# 5. Helper: render one citation card
# ─────────────────────────────────────────────
def render_citation(cite: dict):
    status = cite.get("metadata", {}).get("status", "effective")
    title  = cite.get("metadata", {}).get("title", "Không rõ tên văn bản")
    doc_id = cite.get("doc_id", "")

    if status == "effective":
        badge = "<span class='badge badge-green'>🟢 Còn hiệu lực</span>"
    elif status == "amended":
        badge = "<span class='badge badge-yellow'>🟡 Bị sửa đổi</span>"
    else:
        badge = "<span class='badge badge-red'>🔴 Hết hiệu lực</span>"

    st.markdown(f"""
    <div class='cite-card'>
        {badge}
        <div>
            <div class='doc-title'>{title}</div>
            <div class='doc-id'>{doc_id}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 6. Helper: render one full assistant message
# ─────────────────────────────────────────────
def render_assistant_message(msg: dict):
    with st.chat_message("assistant", avatar="⚖️"):
        st.markdown(msg["content"])

        if msg.get("logs"):
            with st.expander("⚙️ Nhật ký tác tử (Agent Loop Logs)", expanded=False):
                for line in msg["logs"]:
                    st.markdown(f"<div class='log-line'>{line}</div>", unsafe_allow_html=True)

        if msg.get("relations") and msg["relations"].get("descriptions"):
            with st.expander("🕸️ Liên kết văn bản phát hiện qua đồ thị", expanded=False):
                for desc in msg["relations"]["descriptions"]:
                    st.info(desc)

        if msg.get("citations"):
            st.markdown("<div class='section-title'>📚 Nguồn dẫn chiếu</div>", unsafe_allow_html=True)
            for cite in msg["citations"]:
                render_citation(cite)

# ─────────────────────────────────────────────
# Welcome screen when no messages
if not st.session_state.messages and not st.session_state.get("pending_query"):
    st.markdown("""
    <div class='welcome-card'>
        <span class='welcome-icon'>⚖️</span>
        <div class='welcome-title'>Trợ Lý Pháp Luật Việt Nam</div>
        <div class='welcome-sub'>
            Đặt câu hỏi về luật pháp Việt Nam. Tôi sẽ tìm kiếm và trả lời dựa trên các văn bản pháp quy,
            đồng thời tự động phát hiện và cảnh báo các điều khoản đã hết hiệu lực hoặc bị sửa đổi.
        </div>
        <div>
            <span class='example-chip'>🚗 Nồng độ cồn khi lái xe bị phạt thế nào?</span>
            <span class='example-chip'>📱 Đăng tin giả lên mạng bị xử phạt ra sao?</span>
            <span class='example-chip'>🏠 Quy định về chuyển nhượng đất đai?</span>
            <span class='example-chip'>💼 Quyền lợi của người lao động khi bị sa thải?</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Render full conversation history
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        render_assistant_message(msg)




# ─────────────────────────────────────────────
# 8. Processing: handle pending query (step 2 of rerun pattern)
# ─────────────────────────────────────────────
if st.session_state.get("pending_query"):
    query = st.session_state.pending_query

    # Show the "thinking" indicator as a temporary assistant bubble
    with st.chat_message("assistant", avatar="⚖️"):
        with st.spinner("⏳ Đang tra cứu cơ sở pháp lý và kiểm tra hiệu lực văn bản..."):
            try:
                result    = execute_rag_flow(query, strategy=st.session_state.get("strategy", "agentic"))
                answer    = result["answer"]
                logs      = result.get("logs", [])
                relations = result.get("relations")
                chunks    = result.get("retrieved_chunks", [])

                # Save result
                st.session_state.messages.append({
                    "role":      "assistant",
                    "content":   answer,
                    "logs":      logs,
                    "relations": relations,
                    "citations": chunks[:3]
                })

            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ Lỗi hệ thống: {str(e)}",
                    "logs": [], "relations": None, "citations": []
                })
            finally:
                # Always clear the pending flag regardless of success/failure
                st.session_state.pending_query = None

    # Clean rerun: sidebar stays, page renders from top with new message in history
    st.rerun()

# ─────────────────────────────────────────────
# 9. Chat input (always at the bottom, always visible)
# ─────────────────────────────────────────────
if user_query := st.chat_input("Nhập câu hỏi pháp lý của bạn..."):
    # Step 1: Save user message + pending flag, rerun immediately
    # This shows the user message in history before processing starts
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.session_state.pending_query = user_query
    st.rerun()

