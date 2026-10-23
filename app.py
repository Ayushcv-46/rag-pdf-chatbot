import os
import logging
import streamlit as st
from dotenv import load_dotenv

from core_rag import (
    RAGIndexManager,
    configure_settings,
    build_fusion_query_engine,
    compute_bytes_hash,
    DEFAULT_DATA_DIR
)

# ======================================================
# LOGGING CONFIGURATION
# ======================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("rag_pdf_chatbot")

# ======================================================
# LOAD ENV VARIABLES & EARLY KEY CHECK
# ======================================================
load_dotenv()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

# ======================================================
# STREAMLIT PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="DocuMind — RAG PDF Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# GLOBAL CSS — Dark Premium Design
# ======================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root & Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #0a0d14 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stAppViewContainer"] > .main {
    background: #0a0d14 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1420 0%, #0c1018 100%) !important;
    border-right: 1px solid rgba(99, 102, 241, 0.15) !important;
}
[data-testid="stSidebar"] * { font-family: 'Inter', sans-serif !important; }

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stDecoration"] { display: none !important; }

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #1a1f35 0%, #111827 50%, #0f172a 100%);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 20px;
    padding: 36px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; left: -60px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(99,102,241,0.18) 0%, transparent 70%);
    pointer-events: none;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(168,85,247,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a5b4fc 0%, #c084fc 50%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 6px 0;
    line-height: 1.2;
}
.hero-subtitle {
    color: #94a3b8;
    font-size: 0.95rem;
    font-weight: 400;
    margin: 0;
    line-height: 1.5;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.3);
    color: #a5b4fc;
    font-size: 0.75rem;
    font-weight: 500;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 16px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ── Sidebar Header ── */
.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 0 20px;
    border-bottom: 1px solid rgba(99,102,241,0.15);
    margin-bottom: 20px;
}
.sidebar-logo-text {
    font-size: 1.1rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a5b4fc, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.sidebar-section-title {
    font-size: 0.7rem;
    font-weight: 600;
    color: #6366f1;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 20px 0 10px;
}

/* ── Pill Chips ── */
.pipeline-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.2);
    color: #c7d2fe;
    font-size: 0.78rem;
    padding: 5px 12px;
    border-radius: 6px;
    margin: 3px 0;
    width: 100%;
}
.chip-icon { color: #818cf8; }

/* ── File cards in sidebar ── */
.file-card {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(99,102,241,0.07);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 10px;
    padding: 8px 12px;
    margin: 4px 0;
}
.file-card-icon { font-size: 1rem; }
.file-card-name {
    color: #e2e8f0;
    font-size: 0.8rem;
    font-weight: 500;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    flex-grow: 1;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: rgba(99,102,241,0.04) !important;
    border: 1.5px dashed rgba(99,102,241,0.3) !important;
    border-radius: 14px !important;
    padding: 8px !important;
    transition: border-color 0.2s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(99,102,241,0.6) !important;
    background: rgba(99,102,241,0.08) !important;
}
[data-testid="stFileUploader"] label {
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: #131b2e !important;
    border: 1.5px solid rgba(99,102,241,0.3) !important;
    border-radius: 16px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    transition: border-color 0.2s ease !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(99,102,241,0.7) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
}
[data-testid="stChatInput"] textarea {
    color: #e2e8f0 !important;
    background: transparent !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #475569 !important; }

/* ── Citation cards ── */
.citation-card {
    background: linear-gradient(135deg, #0f172a 0%, #131d33 100%);
    border: 1px solid rgba(99,102,241,0.2);
    border-left: 3px solid #6366f1;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 10px 0;
    font-size: 0.85rem;
    color: #94a3b8;
    line-height: 1.6;
}
.citation-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
    flex-wrap: wrap;
}
.citation-badge {
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.3);
    color: #a5b4fc;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.citation-filename {
    background: rgba(15,23,42,0.8);
    border: 1px solid rgba(99,102,241,0.2);
    color: #818cf8;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    padding: 2px 10px;
    border-radius: 6px;
}
.citation-score {
    margin-left: auto;
    color: #475569;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
}
.citation-text {
    color: #94a3b8;
    font-size: 0.83rem;
    line-height: 1.65;
    margin: 0;
}

/* ── Section dividers ── */
.section-divider {
    border: none;
    border-top: 1px solid rgba(99,102,241,0.12);
    margin: 20px 0;
}

/* ── Empty state ── */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 60px 40px;
    text-align: center;
    gap: 12px;
}
.empty-icon { font-size: 3.5rem; margin-bottom: 8px; }
.empty-title { color: #e2e8f0; font-size: 1.1rem; font-weight: 600; margin: 0; }
.empty-sub   { color: #475569; font-size: 0.88rem; margin: 0; line-height: 1.5; }

/* ── Status dot ── */
.status-dot {
    width: 8px; height: 8px;
    background: #22c55e;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
    box-shadow: 0 0 6px rgba(34,197,94,0.5);
    animation: blink 2s infinite;
}
.status-offline {
    background: #ef4444 !important;
    box-shadow: 0 0 6px rgba(239,68,68,0.5) !important;
    animation: none !important;
}
@keyframes blink {
    0%,100% { opacity: 1; }
    50%      { opacity: 0.45; }
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# API KEY VALIDATION
# ======================================================
if not NVIDIA_API_KEY:
    st.error("⚠️ `NVIDIA_API_KEY` not found in environment or `.env` file! Please configure `NVIDIA_API_KEY=your_key_here` to use the chatbot.")
    st.stop()

# Configure LLM & embedding settings with verified key
configure_settings(api_key=NVIDIA_API_KEY)


# ======================================================
# CACHED ENGINE LOADER
# ======================================================
@st.cache_resource
def get_cached_query_engine():
    """Load or restore the persistent index and return the fusion query engine."""
    manager = RAGIndexManager()
    index = manager.sync_index()
    if index is None:
        return None
    try:
        return build_fusion_query_engine(index)
    except Exception as e:
        logger.error(f"Error building query engine: {e}")
        return None


query_engine = get_cached_query_engine()

# ======================================================
# SIDEBAR
# ======================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <span style="font-size:1.6rem;">🧠</span>
        <span class="sidebar-logo-text">DocuMind</span>
    </div>
    """, unsafe_allow_html=True)

    is_ready = query_engine is not None
    dot_class = "status-dot" if is_ready else "status-dot status-offline"
    status_label = "Index Ready" if is_ready else "No Documents"
    st.markdown(f"""
    <div style="display:flex;align-items:center;margin-bottom:18px;">
        <span class="{dot_class}"></span>
        <span style="color:#64748b;font-size:0.8rem;font-weight:500;">{status_label}</span>
    </div>
    """, unsafe_allow_html=True)

    # Upload zone
    st.markdown('<div class="sidebar-section-title">📂 Upload Documents</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drop PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        os.makedirs(DEFAULT_DATA_DIR, exist_ok=True)
        files_updated = False

        for uploaded_file in uploaded_files:
            file_path = os.path.join(DEFAULT_DATA_DIR, uploaded_file.name)
            uploaded_bytes = uploaded_file.getbuffer().tobytes()

            # Detect new or modified content
            should_write = False
            if not os.path.exists(file_path):
                should_write = True
            else:
                try:
                    with open(file_path, "rb") as f:
                        disk_bytes = f.read()
                    if compute_bytes_hash(disk_bytes) != compute_bytes_hash(uploaded_bytes):
                        should_write = True
                except Exception:
                    should_write = True

            if should_write:
                try:
                    with open(file_path, "wb") as f:
                        f.write(uploaded_bytes)
                    files_updated = True
                except Exception as e:
                    st.error(f"Failed to save {uploaded_file.name}: {e}")

        if files_updated:
            with st.spinner("Indexing documents…"):
                try:
                    manager = RAGIndexManager()
                    manager.sync_index()
                    st.cache_resource.clear()
                    st.success("✅ Documents indexed successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Indexing failed: {e}")

    # Indexed Files List with Removal
    if os.path.exists(DEFAULT_DATA_DIR):
        pdfs = [f for f in os.listdir(DEFAULT_DATA_DIR) if f.lower().endswith(".pdf")]
        if pdfs:
            st.markdown(f'<div class="sidebar-section-title">📑 Indexed Files ({len(pdfs)})</div>', unsafe_allow_html=True)
            for f in pdfs:
                col1, col2 = st.columns([0.82, 0.18])
                with col1:
                    st.markdown(f"""
                    <div class="file-card">
                        <span class="file-card-icon">📄</span>
                        <span class="file-card-name" title="{f}">{f}</span>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    if st.button("🗑️", key=f"del_{f}", help=f"Remove {f} from index"):
                        manager = RAGIndexManager()
                        manager.remove_document(f)
                        st.cache_resource.clear()
                        st.rerun()

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Retrieval Pipeline Info
    st.markdown('<div class="sidebar-section-title">⚙️ Retrieval Pipeline</div>', unsafe_allow_html=True)
    with st.expander("View details", expanded=False):
        st.markdown("""
        <div style="padding:4px 0">
            <div class="pipeline-chip"><span class="chip-icon">🔀</span> Hybrid Search (BM25 + Vector)</div>
            <div class="pipeline-chip"><span class="chip-icon">🎯</span> Reciprocal Rank Fusion</div>
            <div class="pipeline-chip"><span class="chip-icon">🏆</span> Cross-encoder Reranker</div>
            <div class="pipeline-chip"><span class="chip-icon">💡</span> HyDE Query Transform</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:30px;text-align:center;">
        <p style="color:#1e293b;font-size:0.72rem;margin:0;">Powered by NVIDIA AI · LlamaIndex</p>
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# MAIN CONTENT — Hero Banner
# ======================================================
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">🧠 Multi-Document RAG</div>
    <h1 class="hero-title">Ask Anything Across Your Documents</h1>
    <p class="hero-subtitle">
        Upload multiple PDFs and query your entire knowledge base simultaneously.<br>
        Powered by hybrid search, neural reranking, and NVIDIA AI.
    </p>
</div>
""", unsafe_allow_html=True)

# ======================================================
# CHAT INTERFACE & SESSION STATE
# ======================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Empty state
if not st.session_state.messages:
    if query_engine is None:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📂</div>
            <p class="empty-title">No documents indexed yet</p>
            <p class="empty-sub">Upload one or more PDF files from the sidebar<br>to start chatting with your knowledge base.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        pdf_count = len([f for f in os.listdir(DEFAULT_DATA_DIR) if f.lower().endswith(".pdf")]) if os.path.exists(DEFAULT_DATA_DIR) else 0
        st.markdown(f"""
        <div class="empty-state">
            <div class="empty-icon">💬</div>
            <p class="empty-title">Ready to answer your questions</p>
            <p class="empty-sub">{pdf_count} document{"s" if pdf_count != 1 else ""} indexed · Type a question below to get started.</p>
        </div>
        """, unsafe_allow_html=True)

# Render existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ======================================================
# CHAT INPUT & QUERY HANDLING
# ======================================================
question = st.chat_input("Ask something about your documents…")

if question:
    if query_engine is None:
        st.warning("⚠️ Please upload at least one PDF via the sidebar to initialize the search engine.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Searching across documents…"):
            try:
                response = query_engine.query(question)
                answer = response.response
            except Exception as e:
                logger.exception(f"Query execution error: {e}")
                st.error(f"❌ An error occurred during search: {e}")
                answer = None

        if answer is not None:
            st.markdown(answer)

            # Citations section
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown("""
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
                <span style="font-size:1rem;">🔍</span>
                <span style="color:#a5b4fc;font-size:0.85rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;">Sources &amp; Citations</span>
            </div>
            """, unsafe_allow_html=True)

            bad_keywords = ["rdf:", "pdfSchema", "stream", "endstream", "<?xpacket"]
            seen_texts = set()
            source_count = 0

            for node in getattr(response, "source_nodes", []):
                text = node.text.strip()
                if any(keyword in text for keyword in bad_keywords) or text in seen_texts:
                    continue

                seen_texts.add(text)
                source_count += 1
                filename = node.metadata.get('source_file', 'Unknown Document')
                score = getattr(node, 'score', None)
                score_str = f"{score:.3f}" if score is not None else "—"

                st.markdown(f"""
                <div class="citation-card">
                    <div class="citation-header">
                        <span class="citation-badge">Source {source_count}</span>
                        <span class="citation-filename">📄 {filename}</span>
                        <span class="citation-score">score: {score_str}</span>
                    </div>
                    <p class="citation-text">{text[:420]}{"…" if len(text) > 420 else ""}</p>
                </div>
                """, unsafe_allow_html=True)

            if source_count == 0:
                st.markdown("""
                <div style="color:#475569;font-size:0.83rem;padding:12px 0;text-align:center;">
                    No verifiable citations found for this response.
                </div>
                """, unsafe_allow_html=True)

            st.session_state.messages.append({"role": "assistant", "content": answer})
