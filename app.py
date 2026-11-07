import os
import logging
import html
import time
import hashlib
import streamlit as st
from dotenv import load_dotenv

from core_rag import (
    RAGIndexManager,
    configure_settings,
    build_fusion_query_engine,
    compute_bytes_hash,
    DEFAULT_DATA_DIR,
    DEFAULT_CHROMA_DIR,
    GROQ_MODEL,
    GROQ_FALLBACK_MODEL,
    build_contextual_query,
    sanitize_filename,
    validate_pdf_bytes,
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
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ======================================================
# STREAMLIT PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="DocuMind — RAG PDF Chatbot",
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

# Professional visual overrides
st.markdown("""
<style>
:root {
    --ink: #172033;
    --muted: #64748b;
    --line: #e2e8f0;
    --surface: #ffffff;
    --surface-soft: #f8fafc;
    --accent: #2563eb;
}
html, body, [data-testid="stAppViewContainer"] {
    background: #f5f7fb !important;
    color: var(--ink) !important;
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}
[data-testid="stAppViewContainer"] > .main {
    background: #f5f7fb !important;
}
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--line) !important;
    min-width: 290px !important;
    max-width: 290px !important;
}
[data-testid="stSidebar"] * { color: var(--ink); }
#MainMenu, footer, header { visibility: hidden !important; }
.hero-banner {
    background: var(--surface) !important;
    border: 1px solid var(--line) !important;
    border-radius: 12px !important;
    box-shadow: 0 8px 24px rgba(15, 23, 42, .05) !important;
    padding: 28px 32px !important;
    margin-bottom: 24px !important;
}
.hero-banner::before, .hero-banner::after { display: none !important; }
.hero-title {
    color: var(--ink) !important;
    background: none !important;
    -webkit-text-fill-color: initial !important;
    font-size: 1.8rem !important;
    letter-spacing: -.02em !important;
}
.hero-subtitle { color: var(--muted) !important; }
.hero-badge {
    color: var(--accent) !important;
    background: #eff6ff !important;
    border: 1px solid #bfdbfe !important;
    border-radius: 6px !important;
}
.sidebar-logo {
    border-bottom: 1px solid var(--line) !important;
    padding-bottom: 16px !important;
}
.sidebar-logo-text {
    color: var(--ink) !important;
    background: none !important;
    -webkit-text-fill-color: initial !important;
}
.sidebar-section-title {
    color: #475569 !important;
    letter-spacing: .06em !important;
    font-size: .68rem !important;
    font-weight: 700 !important;
    margin: 22px 0 9px !important;
}
.pipeline-chip, .file-card {
    background: var(--surface-soft) !important;
    border: 1px solid var(--line) !important;
    border-radius: 7px !important;
    color: #334155 !important;
}
.file-card {
    min-height: 38px !important;
    padding: 8px 10px !important;
}
.file-card-name {
    color: #334155 !important;
    display: block !important;
    min-width: 0 !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
}
[data-testid="stSidebar"] [data-testid="column"]:last-child button {
    width: 100% !important;
    min-height: 38px !important;
    padding: 0 6px !important;
    white-space: nowrap !important;
    font-size: .68rem !important;
    letter-spacing: -.02em !important;
}
[data-testid="stFileUploader"] {
    background: var(--surface-soft) !important;
    border: 1px dashed #cbd5e1 !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploader"] [data-testid="stIconMaterial"] {
    display: none !important;
}
[data-testid="stFileUploader"] button p,
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] p {
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
[data-testid="stChatInput"] {
    background: var(--surface) !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 14px rgba(15, 23, 42, .06) !important;
    min-height: 52px !important;
}
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] textarea:focus {
    color: #0f172a !important;
    -webkit-text-fill-color: #0f172a !important;
    caret-color: #2563eb !important;
    background: #ffffff !important;
    font-size: .95rem !important;
    line-height: 1.45 !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #94a3b8 !important;
    opacity: 1 !important;
}
[data-testid="stChatMessage"] {
    background: transparent !important;
}
[data-testid="stChatMessageAvatar"] {
    width: 2rem !important;
    min-width: 2rem !important;
    height: 2rem !important;
}
.citation-card {
    background: var(--surface) !important;
    border: 1px solid var(--line) !important;
    border-left: 3px solid var(--accent) !important;
    border-radius: 8px !important;
    box-shadow: none !important;
}
.citation-filename, .citation-badge { color: #1d4ed8 !important; }
.citation-text { color: #475569 !important; }
.empty-state { padding: 48px 40px !important; }
.empty-icon { display: none !important; }
.empty-title { color: var(--ink) !important; font-size: 1.05rem !important; }
.empty-sub { color: var(--muted) !important; }
.workspace-meta {
    display: flex;
    gap: 8px;
    align-items: center;
    color: #64748b;
    font-size: .78rem;
    margin: 8px 0 0;
}
.status-dot {
    width: 7px !important;
    height: 7px !important;
    box-shadow: none !important;
    animation: none !important;
}
.status-offline { background: #94a3b8 !important; }
.main .block-container {
    max-width: 1180px !important;
    padding-top: 2rem !important;
    padding-bottom: 10rem !important;
}
[data-testid="stChatMessage"] {
    padding-top: .75rem !important;
    padding-bottom: .75rem !important;
}
[data-testid="stChatMessageContent"] {
    color: #0f172a !important;
    line-height: 1.7 !important;
    max-width: 820px !important;
}
[data-testid="stChatMessageContent"] p {
    margin-bottom: .65rem !important;
}
html body .stApp [data-testid="stChatInput"],
html body .stApp [data-testid="stChatInput"]:focus-within {
    border: 1px solid #cbd5e1 !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, .10) !important;
}
html body .stApp [data-testid="stChatInput"] textarea,
html body .stApp [data-testid="stChatInput"] textarea:focus {
    outline: none !important;
    border: 0 !important;
}
html body .stApp [data-testid="stBottom"] {
    background: rgba(248, 250, 252, .96) !important;
    padding: 10px 0 14px !important;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# API KEY VALIDATION
# ======================================================
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY was not found in the environment or .env file. Configure it before starting the chatbot.")
    st.stop()

APP_PASSWORD = os.getenv("APP_PASSWORD")
APP_USERNAME = "default"
if APP_PASSWORD:
    APP_USERNAME = st.sidebar.text_input("Workspace name", key="workspace_name").strip()
    entered_password = st.sidebar.text_input("Workspace password", type="password", key="workspace_password")
    if not APP_USERNAME or entered_password != APP_PASSWORD:
        st.sidebar.info("Enter your workspace credentials to continue.")
        st.stop()

workspace_id = hashlib.sha256(APP_USERNAME.encode("utf-8")).hexdigest()[:16]
APP_DATA_DIR = os.path.join("data", "users", workspace_id) if APP_PASSWORD else DEFAULT_DATA_DIR
APP_CHROMA_DIR = os.path.join("chroma_db", "users", workspace_id) if APP_PASSWORD else DEFAULT_CHROMA_DIR

# Configure LLM & embedding settings with verified key
configure_settings(api_key=GROQ_API_KEY)


# ======================================================
# CACHED ENGINE LOADER
# ======================================================
@st.cache_resource
def get_cached_query_engine(model: str = GROQ_MODEL):
    """Load the persistent index and bind it to the requested provider model."""
    configure_settings(api_key=GROQ_API_KEY, model=model)
    manager = RAGIndexManager(data_dir=APP_DATA_DIR, chroma_dir=APP_CHROMA_DIR)
    index = manager.sync_index()
    if index is None:
        return None
    try:
        return build_fusion_query_engine(index, streaming=True)
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
    if APP_PASSWORD:
        st.markdown(
            f'<div class="workspace-meta">Workspace <strong>{html.escape(APP_USERNAME)}</strong></div>',
            unsafe_allow_html=True,
        )

    # Upload zone
    st.markdown('<div class="sidebar-section-title">Upload documents</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drop PDF files here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        os.makedirs(APP_DATA_DIR, exist_ok=True)
        files_updated = False

        for uploaded_file in uploaded_files:
            try:
                safe_name = sanitize_filename(uploaded_file.name)
                uploaded_bytes = uploaded_file.getbuffer().tobytes()
                validate_pdf_bytes(uploaded_bytes)
            except ValueError as exc:
                st.error(f"Cannot upload {html.escape(uploaded_file.name)}: {exc}")
                continue
            file_path = os.path.join(APP_DATA_DIR, safe_name)

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
                    logger.exception("Failed to save uploaded PDF")
                    st.error(f"Failed to save {html.escape(safe_name)}. Check the application logs.")

        if files_updated:
            with st.spinner("Indexing documents…"):
                try:
                    manager = RAGIndexManager(data_dir=APP_DATA_DIR, chroma_dir=APP_CHROMA_DIR)
                    manager.sync_index()
                    st.cache_resource.clear()
                    st.success("Documents indexed successfully.")
                    st.rerun()
                except Exception as e:
                    logger.exception("Document indexing failed")
                    st.error("Indexing failed. Check the application logs for details.")

    # Indexed Files List with Removal
    if os.path.exists(APP_DATA_DIR):
        pdfs = [f for f in os.listdir(APP_DATA_DIR) if f.lower().endswith(".pdf")]
        if pdfs:
            st.markdown(f'<div class="sidebar-section-title">Indexed files ({len(pdfs)})</div>', unsafe_allow_html=True)
            for f in pdfs:
                col1, col2 = st.columns([0.62, 0.38], gap="small")
                with col1:
                    st.markdown(f"""
                    <div class="file-card">
                        <span class="file-card-name" title="{html.escape(f)}">{html.escape(f)}</span>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    if st.button("×", key=f"del_{f}", help=f"Remove {f} from index"):
                        manager = RAGIndexManager(data_dir=APP_DATA_DIR, chroma_dir=APP_CHROMA_DIR)
                        manager.remove_document(f)
                        st.cache_resource.clear()
                        st.rerun()

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Retrieval Pipeline Info
    st.markdown('<div class="sidebar-section-title">Retrieval pipeline</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.caption("Active configuration")
        st.markdown("""
        <div style="padding:2px 0">
            <div class="pipeline-chip">Hybrid search: BM25 + vector</div>
            <div class="pipeline-chip">Reciprocal rank fusion</div>
            <div class="pipeline-chip">Cross-encoder reranking</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:30px;text-align:center;">
        <p style="color:#1e293b;font-size:0.72rem;margin:0;">Powered by Groq · LlamaIndex</p>
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# MAIN CONTENT — Hero Banner
# ======================================================
st.markdown("""
<div class="hero-banner">
<div class="hero-badge">Multi-document knowledge base</div>
    <h1 class="hero-title">Ask Anything Across Your Documents</h1>
    <p class="hero-subtitle">
        Upload multiple PDFs and query your entire knowledge base simultaneously.<br>
        Powered by hybrid search, neural reranking, and Groq.
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
            <p class="empty-title">No documents indexed yet</p>
            <p class="empty-sub">Upload one or more PDF files from the sidebar<br>to start chatting with your knowledge base.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        pdf_count = len([f for f in os.listdir(APP_DATA_DIR) if f.lower().endswith(".pdf")]) if os.path.exists(APP_DATA_DIR) else 0
        st.markdown(f"""
        <div class="empty-state">
            <p class="empty-title">Ready to answer your questions</p>
            <p class="empty-sub">{pdf_count} document{"s" if pdf_count != 1 else ""} indexed. Type a question below to get started.</p>
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
        st.warning("Please upload at least one PDF via the sidebar to initialize the search engine.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Searching across documents…"):
            answer_was_streamed = False
            try:
                contextual_query = build_contextual_query(question, st.session_state.messages[:-1])
                started_at = time.perf_counter()
                response = query_engine.query(contextual_query)
                if hasattr(response, "response_gen"):
                    answer = st.write_stream(response.response_gen)
                    answer_was_streamed = True
                else:
                    answer = response.response
                elapsed_ms = (time.perf_counter() - started_at) * 1000
                logger.info("query_completed latency_ms=%.0f source_count=%d", elapsed_ms, len(response.source_nodes))
            except Exception as e:
                logger.exception(f"Query execution error: {e}")
                try:
                    fallback_engine = get_cached_query_engine(GROQ_FALLBACK_MODEL)
                    response = fallback_engine.query(contextual_query)
                    if hasattr(response, "response_gen"):
                        answer = st.write_stream(response.response_gen)
                        answer_was_streamed = True
                    else:
                        answer = response.response
                    st.info("The primary model was unavailable; the fallback model answered this request.")
                except Exception:
                    logger.exception("Fallback model query failed")
                    st.error("The language-model provider could not complete this request.")
                    st.caption(
                        "Verify GROQ_API_KEY and GROQ_MODEL in .env, then restart the app. "
                        "Check the terminal log for the exact provider status."
                    )
                    answer = None

        if answer is not None:
            if not answer_was_streamed:
                st.markdown(answer)
            # Citations section
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.markdown("""
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
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
                filename = html.escape(str(node.metadata.get('source_file', 'Unknown Document')))
                page_number = node.metadata.get("page_number")
                page_label = f" · page {page_number}" if page_number else ""
                score = getattr(node, 'score', None)
                score_str = f"{score:.3f}" if score is not None else "—"

                st.markdown(f"""
                <div class="citation-card">
                    <div class="citation-header">
                        <span class="citation-badge">Source {source_count}</span>
                        <span class="citation-filename">{filename}{page_label}</span>
                        <span class="citation-score">score: {score_str}</span>
                    </div>
                    <p class="citation-text">{html.escape(text[:420])}{"…" if len(text) > 420 else ""}</p>
                </div>
                """, unsafe_allow_html=True)

            if source_count == 0:
                st.markdown("""
                <div style="color:#475569;font-size:0.83rem;padding:12px 0;text-align:center;">
                    No verifiable citations found for this response.
                </div>
                """, unsafe_allow_html=True)

            st.session_state.messages.append({"role": "assistant", "content": answer})
