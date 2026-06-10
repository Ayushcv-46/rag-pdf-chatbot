import os
import streamlit as st
import chromadb
from llama_index.readers.file import PyMuPDFReader
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    PromptTemplate
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini

# ======================================================
# LOAD ENV VARIABLES
# ======================================================
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

# ======================================================
# STREAMLIT PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Multi-RAG PDF Chatbot",
    layout="wide"
)

st.title("📚 Multi-Doc RAG Chatbot with Gemini")
st.caption("Upload multiple PDFs and query across your entire knowledge base simultaneously.")

# ======================================================
# CONFIGURATION (LLM, EMBEDDINGS, CHUNKING)
# ======================================================
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
Settings.llm = Gemini(model="models/gemini-2.5-flash")
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.chunk_size = 512
Settings.chunk_overlap = 50

# ======================================================
# CHROMADB SETUP
# ======================================================
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("pdf_collection")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# ======================================================
# LOAD INDEX FUNCTION
# ======================================================

@st.cache_resource
def load_index():
    if not os.path.exists("./data") or len(os.listdir("./data")) == 0:
        return None

    loader = PyMuPDFReader()
    documents = []

    # Iterate and parse every single PDF inside the data folder
    for file in os.listdir("./data"):
        if file.endswith(".pdf"):
            docs = loader.load(file_path=os.path.join("./data", file))
            # Tag each document chunk with its original filename for better traceability
            for doc in docs:
                doc.metadata["source_file"] = file
            documents.extend(docs)

    if not documents:
        return None

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context
    )
    return index

index = load_index()

# ======================================================
# CREATE QUERY ENGINE
# ======================================================
qa_prompt = PromptTemplate(
    """You are an advanced multi-document assistant. You ONLY answer using the exact text provided below.
Do NOT use any outside knowledge. Do NOT guess or infer beyond what is written.

If the answer is not explicitly stated in the context below, respond with:
"I could not find that information across the uploaded documents."

---------------------
DOCUMENT CONTEXT:
{context_str}
---------------------

Question: {query_str}

Answer using ONLY the context above:"""
)

query_engine = None
if index is not None:
    query_engine = index.as_query_engine(
        similarity_top_k=5,
        text_qa_template=qa_prompt,
        response_mode="compact"
    )

# ======================================================
# MULTI-PDF UPLOAD UI
# ======================================================
with st.sidebar:
    st.header("📂 Document Management")
    # Added accept_multiple_files=True to allow batch uploads
    uploaded_files = st.file_uploader(
        "Upload your PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        os.makedirs("data", exist_ok=True)
        new_files_added = False

        for uploaded_file in uploaded_files:
            file_path = os.path.join("data", uploaded_file.name)
            
            # Save file only if it doesn't already exist to avoid repetitive I/O operations
            if not os.path.exists(file_path):
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                new_files_added = True

        if new_files_added:
            st.success("New PDFs processed successfully!")
            st.cache_resource.clear()  # Drop the single-doc cache
            index = load_index()       # Re-index all files together
            if index is not None:
                query_engine = index.as_query_engine(
                    similarity_top_k=5,
                    text_qa_template=qa_prompt
                )
            st.rerun()

    # Show inventory of currently indexed documents
    if os.path.exists("./data") and len(os.listdir("./data")) > 0:
        st.markdown("---")
        st.markdown("**Currently Indexed Files:**")
        for f in os.listdir("./data"):
            if f.endswith(".pdf"):
                st.caption(f"📄 {f}")

# ======================================================
# CHAT INTERFACE & SESSION STATE
# ======================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input("Ask something about your collective knowledge base...")

if question:
    if query_engine is None:
        st.warning("Please upload at least one PDF file via the sidebar to initialize search.")
        st.stop()

    with st.chat_message("user"):
        st.write(question)

    st.session_state.messages.append({"role": "user", "content": question})

    # Query Execution
    response = query_engine.query(question)
    answer = response.response

    with st.chat_message("assistant"):
        st.write(answer)
        st.write("### 🔍 Citations & Sources")

        bad_keywords = ["rdf:", "pdfSchema", "stream", "endstream", "<?xpacket"]
        seen_texts = set()
        source_count = 0

        for node in response.source_nodes:
            text = node.text.strip()
            if any(keyword in text for keyword in bad_keywords) or text in seen_texts:
                continue
            
            seen_texts.add(text)
            source_count += 1
            
            # Extract metadata origin filename if available
            filename = node.metadata.get('source_file', 'Unknown Document')

            st.write(f"**Source {source_count}** | From: `{filename}`")
            st.write(text[:400] + "...")
            st.divider()

        if source_count == 0:
            st.write("No clean text citations extracted for this response cycle.")

    st.session_state.messages.append({"role": "assistant", "content": answer})