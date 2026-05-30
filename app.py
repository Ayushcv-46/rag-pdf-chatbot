import os
import streamlit as st
import chromadb
from llama_index.readers.file import PyMuPDFReader
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
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
    page_title="RAG PDF Chatbot",
    layout="wide"
)

st.title("RAG PDF Chatbot with Gemini")


# ======================================================
# GEMINI API CONFIGURATION
# ======================================================

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


# ======================================================
# LLM CONFIGURATION
# ======================================================

Settings.llm = Gemini(
    model="models/gemini-2.5-flash"
)


# ======================================================
# LOCAL EMBEDDING MODEL
# ======================================================

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ======================================================
# CHUNK SETTINGS
# ======================================================

Settings.chunk_size = 512
Settings.chunk_overlap = 50


# ======================================================
# CHROMADB SETUP
# ======================================================

db = chromadb.PersistentClient(
    path="./chroma_db"
)

chroma_collection = db.get_or_create_collection(
    "pdf_collection"
)

vector_store = ChromaVectorStore(
    chroma_collection=chroma_collection
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)


# ======================================================
# LOAD INDEX
# ======================================================

@st.cache_resource
def load_index():

    if not os.path.exists("./data"):
        return None

    if len(os.listdir("./data")) == 0:
        return None

    loader = PyMuPDFReader()

    documents = []

    for file in os.listdir("./data"):

        if file.endswith(".pdf"):

            docs = loader.load(
                file_path=os.path.join("./data", file)
            )

            documents.extend(docs)

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
    """You are a document assistant. You ONLY answer using the exact text provided below.
Do NOT use any outside knowledge. Do NOT guess or infer beyond what is written.

If the answer is not explicitly stated in the context below, respond with:
"I could not find that information in the uploaded document."

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
# PDF UPLOAD
# ======================================================

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    os.makedirs("data", exist_ok=True)

    file_path = os.path.join(
        "data",
        uploaded_file.name
    )

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF uploaded successfully!")

    # clear old cache
    st.cache_resource.clear()

    # reload index
    index = load_index()

    # recreate query engine
    if index is not None:

        query_engine = index.as_query_engine(
            similarity_top_k=5,
            text_qa_template=qa_prompt
        )


# ======================================================
# SESSION STATE
# ======================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ======================================================
# DISPLAY OLD CHAT
# ======================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])


# ======================================================
# CHAT INPUT
# ======================================================

question = st.chat_input(
    "Ask a question from your PDF..."
)


# ======================================================
# HANDLE QUESTION
# ======================================================

if question:

    if query_engine is None:

        st.warning("Please upload a PDF first.")

        st.stop()

    # ----------------------------------
    # USER MESSAGE
    # ----------------------------------

    with st.chat_message("user"):

        st.write(question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # ----------------------------------
    # QUERY
    # ----------------------------------

    response = query_engine.query(question)

    answer = response.response

    # ----------------------------------
    # ASSISTANT MESSAGE
    # ----------------------------------

    with st.chat_message("assistant"):

        st.write(answer)

        st.write("### Sources")

        bad_keywords = [
            "rdf:",
            "pdfSchema",
            "stream",
            "endstream",
            "<?xpacket"
        ]

        seen_texts = set()
        source_count = 0

        for node in response.source_nodes:

            text = node.text.strip()

            if any(keyword in text for keyword in bad_keywords):
                continue

            # Skip duplicate chunks
            if text in seen_texts:
                continue
            seen_texts.add(text)

            source_count += 1

            st.write(f"**Source {source_count}**")
            st.write(text[:500])
            st.divider()

        if source_count == 0:
            st.write("No clean sources found.")
           

 
                   
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )