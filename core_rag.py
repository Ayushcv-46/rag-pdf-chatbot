import os
import json
import hashlib
import logging
import re
import time
from typing import List, Optional, Tuple, Dict, Any

import chromadb
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    PromptTemplate,
    Document
)
from llama_index.core.schema import TextNode, BaseNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
import llama_index.llms.openai.utils as openai_utils
from llama_index.readers.file import PyMuPDFReader
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank

logger = logging.getLogger("rag_pdf_chatbot")

# ======================================================
# CONSTANTS & CONFIGURATION
# ======================================================
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b")
GROQ_API_BASE = "https://api.groq.com/openai/v1"
COLLECTION_NAME = "pdf_collection"
DEFAULT_DATA_DIR = "./data"
DEFAULT_CHROMA_DIR = "./chroma_db"
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "250"))
MAX_CONTEXT_MESSAGES = int(os.getenv("MAX_CONTEXT_MESSAGES", "6"))

FALLBACK_RESPONSE = "I could not find that information across the uploaded documents."

APPLICATION_ARCHITECTURE_INFO = (
    "ACTUAL APPLICATION RUNTIME ARCHITECTURE FACTS:\n"
    "- Architecture: Deterministic Multi-Document RAG Pipeline (Streamlit frontend + LlamaIndex engine + ChromaDB storage + Groq).\n"
    "- Retrieval Flow: User Query -> Parallel Hybrid Search (ChromaDB vector similarity top-15 + BM25 keyword top-15) "
    "-> Reciprocal Rank Fusion (RRF top-15) -> Cross-Encoder Reranker (BAAI/bge-reranker-base top-5) -> LLM Synthesis (openai/gpt-oss-120b).\n"
    "- Multi-Document Search Mechanism: All uploaded documents are indexed into a single shared ChromaDB collection and BM25 store. "
    "The application does NOT decide, route, or select which document to search; it searches across ALL uploaded documents simultaneously, "
    "retrieving and reranking the most relevant chunks globally based on similarity to the query.\n"
    "- Explicitly Non-Implemented Components: This application does NOT use a ReAct agent, autonomous agent loop, QueryEngineTool, "
    "calculator tool, LangChain, or LangGraph. Any mention of agents, tools, or prospective architectures in uploaded documents refers to tutorial "
    "exercises or interview questions—NOT the actual implementation of this chatbot."
)

QA_PROMPT_TEMPLATE = (
    "You are an expert multi-document question answering assistant. "
    "Your answers must be grounded strictly and accurately in the provided sources.\n\n"
    f"{APPLICATION_ARCHITECTURE_INFO}\n\n"
    "Guidelines:\n"
    "1. DETERMINE QUESTION TYPE FIRST:\n"
    "   - Architecture / meta questions (about this chatbot, its pipeline, how it searches): use ONLY the ACTUAL APPLICATION RUNTIME ARCHITECTURE FACTS above.\n"
    "   - Document content questions (revision notes, tutorial topics, 'Day N' questions, concepts from uploaded PDFs): use ONLY the DOCUMENT CONTEXT below.\n\n"
    "2. ANSWERING DOCUMENT CONTENT QUESTIONS:\n"
    "   - Base every factual claim, number, parameter, and code snippet on information explicitly stated in the DOCUMENT CONTEXT.\n"
    "   - Do NOT introduce specific numbers, benchmarks, latency estimates, or implementation details that are not directly stated in the DOCUMENT CONTEXT.\n"
    "   - Do NOT extrapolate beyond what the context says. If the context says 'fast but approximate' and 'slow but accurate', report those characterizations "
    "without adding specific millisecond or token-count figures.\n"
    "   - Preserve the exact terminology, key parameters, and code snippets from the context.\n\n"
    "3. ANSWERING ARCHITECTURE / META QUESTIONS:\n"
    "   - Rely strictly on the ACTUAL APPLICATION RUNTIME ARCHITECTURE FACTS.\n"
    "   - Explain that all uploaded documents are indexed into a single shared collection and searched simultaneously using hybrid retrieval and reranking.\n"
    "   - Explicitly state that this chatbot does NOT use a ReAct agent, autonomous agent loop, QueryEngineTool, or document routing.\n\n"
    "4. ARCHITECTURAL BOUNDARIES: Distinguish clearly between this application's implemented components and concepts taught in documents. "
    "Never claim that this application uses a component (such as ReAct or LangGraph) merely because it is taught or discussed in an uploaded document.\n\n"
    "5. FALLBACK RULE: If the DOCUMENT CONTEXT does not contain sufficient information to answer the question, "
    "and the question is not answerable from the runtime architecture facts either, respond EXACTLY with:\n"
    f'   "{FALLBACK_RESPONSE}"\n'
    "   Do NOT use this fallback if the DOCUMENT CONTEXT contains relevant information, even if the information is brief.\n\n"
    "6. Keep your tone objective, precise, and direct, without speculative fluff or conversational commentary.\n\n"
    "---------------------\n"
    "DOCUMENT CONTEXT:\n"
    "{context_str}\n"
    "---------------------\n\n"
    "Question: {query_str}\n\n"
    "Answer:"
)

qa_prompt = PromptTemplate(QA_PROMPT_TEMPLATE)


def get_runtime_architecture_info() -> Dict[str, Any]:
    """Return dictionary of verified runtime architecture components."""
    return {
        "pipeline_type": "deterministic_multi_stage_rag",
        "has_react_agent": False,
        "has_query_engine_tool": False,
        "has_langgraph": False,
        "document_routing": "simultaneous_unified_collection_search",
        "retrieval": {
            "vector": "chromadb_similarity_top_15",
            "keyword": "bm25_top_15",
            "fusion": "reciprocal_rank_fusion",
            "reranker": "BAAI/bge-reranker-base_top_5",
            "query_transform": "none"
        },
        "models": {
            "llm": GROQ_MODEL,
            "embed_model": "sentence-transformers/all-MiniLM-L6-v2"
        }
    }


def is_architecture_or_meta_query(query: str) -> bool:
    """Identify if a user question pertains to the chatbot's own architecture or search decision mechanism."""
    q = query.lower()
    meta_keywords = [
        "this chatbot", "the chatbot", "the app", "this app",
        "which document to search", "decide which document",
        "how does it search", "what architecture", "react agent"
    ]
    return any(kw in q for kw in meta_keywords)


def compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file for change detection."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_bytes_hash(file_bytes: bytes) -> str:
    """Compute SHA-256 hash of raw bytes."""
    return hashlib.sha256(file_bytes).hexdigest()


def sanitize_filename(filename: str) -> str:
    """Return a safe, flat PDF filename suitable for storage under the data directory."""
    basename = os.path.basename(filename).strip()
    safe_name = re.sub(r"[^A-Za-z0-9._ -]", "_", basename)
    safe_name = re.sub(r"\s+", " ", safe_name).strip(" .")
    if not safe_name or not safe_name.lower().endswith(".pdf"):
        raise ValueError("Only files with a .pdf extension are accepted.")
    return safe_name[:180]


def validate_pdf_bytes(file_bytes: bytes) -> None:
    """Validate upload size, PDF signature, and page count before writing to disk."""
    if not file_bytes:
        raise ValueError("The uploaded file is empty.")
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError(f"PDF exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.")
    if not file_bytes.startswith(b"%PDF-"):
        raise ValueError("The uploaded file is not a valid PDF.")

    import fitz
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as document:
            if document.page_count > MAX_PDF_PAGES:
                raise ValueError(f"PDF exceeds the {MAX_PDF_PAGES}-page limit.")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not validate PDF: {exc}") from exc


def build_contextual_query(question: str, messages: Optional[List[Dict[str, str]]] = None) -> str:
    """Add a bounded conversation window so follow-up questions resolve correctly."""
    if not messages:
        return question.strip()
    recent = messages[-MAX_CONTEXT_MESSAGES:]
    history = "\n".join(
        f"{message.get('role', 'user')}: {message.get('content', '')[:1000]}"
        for message in recent
        if message.get("content")
    )
    return f"Conversation context:\n{history}\n\nCurrent question: {question.strip()}"


def configure_settings(api_key: Optional[str] = None, model: str = GROQ_MODEL):
    """Configure LLM and embedding settings matching existing parameters."""
    openai_utils.ALL_AVAILABLE_MODELS[model] = 131072
    openai_utils.CHAT_MODELS[model] = 131072

    if api_key:
        Settings.llm = OpenAI(
            model=model,
            api_key=api_key,
            api_base=GROQ_API_BASE,
            temperature=0.2,
            max_tokens=4096,
            timeout=30.0,
            max_retries=2,
            additional_kwargs={"top_p": 1}
        )

    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50


def load_pdf_file(file_path: str) -> List[Document]:
    """
    Safely load a PDF file using PyMuPDFReader.
    Returns empty list if file cannot be parsed.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    if file_size > MAX_UPLOAD_BYTES:
        raise ValueError(f"PDF exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.")
    if file_size == 0:
        logger.warning(f"File {file_name} is empty (0 bytes). Skipping.")
        return []

    loader = PyMuPDFReader()
    try:
        docs = loader.load(file_path=file_path)
    except Exception as e:
        logger.error(f"Failed to parse PDF {file_name}: {e}")
        raise ValueError(f"Could not read PDF '{file_name}': {e}") from e

    if len(docs) > MAX_PDF_PAGES:
        raise ValueError(f"PDF exceeds the {MAX_PDF_PAGES}-page limit.")
    for page_number, doc in enumerate(docs, start=1):
        doc.metadata["source_file"] = file_name
        doc.metadata["page_number"] = page_number

    return docs


class RAGIndexManager:
    """
    Manages persistent ChromaDB vector storage and docstore synchronization.
    Supports atomic index updates, replacement of modified documents, and removal.
    """

    def __init__(self, data_dir: str = DEFAULT_DATA_DIR, chroma_dir: str = DEFAULT_CHROMA_DIR):
        self.data_dir = data_dir
        self.chroma_dir = chroma_dir
        self.manifest_path = os.path.join(chroma_dir, "manifest.json")
        self.docstore_path = os.path.join(chroma_dir, "docstore.json")

        # Ensure default HuggingFaceEmbedding is set if no custom embed_model (e.g. MockEmbedding) is provided
        if getattr(Settings, "_embed_model", None) is None:
            configure_settings()

        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.chroma_dir, exist_ok=True)


        self.db = chromadb.PersistentClient(path=self.chroma_dir)
        self.chroma_collection = self.db.get_or_create_collection(COLLECTION_NAME)
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self._load_or_init_storage()

    def _load_or_init_storage(self):
        """Load manifest and docstore from disk, or initialize empty."""
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    self.manifest: Dict[str, Any] = json.load(f)
            except Exception as e:
                logger.warning(f"Could not read manifest at {self.manifest_path}: {e}. Reinitializing.")
                self.manifest = {}
        else:
            self.manifest = {}

        if os.path.exists(self.docstore_path):
            try:
                self.docstore = SimpleDocumentStore.from_persist_path(self.docstore_path)
            except Exception as e:
                logger.warning(f"Could not read docstore at {self.docstore_path}: {e}. Reinitializing.")
                self.docstore = SimpleDocumentStore()
        else:
            self.docstore = SimpleDocumentStore()

    def _save_storage(self):
        """Persist manifest and docstore to disk."""
        try:
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                json.dump(self.manifest, f, indent=2)
            self.docstore.persist(self.docstore_path)
        except Exception as e:
            logger.error(f"Failed to persist index storage: {e}")

    def get_indexed_files(self) -> List[str]:
        """Return list of filenames currently indexed in manifest."""
        return list(self.manifest.keys())

    def remove_document(self, filename: str) -> bool:
        """
        Remove all chunks for a document from Chroma, docstore, and manifest.
        Also deletes physical file from data directory if present.
        """
        removed = False
        file_path = os.path.join(self.data_dir, filename)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                removed = True
                logger.info(f"Removed physical file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to delete file {file_path}: {e}")

        try:
            self.chroma_collection.delete(where={"source_file": filename})
            removed = True
        except Exception as e:
            logger.debug(f"Chroma delete for {filename}: {e}")

        if filename in self.manifest:
            node_ids = self.manifest[filename].get("node_ids", [])
            for nid in node_ids:
                try:
                    self.docstore.delete_document(nid)
                except Exception as e:
                    logger.debug(f"Docstore delete error for {nid}: {e}")
            del self.manifest[filename]
            removed = True

        self._save_storage()
        return removed

    def clear_all(self):
        """Completely clear Chroma collection, docstore, and manifest."""
        existing_ids = self.chroma_collection.get(include=[]).get("ids", [])
        if existing_ids:
            self.chroma_collection.delete(ids=existing_ids)
        self.docstore = SimpleDocumentStore()
        self.manifest = {}
        self._save_storage()
        logger.info("Cleared all index state.")

    def sync_index(self) -> Optional[VectorStoreIndex]:
        """
        Synchronize the persistent index with files in data_dir.
        - Detects new files and indexes them.
        - Detects modified files (content hash changed), removes old chunks, and re-indexes.
        - Detects deleted files and purges their chunks.
        - If no files exist, returns None without crashing.
        - Does NOT wipe valid index data before new indexing succeeds.
        """
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)

        current_files = {f: os.path.join(self.data_dir, f) 
                         for f in os.listdir(self.data_dir) 
                         if f.lower().endswith(".pdf")}

        # 1. Purge files that no longer exist on disk
        deleted_files = [f for f in list(self.manifest.keys()) if f not in current_files]
        for f in deleted_files:
            logger.info(f"Purging deleted document from index: {f}")
            self.remove_document(f)

        if not current_files:
            self.clear_all()
            return None

        splitter = SentenceSplitter(chunk_size=Settings.chunk_size, chunk_overlap=Settings.chunk_overlap)
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store, docstore=self.docstore)

        # 2. Identify new and modified files
        files_to_index: List[Tuple[str, str, str]] = []  # (filename, path, file_hash)
        for fname, fpath in current_files.items():
            try:
                fhash = compute_file_hash(fpath)
            except Exception as e:
                logger.error(f"Cannot compute hash for {fpath}: {e}")
                continue

            if fname not in self.manifest or self.manifest[fname].get("hash") != fhash:
                files_to_index.append((fname, fpath, fhash))

        # 3. Process new and modified files
        for fname, fpath, fhash in files_to_index:
            logger.info(f"Indexing document: {fname}")
            try:
                docs = load_pdf_file(fpath)
                if not docs:
                    continue

                nodes = splitter.get_nodes_from_documents(docs)
                for n in nodes:
                    n.metadata["source_file"] = fname
                    n.metadata["file_hash"] = fhash

                # If replacing an existing document, delete previous chunks first
                if fname in self.manifest:
                    logger.info(f"Replacing existing document chunks for: {fname}")
                    try:
                        self.chroma_collection.delete(where={"source_file": fname})
                    except Exception as e:
                        logger.debug(f"Chroma delete old chunks for {fname}: {e}")
                    for old_nid in self.manifest[fname].get("node_ids", []):
                        try:
                            self.docstore.delete_document(old_nid)
                        except Exception as e:
                            logger.debug(f"Docstore delete old chunk {old_nid}: {e}")

                # Insert into vector store and docstore
                VectorStoreIndex(
                    nodes=nodes,
                    storage_context=storage_context,
                    store_nodes_override=True
                )

                self.manifest[fname] = {
                    "hash": fhash,
                    "node_ids": [n.node_id for n in nodes],
                    "chunk_count": len(nodes)
                }
                self._save_storage()
            except Exception as e:
                logger.error(f"Failed to index document {fname}: {e}")
                raise

        # 4. Build or restore the VectorStoreIndex
        if not self.manifest:
            return None

        index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            storage_context=storage_context
        )
        index._docstore = self.docstore
        return index


def build_fusion_query_engine(
    index: VectorStoreIndex,
    reranker=None,
    qa_template=None,
    streaming: bool = False,
) -> RetrieverQueryEngine:
    """
    Combines vector search + BM25 keyword search using reciprocal rank fusion,
    followed by cross-encoder reranking.
    Preserves exact retrieval hyperparameters.
    """
    nodes = list(index.docstore.docs.values())
    logger.debug(f"Nodes found in docstore for fusion: {len(nodes)}")

    if not nodes:
        raise ValueError("Docstore is empty; cannot construct BM25Retriever.")

    vector_retriever = index.as_retriever(similarity_top_k=15)
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=15)

    fusion_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=15,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=False,
    )

    if reranker is None:
        reranker = SentenceTransformerRerank(
            model="BAAI/bge-reranker-base",
            top_n=5,
        )

    if qa_template is None:
        qa_template = qa_prompt

    base_query_engine = RetrieverQueryEngine.from_args(
        fusion_retriever,
        node_postprocessors=[reranker],
        text_qa_template=qa_template,
        response_mode="compact",
        streaming=streaming,
    )

    return base_query_engine
