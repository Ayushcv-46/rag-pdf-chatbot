import os
import asyncio
import chromadb
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
import llama_index.llms.openai.utils as openai_utils
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PyMuPDFReader
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine, TransformQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core import PromptTemplate

# Import LlamaIndex Evaluation Modules
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    BatchEvalRunner
)

# 1. Load environment and configurations
load_dotenv()

# Build NVIDIA API setup
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_MODEL = "openai/gpt-oss-20b"

if not NVIDIA_API_KEY:
    raise ValueError("NVIDIA_API_KEY not found in .env file")

# Register the model in LlamaIndex's OpenAI utils so it recognizes the context window
openai_utils.ALL_AVAILABLE_MODELS[NVIDIA_MODEL] = 128000
openai_utils.CHAT_MODELS[NVIDIA_MODEL] = 128000

# Set up the LLM pointing to NVIDIA NIM using the OpenAI interface
Settings.llm = OpenAI(
    model=NVIDIA_MODEL,
    api_key=NVIDIA_API_KEY,
    api_base="https://integrate.api.nvidia.com/v1",
    temperature=1.0,
    max_tokens=4096,
    additional_kwargs={"top_p": 1}
)
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

reranker = SentenceTransformerRerank(model="BAAI/bge-reranker-base", top_n=5)
hyde = HyDEQueryTransform(include_original=True)

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

def build_fusion_query_engine(index):
    nodes = list(index.docstore.docs.values())
    print(f"[DEBUG] Nodes found in docstore: {len(nodes)}")
    vector_retriever = index.as_retriever(similarity_top_k=15)
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=15)
    fusion_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=15,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=False,
    )
    base_query_engine = RetrieverQueryEngine.from_args(
        fusion_retriever,
        node_postprocessors=[reranker],
        text_qa_template=qa_prompt,
        response_mode="compact",
    )
    return TransformQueryEngine(base_query_engine, query_transform=hyde)

async def run_rag_evaluation():
    print("🔄 Initializing Vector Database Context...")
    
    # 2. Connect to your existing persistent ChromaDB
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("pdf_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    loader = PyMuPDFReader()
    documents = []

    for file in os.listdir("./data"):
        if file.endswith(".pdf"):
            docs = loader.load(file_path=os.path.join("./data", file))
            for doc in docs:
                doc.metadata["source_file"] = file
            documents.extend(docs)

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        store_nodes_override=True
    )
    
    # Create the query engine using the same fusion+rerank+HyDE pipeline as app.py
    query_engine = build_fusion_query_engine(index)
    
    print(f"[DEBUG] Docstore nodes after load: {len(index.docstore.docs)}")
    if len(index.docstore.docs) == 0:
        print("[WARNING] Docstore is empty — BM25Retriever will fail. "
              "The vector store may need to be rebuilt with store_nodes_override=True "
              "at index-creation time (in app.py), not just at load time.")

    # 3. Define evaluation questions 
    eval_questions = [
        # Category A - vague/paraphrased
        "How do you make sure retrieval finds the right chunks?",
        "What's the tradeoff between fast search and accurate search?",
        "How does the app remember earlier messages in a conversation?",
        # Category B - Day N questions
        "What did you build on Day 9?",
        "What happened on Day 14?",
        "What's covered on Day 17?",
        "What did you do on Day 25?",
        # Category C - meta/self-referential
        "How do you make sure the chatbot doesn't make things up?",
        "What happens if the answer isn't in the documents?",
        "How does this chatbot decide which document to search?",
    ]
    
    # 4. Initialize Evaluators and Runner
    print(f"🧪 Setting up evaluators for {len(eval_questions)} test queries...")
    faithfulness_eval = FaithfulnessEvaluator(llm=Settings.llm)
    relevancy_eval = RelevancyEvaluator(llm=Settings.llm)
    
    runner = BatchEvalRunner(
        evaluators={
            "faithfulness": faithfulness_eval,
            "relevancy": relevancy_eval
        },
        workers=2 # Parallel execution
    )

    print("🚀 Running batch evaluation pipeline via NVIDIA LLaMA 3 Judge...")
    # 5. Execute evaluation
    eval_results = await runner.aevaluate_queries(
        query_engine,
        queries=eval_questions
    )
    
    # 6. Parse and display scores
    print("\n" + "="*50)
    print("📊 RAG EVALUATION REPORT")
    print("="*50)
    
    faith_scores = [res.score for res in eval_results["faithfulness"]]
    rel_scores = [res.score for res in eval_results["relevancy"]]
    
    avg_faith = sum(faith_scores) / len(faith_scores) if faith_scores else 0
    avg_rel = sum(rel_scores) / len(rel_scores) if rel_scores else 0
    
    print(f"✅ Average Faithfulness Score (No Hallucinations): {avg_faith:.2f} / 1.0")
    print(f"🎯 Average Relevancy Score (Directly Answers Query): {avg_rel:.2f} / 1.0")
    print("="*50)
    
    # Detail breakdown - now with full response text and feedback
    responses = eval_results["faithfulness"]  # each result also has .response and .contexts
    for i, q in enumerate(eval_questions):
        print(f"\n{'='*60}")
        print(f"[Q{i+1}]: {q}")
        print(f"{'='*60}")
        safe_response = responses[i].response.encode('ascii', errors='replace').decode('ascii')
        print(f"GENERATED ANSWER:\n{safe_response}\n")
        print(f"|- Faithfulness: {eval_results['faithfulness'][i].score}")
        print(f"|  Feedback: {eval_results['faithfulness'][i].feedback}")
        print(f"|- Relevancy:    {eval_results['relevancy'][i].score}")
        print(f"|  Feedback: {eval_results['relevancy'][i].feedback}")
        contexts = responses[i].contexts
        print(f"|- Retrieved context count: {len(contexts) if contexts else 0}")
        if contexts:
            for j, ctx in enumerate(contexts[:2]):
                safe_ctx = ctx[:200].encode('ascii', errors='replace').decode('ascii')
                print(f"     [{j}] {safe_ctx}...")

if __name__ == "__main__":
    asyncio.run(run_rag_evaluation())
