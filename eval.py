import os
import asyncio
import chromadb
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
import llama_index.llms.openai.utils as openai_utils
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

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

async def run_rag_evaluation():
    print("🔄 Initializing Vector Database Context...")
    
    # 2. Connect to your existing persistent ChromaDB
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("pdf_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Load index from existing vector store
    index = VectorStoreIndex.from_vector_store(
        vector_store, 
        storage_context=storage_context
    )
    
    # Create the query engine
    query_engine = index.as_query_engine(similarity_top_k=5)

    # 3. Define evaluation questions 
    # Change these questions to match the specific PDF you currently have uploaded!
    eval_questions = [
        "What is the main objective discussed in this document?",
        "List three key metrics or findings mentioned by the author.",
        "What are the limitations or challenges highlighted in the text?"
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
    
    # Detail breakdown
    for i, q in enumerate(eval_questions):
        print(f"\n[Q{i+1}]: {q}")
        print(f"  ├─ Faithfulness: {eval_results['faithfulness'][i].score}")
        print(f"  ├─ Relevancy:    {eval_results['relevancy'][i].score}")
        print(f"  └─ Feedback:     {eval_results['relevancy'][i].feedback}")

if __name__ == "__main__":
    asyncio.run(run_rag_evaluation())
