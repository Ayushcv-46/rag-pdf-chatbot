import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from llama_index.core import Settings, QueryBundle
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
)

from core_rag import (
    RAGIndexManager,
    configure_settings,
    build_fusion_query_engine,
    NVIDIA_MODEL,
    qa_prompt
)
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.indices.query.query_transform import HyDEQueryTransform

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("rag_diagnostic_eval")

load_dotenv()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

EVAL_QUESTIONS = [
    # Category A - vague/paraphrased
    {"id": "Q1", "cat": "Category A (vague/paraphrased)", "q": "How do you make sure retrieval finds the right chunks?"},
    {"id": "Q2", "cat": "Category A (vague/paraphrased)", "q": "What's the tradeoff between fast search and accurate search?"},
    {"id": "Q3", "cat": "Category A (vague/paraphrased)", "q": "How does the app remember earlier messages in a conversation?"},
    # Category B - Day N questions
    {"id": "Q4", "cat": "Category B (day-specific lookup)", "q": "What did you build on Day 9?"},
    {"id": "Q5", "cat": "Category B (day-specific lookup)", "q": "What happened on Day 14?"},
    {"id": "Q6", "cat": "Category B (day-specific lookup)", "q": "What's covered on Day 17?"},
    {"id": "Q7", "cat": "Category B (day-specific lookup)", "q": "What did you do on Day 25?"},
    # Category C - meta/self-referential
    {"id": "Q8", "cat": "Category C (meta/self-referential)", "q": "How do you make sure the chatbot doesn't make things up?"},
    {"id": "Q9", "cat": "Category C (meta/self-referential)", "q": "What happens if the answer isn't in the documents?"},
    {"id": "Q10", "cat": "Category C (meta/self-referential)", "q": "How does this chatbot decide which document to search?"},
]


async def run_diagnostics():
    if not NVIDIA_API_KEY:
        print("ERROR: NVIDIA_API_KEY not configured in .env")
        return None

    configure_settings(api_key=NVIDIA_API_KEY)

    manager = RAGIndexManager()
    index = manager.sync_index()

    if index is None or not manager.get_indexed_files():
        print("ERROR: No documents indexed in ./data. Please add the target PDF before running diagnostics.")
        return None

    nodes = list(index.docstore.docs.values())
    print(f"Indexed files: {manager.get_indexed_files()}")
    print(f"Docstore node count: {len(nodes)}")

    # Construct the exact same components for intermediate step inspection
    vector_retriever = index.as_retriever(similarity_top_k=15)
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=15)
    fusion_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=15,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=False,
    )
    reranker = SentenceTransformerRerank(model="BAAI/bge-reranker-base", top_n=5)
    hyde = HyDEQueryTransform(include_original=True)
    query_engine = build_fusion_query_engine(index, reranker=reranker, hyde=hyde, qa_template=qa_prompt)

    faithfulness_eval = FaithfulnessEvaluator(llm=Settings.llm)
    relevancy_eval = RelevancyEvaluator(llm=Settings.llm)

    results_dir = "evaluation_results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    diagnostics = []

    print(f"\n🚀 Running detailed diagnostic evaluation across {len(EVAL_QUESTIONS)} queries...\n")

    for item in EVAL_QUESTIONS:
        qid = item["id"]
        cat = item["cat"]
        query_str = item["q"]
        print(f"[{qid}] Processing: '{query_str}'")

        # 1. Capture HyDE query transformation
        transformed_query = hyde(QueryBundle(query_str))
        hypothetical_doc = transformed_query.query_str

        # 2. Retrieve raw fused candidates before reranking
        raw_candidates = fusion_retriever.retrieve(transformed_query)
        raw_nodes_info = [
            {
                "rank": i + 1,
                "score": getattr(n, "score", None),
                "source_file": n.metadata.get("source_file"),
                "text": n.text.strip()
            }
            for i, n in enumerate(raw_candidates)
        ]

        # 3. Postprocess with reranker
        reranked_nodes = reranker.postprocess_nodes(raw_candidates, query_bundle=QueryBundle(query_str))
        reranked_nodes_info = [
            {
                "rank": i + 1,
                "score": getattr(n, "score", None),
                "source_file": n.metadata.get("source_file"),
                "text": n.text.strip()
            }
            for i, n in enumerate(reranked_nodes)
        ]

        # 4. Generate final answer through query engine
        response = query_engine.query(query_str)
        answer = response.response

        # 5. Run evaluators
        faith_eval = await faithfulness_eval.aevaluate(
            query=query_str,
            response=answer,
            contexts=[n.text for n in response.source_nodes]
        )

        rel_eval = await relevancy_eval.aevaluate(
            query=query_str,
            response=answer,
            contexts=[n.text for n in response.source_nodes]
        )

        # 6. Preliminary Failure Classification
        fallback_phrase = "I could not find that information across the uploaded documents."
        is_fallback = fallback_phrase.lower() in answer.lower()
        faith_score = faith_eval.score
        rel_score = rel_eval.score

        suspected_category = "Pass"
        diagnosis_reason = "Answer is supported by retrieved context and relevant."

        if faith_score < 0.5:
            # Check if answer contains facts not in reranked contexts
            contexts_combined = " ".join([n["text"] for n in reranked_nodes_info])
            if is_fallback:
                suspected_category = "E. Evaluation Failure"
                diagnosis_reason = "Model provided standard fallback, but judge marked unfaithful."
            elif not any(phrase in contexts_combined for phrase in answer.split()[:5]):
                suspected_category = "C. Generation or Grounding Failure"
                diagnosis_reason = "Model elaborated or answered from outside parametric knowledge."
            else:
                suspected_category = "C. Generation or Grounding Failure"
                diagnosis_reason = "Model added unsupported claims beyond literal context."
        elif rel_score < 0.5:
            suspected_category = "A. Retrieval Failure"
            diagnosis_reason = "Generated answer did not directly address the query."

        # Special check for Day N questions
        if "day" in query_str.lower():
            day_target = [word for word in query_str.split() if "day" in word.lower()]
            day_token = day_target[0] if day_target else "day"
            in_raw = any(day_token.lower() in c["text"].lower() for c in raw_nodes_info)
            in_reranked = any(day_token.lower() in r["text"].lower() for r in reranked_nodes_info)
            if not in_raw:
                suspected_category = "A. Retrieval Failure"
                diagnosis_reason = f"Keyword/vector search failed to retrieve chunks containing '{day_token}'."
            elif in_raw and not in_reranked:
                suspected_category = "B. Reranking Failure"
                diagnosis_reason = f"Chunks containing '{day_token}' were retrieved initially, but dropped during cross-encoder reranking."

        diag_entry = {
            "id": qid,
            "category": cat,
            "query": query_str,
            "hypothetical_doc": hypothetical_doc,
            "raw_candidate_count": len(raw_candidates),
            "raw_candidates": raw_nodes_info[:5],  # top 5
            "reranked_count": len(reranked_nodes),
            "reranked_nodes": reranked_nodes_info,
            "generated_answer": answer,
            "faithfulness": {
                "score": faith_score,
                "feedback": faith_eval.feedback
            },
            "relevancy": {
                "score": rel_score,
                "feedback": rel_eval.feedback
            },
            "suspected_failure_category": suspected_category,
            "diagnosis_reason": diagnosis_reason
        }
        diagnostics.append(diag_entry)
        print(f"    Faithfulness: {faith_score} | Relevancy: {rel_score} | Category: {suspected_category}")

    # Save JSON artifact
    json_path = os.path.join(results_dir, f"diagnostic_run_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, ensure_ascii=False, default=lambda o: float(o) if hasattr(o, "item") else str(o))

    print(f"\n✅ Diagnostic run complete. Saved raw trace to: {json_path}")

    return json_path, diagnostics


if __name__ == "__main__":
    asyncio.run(run_diagnostics())
