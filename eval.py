import os
import sys
import asyncio
import logging
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


from llama_index.core import Settings
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    BatchEvalRunner
)

from core_rag import (
    RAGIndexManager,
    configure_settings,
    build_fusion_query_engine,
    NVIDIA_MODEL,
    FALLBACK_RESPONSE,
    APPLICATION_ARCHITECTURE_INFO,
    is_architecture_or_meta_query
)
from eval_utils import (
    is_fallback_response,
    evaluate_abstention_structured,
    evaluate_faithfulness_structured,
    classify_evaluation_outcome,
    compute_evaluation_metrics,
    OutcomeCategory
)
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("rag_pdf_chatbot_eval")

# 1. Load environment and configurations
load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise ValueError("NVIDIA_API_KEY not found in .env file. Please set NVIDIA_API_KEY in your .env.")

# Configure LLM and embedding settings matching application parameters
configure_settings(api_key=NVIDIA_API_KEY)


async def run_rag_evaluation():
    print("🔄 Initializing Vector Database Context from persistent storage...")

    # Connect to the persistent index manager without duplicating or corrupting data
    manager = RAGIndexManager()
    index = manager.sync_index()

    if index is None or not manager.get_indexed_files():
        print("❌ No indexed documents found. Please upload at least one PDF via app.py or place a PDF in ./data.")
        return

    print(f"📑 Indexed documents detected: {manager.get_indexed_files()}")
    print(f"📊 Total nodes in docstore: {len(index.docstore.docs)}")

    # Create the query engine using the same fusion+rerank+HyDE pipeline as app.py
    query_engine = build_fusion_query_engine(index)

    # 3. Define evaluation questions (Category A, B, C)
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
        workers=1  # Sequential execution to prevent request contention and rate limits
    )

    print(f"🚀 Running batch evaluation pipeline via NVIDIA LLM ({NVIDIA_MODEL}) Judge...")
    # 5. Execute evaluation
    eval_results = await runner.aevaluate_queries(
        query_engine,
        queries=eval_questions
    )

    records = []
    responses = eval_results["faithfulness"]

    print("\n🔍 Validating judge responses and executing structured evaluations...")
    for i, q in enumerate(eval_questions):
        print(f"  [{i + 1}/{len(eval_questions)}] Evaluating query: '{q[:50]}...'")
        resp_obj = responses[i]
        answer_text = resp_obj.response
        contexts = resp_obj.contexts or []
        legacy_faith = eval_results["faithfulness"][i].score
        rel_score = eval_results["relevancy"][i].score
        raw_feedback = eval_results["faithfulness"][i].feedback

        fallback = is_fallback_response(answer_text)

        if fallback:
            is_justified, judge_reason, is_valid = await evaluate_abstention_structured(
                query=q,
                contexts=contexts,
                llm=Settings.llm
            )
            structured_faith_score = None
        else:
            is_justified = None
            eval_contexts = contexts
            if is_architecture_or_meta_query(q):
                eval_contexts = [APPLICATION_ARCHITECTURE_INFO] + contexts

            structured_faith_score, judge_reason, is_valid = await evaluate_faithfulness_structured(
                query=q,
                response=answer_text,
                contexts=eval_contexts,
                initial_feedback=raw_feedback if not is_architecture_or_meta_query(q) else None,
                llm=Settings.llm
            )

        outcome_cat = classify_evaluation_outcome(
            is_fallback=fallback,
            faithfulness_score=structured_faith_score,
            is_abstention_justified=is_justified,
            is_evaluation_valid=is_valid
        )

        records.append({
            "id": f"Q{i + 1}",
            "query": q,
            "answer": answer_text,
            "is_fallback": fallback,
            "is_abstention_justified": is_justified,
            "outcome_category": outcome_cat,
            "legacy_faithfulness": legacy_faith,
            "structured_faithfulness": structured_faith_score,
            "judge_explanation": judge_reason,
            "is_evaluation_valid": is_valid,
            "raw_feedback": raw_feedback,
            "relevancy": rel_score,
            "relevancy_feedback": eval_results["relevancy"][i].feedback,
            "contexts": contexts
        })

        # Save incremental checkpoint
        try:
            os.makedirs("evaluation_results", exist_ok=True)
            checkpoint_path = os.path.join("evaluation_results", ".eval_checkpoint.json")
            with open(checkpoint_path, "w", encoding="utf-8") as cp_f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "completed_count": len(records),
                    "total_count": len(eval_questions),
                    "records": [
                        {k: (v.value if hasattr(v, "value") else v) for k, v in rec.items()}
                        for rec in records
                    ]
                }, cp_f, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))
        except Exception as cp_err:
            logger.warning(f"Could not save checkpoint: {cp_err}")

    metrics = compute_evaluation_metrics(records)

    # 6. Parse and display scores
    print("\n" + "=" * 65)
    print("📊 RAG EVALUATION & GROUNDING REPORT (STRUCTURED JUDGE)")
    print("=" * 65)
    print(f"Total Evaluated Queries: {metrics['total_queries']}")
    print(f"|- Substantive Answers (Valid):  {metrics['total_substantive_answers']}")
    print(f"|   |- Supported Answers:        {metrics['supported_answers']}")
    print(f"|   '- Unsupported Answers:      {metrics['unsupported_answers']}")
    print(f"|- Abstentions (Valid):          {metrics['total_abstentions']}")
    print(f"|   |- Justified Abstentions:    {metrics['justified_abstentions']}")
    print(f"|   '- Unjustified Abstentions:  {metrics['unjustified_abstentions']}")
    print(f"'- Evaluation Errors / Invalid:  {metrics['evaluation_errors']}")
    print("-" * 65)
    print("📈 METRICS SUMMARY:")
    print(f"1. Legacy Faithfulness (Raw historical metric):   {metrics['legacy_faithfulness']:.2f} / 1.0")
    g_faith = f"{metrics['grounded_answer_faithfulness']:.2f} / 1.0" if metrics['grounded_answer_faithfulness'] is not None else "N/A"
    print(f"2. Grounded Answer Faithfulness (Non-fallback):   {g_faith}")
    abs_acc = f"{metrics['abstention_accuracy']:.2f} / 1.0" if metrics['abstention_accuracy'] is not None else "N/A"
    print(f"3. Abstention Accuracy (Correct Refusals):        {abs_acc}")
    print(f"4. Overall Grounding Rate (Supported + Justified): {metrics['overall_grounding_rate']:.2f} / 1.0")
    print(f"5. Average Relevancy:                             {metrics['average_relevancy']:.2f} / 1.0")
    print("=" * 65)

    # Detail breakdown
    for r in records:
        print(f"\n[{r['id']}]: {r['query']}")
        print(f"Outcome: {r['outcome_category'].value}")
        safe_response = r['answer'].encode('ascii', errors='replace').decode('ascii')
        print(f"GENERATED ANSWER:\n{safe_response}\n")
        print(f"|- Structured Verdict / Explanation: {r['judge_explanation']}")
        print(f"|- Legacy Faithfulness:              {r['legacy_faithfulness']} (Raw: {r['raw_feedback']})")
        print(f"|- Relevancy:                        {r['relevancy']} (Feedback: {r['relevancy_feedback']})")
        if r['is_fallback']:
            just_str = "JUSTIFIED (Context lacked answer)" if r['is_abstention_justified'] else "UNJUSTIFIED (Context had answer)"
            print(f"|- Abstention Status:                {just_str}")
        print(f"|- Retrieved Contexts:               {len(r['contexts'])} chunks")
        if r['contexts']:
            safe_ctx = r['contexts'][0][:180].encode('ascii', errors='replace').decode('ascii')
            print(f"     [Top chunk] {safe_ctx}...")

    # Save results to evaluation_results folder
    os.makedirs("evaluation_results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join("evaluation_results", f"phase3_evaluation_{ts}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": metrics,
            "records": [
                {k: (v.value if hasattr(v, "value") else v) for k, v in rec.items()}
                for rec in records
            ]
        }, f, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))
    print(f"\n💾 Saved full Phase 3 evaluation trace to: {output_path}")

    # Remove temporary checkpoint on successful completion
    checkpoint_path = os.path.join("evaluation_results", ".eval_checkpoint.json")
    if os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(run_rag_evaluation())
