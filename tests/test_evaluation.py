import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from eval_utils import (
    is_fallback_response,
    parse_and_validate_judge_response,
    evaluate_abstention,
    evaluate_faithfulness_structured,
    evaluate_abstention_structured,
    classify_evaluation_outcome,
    compute_evaluation_metrics,
    OutcomeCategory,
    INVALID_JUDGE_RESPONSE
)
from core_rag import FALLBACK_RESPONSE


def test_fallback_detection():
    """Verify that designated fallback responses are reliably detected."""
    # Exact match
    assert is_fallback_response(FALLBACK_RESPONSE) is True
    # Extra whitespace and lower case
    assert is_fallback_response("  i could not find that information across the uploaded documents. \n") is True
    # Substantive answers should NOT be detected as fallback
    assert is_fallback_response("Reranking uses a cross-encoder to balance speed and accuracy.") is False
    assert is_fallback_response("") is False
    assert is_fallback_response(None) is False


def test_parse_and_validate_valid_json_verdicts():
    """Verify that clean JSON with boolean passing or string verdict is parsed correctly."""
    # 1. Standard boolean passing
    raw_1 = '{"passing": true, "reason": "All facts match context."}'
    verdict, reason, is_valid = parse_and_validate_judge_response(raw_1)
    assert is_valid is True
    assert verdict is True
    assert "All facts match" in reason

    # 2. Boolean passing False
    raw_2 = '{"passing": false, "reason": "Claims were not found in text."}'
    verdict, reason, is_valid = parse_and_validate_judge_response(raw_2)
    assert is_valid is True
    assert verdict is False
    assert "not found" in reason

    # 3. String verdict YES / NO
    raw_3 = '{"verdict": "YES", "reason": "Grounded."}'
    verdict, reason, is_valid = parse_and_validate_judge_response(raw_3)
    assert is_valid is True
    assert verdict is True

    # 4. JSON wrapped in markdown code blocks
    raw_4 = '```json\n{"passing": true, "reason": "Code snippet matches."}\n```'
    verdict, reason, is_valid = parse_and_validate_judge_response(raw_4)
    assert is_valid is True
    assert verdict is True


def test_parse_and_validate_valid_yes_no_text():
    """Verify that explicit YES or NO responses with explanations are recognized."""
    verdict, reason, is_valid = parse_and_validate_judge_response("YES")
    assert is_valid is True
    assert verdict is True

    verdict, reason, is_valid = parse_and_validate_judge_response("NO")
    assert is_valid is True
    assert verdict is False

    verdict, reason, is_valid = parse_and_validate_judge_response("YES: The statement accurately reflects Day 14.")
    assert is_valid is True
    assert verdict is True
    assert "accurately reflects" in reason

    verdict, reason, is_valid = parse_and_validate_judge_response("NO - Extrapolated beyond context.")
    assert is_valid is True
    assert verdict is False
    assert "Extrapolated" in reason


def test_parse_and_validate_malformed_and_conversational():
    """
    Verify that conversational responses lacking a verdict (such as those observed
    in Phase 3 on Q1 and Q3) are identified as invalid.
    """
    # Phase 3 Q1 actual output
    q1_msg = "I’m ready to check a statement for you. Please give me the specific “Information” you’d like verified against the context."
    verdict, reason, is_valid = parse_and_validate_judge_response(q1_msg)
    assert is_valid is False
    assert verdict is None
    assert "Conversational response" in reason

    # Phase 3 Q3 actual output
    q3_msg = "I need the specific “Information” and the corresponding “Context” to determine whether the statement is supported. Could you please provide those?"
    verdict, reason, is_valid = parse_and_validate_judge_response(q3_msg)
    assert is_valid is False
    assert verdict is None

    # Empty string and whitespace
    verdict, reason, is_valid = parse_and_validate_judge_response("   \n  ")
    assert is_valid is False
    assert verdict is None

    # Vague text without clear verdict
    verdict, reason, is_valid = parse_and_validate_judge_response("The document discusses RAG architecture.")
    assert is_valid is False
    assert verdict is None


def test_evaluate_faithfulness_retry_success():
    """Verify retry behavior: invalid response on first attempt, valid response on retry."""
    mock_llm = MagicMock()
    # First attempt: conversational; Second attempt: valid JSON
    mock_llm.acomplete = AsyncMock(side_effect=[
        MagicMock(text="I'm ready to check a statement for you. Please give me the specific Information..."),
        MagicMock(text='{"passing": true, "reason": "Grounded accurately in the retrieved Day 9 text."}')
    ])

    score, reason, is_valid = asyncio.run(evaluate_faithfulness_structured(
        query="What did you build on Day 9?",
        response="VectorStoreIndex was constructed on Day 9.",
        contexts=["Day 9: Build the Vector Index with ChromaDB."],
        initial_feedback=None,
        llm=mock_llm,
        max_retries=2
    ))

    assert is_valid is True
    assert score == 1.0
    assert "Grounded accurately" in reason
    assert mock_llm.acomplete.call_count == 2


def test_evaluate_faithfulness_retry_exhausted():
    """Verify exhausted retries: all attempts invalid -> returns is_valid=False and score=None."""
    mock_llm = MagicMock()
    # All attempts return conversational or unparseable messages
    mock_llm.acomplete = AsyncMock(return_value=MagicMock(
        text="Could you please provide the specific statement and context?"
    ))

    score, reason, is_valid = asyncio.run(evaluate_faithfulness_structured(
        query="What happened on Day 14?",
        response="Functions were refactored.",
        contexts=["Day 14: Refactor into clean functions."],
        initial_feedback=None,
        llm=mock_llm,
        max_retries=2
    ))

    # Must be marked invalid and score None (not 0.0 failure!)
    assert is_valid is False
    assert score is None
    assert "Exhausted" in reason
    # Total calls = 1 initial + 2 retries = 3
    assert mock_llm.acomplete.call_count == 3


def test_evaluate_faithfulness_bare_unreasoned_initial_feedback_triggers_retry():
    """
    Verify that bare unreasoned 'NO' from default evaluator does NOT short-circuit
    faithfulness evaluation; it must invoke the structured judge prompt with retries.
    """
    mock_llm = MagicMock()
    mock_llm.acomplete = AsyncMock(return_value=MagicMock(
        text='{"passing": true, "reason": "All steps are grounded in context."}'
    ))

    score, reason, is_valid = asyncio.run(evaluate_faithfulness_structured(
        query="How do you make sure retrieval finds the right chunks?",
        response="Inspect source nodes and adjust parameters.",
        contexts=["Check source_nodes first."],
        initial_feedback="NO",  # Bare unreasoned string from BatchEvalRunner
        llm=mock_llm
    ))

    assert is_valid is True
    assert score == 1.0
    assert "All steps are grounded" in reason
    assert mock_llm.acomplete.call_count == 1


def test_evaluate_faithfulness_explanatory_initial_feedback_accepted():
    """
    Verify that structured or explanatory initial feedback is accepted directly
    without extra LLM calls.
    """
    mock_llm = MagicMock()

    score, reason, is_valid = asyncio.run(evaluate_faithfulness_structured(
        query="What happened on Day 14?",
        response="Functions were refactored.",
        contexts=["Day 14: Refactor into clean functions."],
        initial_feedback='{"passing": true, "reason": "Accurately reflects Day 14 refactoring."}',
        llm=mock_llm
    ))

    assert is_valid is True
    assert score == 1.0
    assert "Accurately reflects" in reason
    assert mock_llm.acomplete.call_count == 0


def test_evaluate_abstention_empty_context():
    """Empty context must always be evaluated as justified abstention without LLM call."""
    is_justified = asyncio.run(evaluate_abstention(
        query="What did you build on Day 9?",
        contexts=[]
    ))
    assert is_justified is True


def test_evaluate_abstention_justified_with_mock_llm():
    """When judge LLM responds that context lacks answer, abstention is justified."""
    mock_llm = MagicMock()
    mock_llm.acomplete = AsyncMock(return_value=MagicMock(
        text='{"passing": true, "reason": "Document context does not mention the internal guardrail prompt."}'
    ))

    is_justified, reason, is_valid = asyncio.run(evaluate_abstention_structured(
        query="What is the internal prompt for guardrails?",
        contexts=["This document only talks about cooking recipes."],
        llm=mock_llm
    ))
    assert is_valid is True
    assert is_justified is True


def test_evaluate_abstention_unjustified_with_mock_llm():
    """When judge LLM responds that context contains answer, abstention is unjustified."""
    mock_llm = MagicMock()
    mock_llm.acomplete = AsyncMock(return_value=MagicMock(
        text='{"passing": false, "reason": "The context clearly contains the Day 9 index building instructions."}'
    ))

    is_justified, reason, is_valid = asyncio.run(evaluate_abstention_structured(
        query="What did you build on Day 9?",
        contexts=["Day 9: Build the Vector Index using VectorStoreIndex."],
        llm=mock_llm
    ))
    assert is_valid is True
    assert is_justified is False


def test_classify_evaluation_outcome():
    """Test classification into all five outcome categories including EVALUATION_ERROR."""
    # 1. Justified abstention
    cat1 = classify_evaluation_outcome(
        is_fallback=True,
        faithfulness_score=None,
        is_abstention_justified=True,
        is_evaluation_valid=True
    )
    assert cat1 == OutcomeCategory.JUSTIFIED_ABSTENTION

    # 2. Unjustified abstention
    cat2 = classify_evaluation_outcome(
        is_fallback=True,
        faithfulness_score=None,
        is_abstention_justified=False,
        is_evaluation_valid=True
    )
    assert cat2 == OutcomeCategory.UNJUSTIFIED_ABSTENTION

    # 3. Supported answer
    cat3 = classify_evaluation_outcome(
        is_fallback=False,
        faithfulness_score=1.0,
        is_abstention_justified=None,
        is_evaluation_valid=True
    )
    assert cat3 == OutcomeCategory.SUPPORTED_ANSWER

    # 4. Unsupported answer
    cat4 = classify_evaluation_outcome(
        is_fallback=False,
        faithfulness_score=0.0,
        is_abstention_justified=None,
        is_evaluation_valid=True
    )
    assert cat4 == OutcomeCategory.UNSUPPORTED_ANSWER

    # 5. Evaluation error (invalid judge response)
    cat5 = classify_evaluation_outcome(
        is_fallback=False,
        faithfulness_score=None,
        is_abstention_justified=None,
        is_evaluation_valid=False
    )
    assert cat5 == OutcomeCategory.EVALUATION_ERROR
    assert cat5 == INVALID_JUDGE_RESPONSE


def test_metric_aggregation():
    """
    Test calculation of decoupled metrics vs legacy metrics:
    - 6 Supported answers
    - 1 Unsupported answer
    - 2 Justified abstentions
    - 0 Unjustified abstentions
    - 1 Evaluation error (judge was invalid)
    Total = 10 queries
    """
    records = [
        # 6 Supported answers (legacy faith = 1.0, rel = 1.0)
        *[
            {
                "outcome_category": OutcomeCategory.SUPPORTED_ANSWER,
                "legacy_faithfulness": 1.0,
                "relevancy": 1.0
            }
            for _ in range(6)
        ],
        # 1 Unsupported answer (legacy faith = 0.0, rel = 1.0)
        {
            "outcome_category": OutcomeCategory.UNSUPPORTED_ANSWER,
            "legacy_faithfulness": 0.0,
            "relevancy": 1.0
        },
        # 2 Justified abstentions (legacy faith = 0.0, rel = 1.0)
        *[
            {
                "outcome_category": OutcomeCategory.JUSTIFIED_ABSTENTION,
                "legacy_faithfulness": 0.0,
                "relevancy": 1.0
            }
            for _ in range(2)
        ],
        # 1 Evaluation error (legacy faith = 0.0, rel = 1.0)
        {
            "outcome_category": OutcomeCategory.EVALUATION_ERROR,
            "legacy_faithfulness": 0.0,
            "relevancy": 1.0
        }
    ]

    metrics = compute_evaluation_metrics(records)

    assert metrics["total_queries"] == 10
    assert metrics["supported_answers"] == 6
    assert metrics["unsupported_answers"] == 1
    assert metrics["justified_abstentions"] == 2
    assert metrics["unjustified_abstentions"] == 0
    assert metrics["evaluation_errors"] == 1

    # Substantive answer faithfulness: 6 / (6 + 1) = 6 / 7 = 0.857
    # Note: evaluation error is NOT counted as unsupported!
    assert round(metrics["grounded_answer_faithfulness"], 3) == 0.857

    # Abstention accuracy: 2 / 2 = 1.00
    assert metrics["abstention_accuracy"] == 1.00

    # Overall grounding rate over valid evaluations: (6 + 2) / 9 = 8 / 9 = 0.889
    assert round(metrics["overall_grounding_rate"], 3) == 0.889

    # Legacy faithfulness: raw historical average across all 10 responses (6 * 1.0 + 4 * 0.0) / 10 = 0.60
    assert metrics["legacy_faithfulness"] == 0.60

    # Relevancy: 1.0
    assert metrics["average_relevancy"] == 1.00
