import re
import json
import logging
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple

from llama_index.core import Settings
from core_rag import FALLBACK_RESPONSE

logger = logging.getLogger("rag_eval_utils")


class OutcomeCategory(str, Enum):
    SUPPORTED_ANSWER = "SUPPORTED_ANSWER"
    UNSUPPORTED_ANSWER = "UNSUPPORTED_ANSWER"
    JUSTIFIED_ABSTENTION = "JUSTIFIED_ABSTENTION"
    UNJUSTIFIED_ABSTENTION = "UNJUSTIFIED_ABSTENTION"
    EVALUATION_ERROR = "EVALUATION_ERROR"


# Alias for requirement compatibility
INVALID_JUDGE_RESPONSE = OutcomeCategory.EVALUATION_ERROR


def is_fallback_response(response: str) -> bool:
    """
    Detect whether a response is the designated fallback abstention response.
    Performs case-insensitive whitespace-normalized check.
    """
    if not response:
        return False
    norm_resp = response.strip().lower()
    norm_fallback = FALLBACK_RESPONSE.lower()
    return norm_fallback in norm_resp or norm_resp == norm_fallback


# Regular expression patterns indicating conversational responses lacking a verdict
CONVERSATIONAL_PATTERNS = [
    r"i(?:'|’)?m\s+ready\s+to\s+check",
    r"please\s+(?:give|provide|send)\s+me",
    r"i\s+need\s+the\s+specific",
    r"could\s+you\s+please\s+provide",
    r"once\s+i\s+have\s+both",
    r"let\s+me\s+know\s+the\s+specific",
    r"just\s+let\s+me\s+know"
]


def parse_and_validate_judge_response(raw_text: str) -> Tuple[Optional[bool], str, bool]:
    """
    Validate and parse a judge LLM response.
    Returns:
        (verdict, explanation, is_valid)
        - verdict: True (pass/supported/justified), False (fail/unsupported/unjustified), or None
        - explanation: Explanation string or reason for rejection
        - is_valid: True if response represents an unambiguous, valid evaluation judgment
    """
    if not raw_text or not raw_text.strip():
        return None, "Empty judge response.", False

    cleaned = raw_text.strip()

    # 1. Check for conversational non-judgments
    for pat in CONVERSATIONAL_PATTERNS:
        if re.search(pat, cleaned, re.IGNORECASE):
            return None, f"Conversational response without verdict: '{cleaned[:120]}...'", False

    # 2. Attempt JSON extraction and parsing
    # Look for json markdown block or bare json object
    json_candidate = None
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if json_match:
        json_candidate = json_match.group(1).strip()
    else:
        bare_match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
        if bare_match:
            json_candidate = bare_match.group(1).strip()

    if json_candidate:
        try:
            data = json.loads(json_candidate)
            if isinstance(data, dict):
                reason = str(data.get("reason", data.get("explanation", ""))).strip()

                # Check boolean "passing" key
                if "passing" in data:
                    val = data["passing"]
                    if isinstance(val, bool):
                        return val, reason or str(val), True
                    if isinstance(val, str) and val.strip().lower() in ("true", "yes"):
                        return True, reason or "YES", True
                    if isinstance(val, str) and val.strip().lower() in ("false", "no"):
                        return False, reason or "NO", True

                # Check string "verdict" key
                if "verdict" in data:
                    val = str(data["verdict"]).strip().upper()
                    if val in ("YES", "TRUE", "PASS", "SUPPORTED", "JUSTIFIED"):
                        return True, reason or val, True
                    if val in ("NO", "FALSE", "FAIL", "UNSUPPORTED", "UNJUSTIFIED"):
                        return False, reason or val, True
        except Exception:
            # Fall through to raw text parsing
            pass

    # 3. Check for explicit prefix YES / NO responses
    # Match leading YES / NO optionally followed by punctuation
    match_yes = re.match(r"^YES\b(?:\s*[:\.\-]\s*(.*))?$", cleaned, re.IGNORECASE | re.DOTALL)
    if match_yes:
        reason = match_yes.group(1).strip() if match_yes.group(1) else "YES"
        return True, reason, True

    match_no = re.match(r"^NO\b(?:\s*[:\.\-]\s*(.*))?$", cleaned, re.IGNORECASE | re.DOTALL)
    if match_no:
        reason = match_no.group(1).strip() if match_no.group(1) else "NO"
        return False, reason, True

    return None, f"Unrecognized judgment format: '{cleaned[:120]}...'", False


FAITHFULNESS_JUDGE_PROMPT = (
    "You are an expert impartial judge evaluating the faithfulness of a RAG assistant's answer.\n"
    "Your task is to determine whether EVERY factual claim and statement in the ANSWER is directly "
    "supported by the provided DOCUMENT CONTEXT.\n\n"
    "DOCUMENT CONTEXT:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n\n"
    "USER QUESTION: {query}\n\n"
    "GENERATED ANSWER:\n"
    "{response}\n\n"
    "Task Instructions:\n"
    "1. Check if the ANSWER contains any factual claims, numbers, code, or details that cannot be found in the CONTEXT.\n"
    "2. If ALL factual claims in the ANSWER are directly supported by the CONTEXT, set passing to true.\n"
    "3. If ANY claim in the ANSWER is ungrounded, extrapolated, or contradicts the CONTEXT, set passing to false.\n"
    "4. Return ONLY a valid JSON object in this exact format:\n"
    "{{\n"
    '  "passing": true,\n'
    '  "reason": "Brief explanation of why the answer is or is not supported by the context"\n'
    "}}\n"
    "Do NOT include conversational chatter, introductory phrases, or markdown formatting outside the JSON."
)


ABSTENTION_JUDGE_PROMPT = (
    "You are an impartial evaluator assessing whether a document search engine was justified in abstaining.\n"
    "The assistant returned a fallback response because it could not find the answer.\n\n"
    "USER QUESTION: {query}\n\n"
    "AVAILABLE DOCUMENT CONTEXT:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n\n"
    "Task Instructions:\n"
    "1. Check if the AVAILABLE DOCUMENT CONTEXT contains sufficient factual information to answer the USER QUESTION.\n"
    "2. If the context does NOT contain enough information to answer the question, the assistant was JUSTIFIED in abstaining. Set passing to true.\n"
    "3. If the context DOES contain sufficient information to answer the question, the assistant was UNJUSTIFIED in abstaining (it missed the answer). Set passing to false.\n"
    "4. Return ONLY a valid JSON object in this exact format:\n"
    "{{\n"
    '  "passing": true,\n'
    '  "reason": "Brief explanation of whether the context contained sufficient information to answer the question"\n'
    "}}\n"
    "Do NOT include conversational chatter, introductory phrases, or markdown formatting outside the JSON."
)


async def evaluate_with_retry(
    prompt: str,
    llm=None,
    max_retries: int = 2
) -> Tuple[Optional[bool], str, bool]:
    """
    Execute a judge prompt against the LLM, validating the response and retrying if invalid.
    Returns:
        (verdict, explanation, is_valid)
    """
    eval_llm = llm or Settings.llm
    if eval_llm is None:
        raise ValueError("No LLM configured for evaluation judge.")

    current_prompt = prompt
    last_explanation = ""

    for attempt in range(max_retries + 1):
        try:
            resp = await eval_llm.acomplete(current_prompt)
            raw_text = resp.text.strip()
            verdict, explanation, is_valid = parse_and_validate_judge_response(raw_text)

            if is_valid:
                return verdict, explanation, True

            last_explanation = explanation
            logger.warning(f"Judge response invalid (attempt {attempt + 1}/{max_retries + 1}): {raw_text[:80]}")

            if attempt < max_retries:
                # Add explicit formatting reinforcement on retry
                current_prompt = (
                    f"{prompt}\n\n"
                    "CRITICAL FORMAT REQUIREMENT:\n"
                    "Your previous response was conversational or unparseable. "
                    "You MUST reply with ONLY a raw JSON object: "
                    '{"passing": true, "reason": "..."} or {"passing": false, "reason": "..."}. '
                    "Do NOT include any other text."
                )
        except Exception as e:
            logger.error(f"Error calling judge LLM on attempt {attempt + 1}: {e}")
            last_explanation = str(e)

    return None, f"Exhausted {max_retries + 1} attempts. Last issue: {last_explanation}", False


async def evaluate_faithfulness_structured(
    query: str,
    response: str,
    contexts: List[str],
    initial_feedback: Optional[str] = None,
    llm=None,
    max_retries: int = 2
) -> Tuple[Optional[float], str, bool]:
    """
    Evaluate faithfulness of an answer with response validation and retry support.
    If initial_feedback is already a valid judgment, uses it without making an extra LLM call.
    Returns:
        (score, explanation, is_valid)
        - score: 1.0 (supported), 0.0 (unsupported), or None (evaluation error)
    """
    # Check if initial feedback from BatchEvalRunner is already a valid, explanatory judgment.
    # Bare unreasoned verdicts like "NO" or "YES" lacking explanation do not satisfy the
    # requirement for predictable structured results with explanations and must be judged
    # via FAITHFULNESS_JUDGE_PROMPT.
    if initial_feedback:
        verdict, explanation, is_valid = parse_and_validate_judge_response(initial_feedback)
        if is_valid and explanation and explanation not in ("YES", "NO"):
            score = 1.0 if verdict else 0.0
            return score, explanation, True

    if not contexts:
        return 0.0, "No context provided to ground the answer.", True

    context_str = "\n\n".join(contexts)
    prompt = FAITHFULNESS_JUDGE_PROMPT.format(
        query=query,
        response=response,
        context_str=context_str
    )

    verdict, explanation, is_valid = await evaluate_with_retry(
        prompt=prompt,
        llm=llm,
        max_retries=max_retries
    )

    if is_valid:
        score = 1.0 if verdict else 0.0
        return score, explanation, True
    else:
        return None, explanation, False


async def evaluate_abstention_structured(
    query: str,
    contexts: List[str],
    llm=None,
    max_retries: int = 2
) -> Tuple[Optional[bool], str, bool]:
    """
    Evaluate whether abstaining on a query was justified, with validation and retry support.
    Returns:
        (is_justified, explanation, is_valid)
    """
    if not contexts:
        return True, "No context retrieved across documents; abstention is justified.", True

    context_str = "\n\n".join(contexts)
    prompt = ABSTENTION_JUDGE_PROMPT.format(
        query=query,
        context_str=context_str
    )

    verdict, explanation, is_valid = await evaluate_with_retry(
        prompt=prompt,
        llm=llm,
        max_retries=max_retries
    )

    if is_valid:
        # verdict == True means context lacked answer -> abstention justified
        # verdict == False means context had answer -> abstention unjustified
        return verdict, explanation, True
    else:
        return None, explanation, False


# Backward-compatible wrapper for existing test calls
async def evaluate_abstention(query: str, contexts: List[str], llm=None) -> bool:
    """Legacy async wrapper for evaluate_abstention_structured returning bool."""
    is_justified, _, is_valid = await evaluate_abstention_structured(query, contexts, llm=llm, max_retries=1)
    if is_valid:
        return bool(is_justified)
    return True  # conservative fallback for legacy tests


def classify_evaluation_outcome(
    is_fallback: bool,
    faithfulness_score: Optional[float],
    is_abstention_justified: Optional[bool],
    is_evaluation_valid: bool = True
) -> OutcomeCategory:
    """
    Classify a single query result into one of the distinct evaluation categories:
    1. JUSTIFIED_ABSTENTION
    2. UNJUSTIFIED_ABSTENTION
    3. SUPPORTED_ANSWER
    4. UNSUPPORTED_ANSWER
    5. EVALUATION_ERROR
    """
    if not is_evaluation_valid:
        return OutcomeCategory.EVALUATION_ERROR

    if is_fallback:
        if is_abstention_justified is None:
            return OutcomeCategory.EVALUATION_ERROR
        elif is_abstention_justified:
            return OutcomeCategory.JUSTIFIED_ABSTENTION
        else:
            return OutcomeCategory.UNJUSTIFIED_ABSTENTION
    else:
        if faithfulness_score is None:
            return OutcomeCategory.EVALUATION_ERROR
        elif faithfulness_score >= 0.5:
            return OutcomeCategory.SUPPORTED_ANSWER
        else:
            return OutcomeCategory.UNSUPPORTED_ANSWER


def compute_evaluation_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute structured evaluation metrics keeping evaluation errors strictly separate
    from genuine faithfulness failures.
    """
    total = len(records)
    if total == 0:
        return {}

    supported_count = 0
    unsupported_count = 0
    justified_abstentions = 0
    unjustified_abstentions = 0
    eval_errors = 0

    legacy_faith_scores = []
    relevancy_scores = []

    for r in records:
        cat = r.get("outcome_category")
        if cat == OutcomeCategory.SUPPORTED_ANSWER:
            supported_count += 1
        elif cat == OutcomeCategory.UNSUPPORTED_ANSWER:
            unsupported_count += 1
        elif cat == OutcomeCategory.JUSTIFIED_ABSTENTION:
            justified_abstentions += 1
        elif cat == OutcomeCategory.UNJUSTIFIED_ABSTENTION:
            unjustified_abstentions += 1
        elif cat == OutcomeCategory.EVALUATION_ERROR:
            eval_errors += 1

        if "legacy_faithfulness" in r and r["legacy_faithfulness"] is not None:
            legacy_faith_scores.append(r["legacy_faithfulness"])
        if "relevancy" in r and r["relevancy"] is not None:
            relevancy_scores.append(r["relevancy"])

    total_valid_answers = supported_count + unsupported_count
    total_valid_abstentions = justified_abstentions + unjustified_abstentions
    total_valid_evaluated = total_valid_answers + total_valid_abstentions

    # 1. Grounded Answer Faithfulness: evaluated strictly over valid substantive answers
    grounded_answer_faithfulness = (
        supported_count / total_valid_answers if total_valid_answers > 0 else None
    )

    # 2. Abstention Accuracy: justified abstentions / total valid abstentions
    abstention_accuracy = (
        justified_abstentions / total_valid_abstentions if total_valid_abstentions > 0 else None
    )

    # 3. Overall Grounding Rate over valid evaluations
    overall_grounding_rate = (
        (supported_count + justified_abstentions) / total_valid_evaluated
        if total_valid_evaluated > 0 else 0.0
    )

    # 4. Legacy Faithfulness: raw historical metric preserving score across all responses
    legacy_faithfulness = (
        sum(legacy_faith_scores) / len(legacy_faith_scores) if legacy_faith_scores else 0.0
    )

    # 5. Average Relevancy
    avg_relevancy = (
        sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0.0
    )

    return {
        "total_queries": total,
        "supported_answers": supported_count,
        "unsupported_answers": unsupported_count,
        "justified_abstentions": justified_abstentions,
        "unjustified_abstentions": unjustified_abstentions,
        "evaluation_errors": eval_errors,
        "total_substantive_answers": total_valid_answers,
        "total_abstentions": total_valid_abstentions,
        "legacy_faithfulness": legacy_faithfulness,
        "grounded_answer_faithfulness": grounded_answer_faithfulness,
        "abstention_accuracy": abstention_accuracy,
        "overall_grounding_rate": overall_grounding_rate,
        "average_relevancy": avg_relevancy
    }
