import pytest
from core_rag import (
    APPLICATION_ARCHITECTURE_INFO,
    QA_PROMPT_TEMPLATE,
    qa_prompt,
    get_runtime_architecture_info,
    is_architecture_or_meta_query,
    FALLBACK_RESPONSE
)


def test_runtime_architecture_facts():
    """Verify that runtime architecture facts match actual implementation."""
    info = get_runtime_architecture_info()
    assert info["pipeline_type"] == "deterministic_multi_stage_rag"
    assert info["has_react_agent"] is False
    assert info["has_query_engine_tool"] is False
    assert info["has_langgraph"] is False
    assert info["document_routing"] == "simultaneous_unified_collection_search"
    assert info["retrieval"]["reranker"] == "BAAI/bge-reranker-base_top_5"
    assert info["retrieval"]["query_transform"] == "hyde"


def test_architecture_meta_query_detection():
    """Verify that queries about this chatbot's architecture or search decision are accurately detected."""
    # Queries about the chatbot itself
    assert is_architecture_or_meta_query("How does this chatbot decide which document to search?") is True
    assert is_architecture_or_meta_query("What architecture does this app use?") is True
    assert is_architecture_or_meta_query("Does the chatbot use a ReAct agent?") is True
    assert is_architecture_or_meta_query("How does the app search across multiple files?") is True

    # Ordinary questions from uploaded documents
    assert is_architecture_or_meta_query("What did you build on Day 9?") is False
    assert is_architecture_or_meta_query("What's the tradeoff between fast search and accurate search?") is False
    assert is_architecture_or_meta_query("What happened on Day 14?") is False
    assert is_architecture_or_meta_query("How to plan your day during a dopamine detox?") is False


def test_prompt_template_contains_architecture_guardrails():
    """Verify that the generation prompt explicitly establishes the true search mechanism and non-implemented tools."""
    # Must establish simultaneous search across unified collection
    assert "simultaneously" in QA_PROMPT_TEMPLATE.lower()
    assert "single shared" in QA_PROMPT_TEMPLATE.lower() or "single unified" in QA_PROMPT_TEMPLATE.lower()

    # Must explicitly state that ReAct, QueryEngineTool, and LangGraph are NOT implemented
    assert "does not use a react agent" in QA_PROMPT_TEMPLATE.lower()
    assert "queryenginetool" in QA_PROMPT_TEMPLATE.lower()
    assert "langgraph" in QA_PROMPT_TEMPLATE.lower()

    # Must instruct the model to distinguish tutorial notes from actual runtime code
    assert "tutorial" in QA_PROMPT_TEMPLATE.lower() or "concepts taught" in QA_PROMPT_TEMPLATE.lower()


def test_architecture_query_prompt_formatting():
    """
    Simulate formatting the prompt with misleading Day 20/21 tutorial notes about ReActAgent,
    ensuring that the formatted prompt instructs the LLM to rely on runtime facts rather than
    hallucinating that the chatbot is an agent.
    """
    misleading_context = (
        "DAY 21 Combine RAG + Agent = Agentic RAG\n"
        "from llama_index.core.agent import ReActAgent\n"
        "rag_tool = QueryEngineTool.from_defaults(query_engine=query_engine)\n"
        "agent = ReActAgent.from_tools([rag_tool])\n"
        "The agent picks ONE OR MORE tools per query."
    )
    query = "How does this chatbot decide which document to search?"

    rendered = qa_prompt.format(context_str=misleading_context, query_str=query)

    # Context is preserved
    assert "QueryEngineTool" in rendered
    assert "ReActAgent" in rendered

    # Runtime ground truth is present and instructs how this chatbot actually searches
    assert "ACTUAL APPLICATION RUNTIME ARCHITECTURE FACTS" in rendered
    assert "does not decide, route, or select which document to search" in rendered.lower()
    assert "searches across all uploaded documents simultaneously" in rendered.lower()


def test_ordinary_multidocument_queries_not_restricted():
    """Verify that ordinary queries across multiple documents retain full context without restriction."""
    multi_doc_context = (
        "--- Document 1: RAG_GenAI_30Day_Revision_Notes.pdf ---\n"
        "Day 12: Similarity top-k controls how many chunks to retrieve.\n\n"
        "--- Document 2: Dopamine detox.pdf ---\n"
        "How to plan your day: write down three to five major tasks on paper."
    )
    query = "What is covered about planning your day?"

    rendered = qa_prompt.format(context_str=multi_doc_context, query_str=query)

    assert "Document 1: RAG_GenAI_30Day_Revision_Notes.pdf" in rendered
    assert "Document 2: Dopamine detox.pdf" in rendered
    assert "three to five major tasks" in rendered
    assert query in rendered


def test_unestablished_behavior_guideline():
    """Verify that unestablished behavior instructs returning the exact fallback response."""
    assert FALLBACK_RESPONSE in QA_PROMPT_TEMPLATE
    assert f'"{FALLBACK_RESPONSE}"' in QA_PROMPT_TEMPLATE


def test_fallback_scoped_to_document_context():
    """
    Regression test for Q5 unjustified abstention.

    Root cause: The old prompt's fallback guideline said "cannot be established from
    the runtime facts or context", causing the LLM to check runtime architecture facts
    for Day-N document content questions and trigger fallback when the architecture
    info was irrelevant, even though the DOCUMENT CONTEXT contained Day 14 text.

    The fix scopes the fallback to fire only when the DOCUMENT CONTEXT lacks sufficient
    information, and explicitly instructs not to use fallback when context contains
    relevant information.
    """
    # Verify fallback is scoped to DOCUMENT CONTEXT insufficiency
    assert "document context does not contain sufficient information" in QA_PROMPT_TEMPLATE.lower()
    # Verify explicit anti-premature-fallback instruction
    assert "do not use this fallback if the document context contains relevant information" in QA_PROMPT_TEMPLATE.lower()

    # Verify that Day-N questions are routed to DOCUMENT CONTEXT, not runtime facts
    assert "day n" in QA_PROMPT_TEMPLATE.lower() or "'day n'" in QA_PROMPT_TEMPLATE.lower()
    assert "document content questions" in QA_PROMPT_TEMPLATE.lower()

    # Simulate formatting with Day 14 context — the rendered prompt must contain
    # the Day 14 content AND the anti-premature-fallback instruction
    day14_context = (
        "DAY 14 Refactor into Clean Functions\n"
        "Software-engineering discipline applied to ML code — wrap the pipeline into "
        "named, reusable, documented functions instead of one giant script.\n"
        "def index_document(path: str) -> VectorStoreIndex:\n"
        "    docs = SimpleDirectoryReader(path).load_data()\n"
        "    return VectorStoreIndex.from_documents(docs, storage_context=storage_context)"
    )
    rendered = qa_prompt.format(context_str=day14_context, query_str="What happened on Day 14?")
    assert "DAY 14" in rendered
    assert "Refactor into Clean Functions" in rendered
    assert FALLBACK_RESPONSE in rendered  # fallback instruction is present but contextual


def test_prompt_prohibits_extrapolation():
    """
    Regression test for Q2 unsupported answer.

    Root cause: The old prompt instructed the LLM to answer "using ONLY facts, concepts,
    and code explicitly supported by the DOCUMENT CONTEXT" but didn't explicitly prohibit
    extrapolating with specific numbers/benchmarks not in the context. The LLM added
    unsupported details like "runs in milliseconds" and "hundreds-to-thousands of token
    operations" that weren't in the retrieved chunks.

    The fix adds explicit anti-extrapolation instructions.
    """
    prompt_lower = QA_PROMPT_TEMPLATE.lower()
    # Must prohibit introducing specific numbers not in context
    assert "do not introduce specific numbers" in prompt_lower
    assert "benchmarks" in prompt_lower
    assert "latency estimates" in prompt_lower
    # Must explicitly prohibit extrapolation
    assert "do not extrapolate beyond what the context says" in prompt_lower
