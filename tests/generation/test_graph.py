"""
End-to-end pipeline tests for the generation graph.

All LLM and retrieval calls are mocked — no live server or database needed.
Each test builds a real compiled LangGraph graph and invokes it with
graph.invoke(), verifying the final state reflects the expected execution path.
"""

import uuid
from datetime import date, datetime
from unittest.mock import MagicMock

import pytest

from config.loader import GenerationConfig
from generation.factory import make_initial_state
from generation.graph import build_graph
from generation.nodes import (
    make_analyze_query, make_check_hop, make_embed_queries, make_generate,
    make_hyde_expand, make_reflect, make_retrieve,
)
from generation.types import (
    GenerationResponse, GenerationResult, HopDecision, QueryPlan,
    ReflectionDecision, RetrievalTask,
)
from llm.types import LLMResponse, LLMUsage, StructuredResponse
from retrieval.types import MetadataFilter, RetrievalResult
from db.models import ChunkRecord, FilingRecord, ParentChunkRecord
from tests.conftest import VALID_GENERATION


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _usage() -> LLMUsage:
    return LLMUsage(input_tokens=10, output_tokens=20)


def _make_retrieval_result(ticker: str = "NVDA") -> RetrievalResult:
    filing = FilingRecord(
        id=uuid.uuid4(), ticker=ticker, company_name=f"{ticker} Corp",
        cik="0001045810", accession_number=f"{ticker}-acc-001",
        form_type="10-K", filing_date=date(2024, 2, 21),
        source_url="https://sec.gov/", downloaded_at=datetime(2024, 3, 1),
        fiscal_year_end=date(2024, 1, 28),
    )
    parent = ParentChunkRecord(
        id=uuid.uuid4(), filing_id=filing.id, chunk_index=0,
        section="Financial Statements",
        text=f"{ticker} total revenue was $60B.", token_count=10,
        content_hash="abc", created_at=datetime(2024, 3, 1),
    )
    chunk = ChunkRecord(
        id=uuid.uuid4(), filing_id=filing.id, parent_chunk_id=parent.id,
        chunk_index=0, section="Financial Statements", chunk_type="child",
        text=f"{ticker} revenue $60B", token_count=5,
        content_hash="def", created_at=datetime(2024, 3, 1),
    )
    return RetrievalResult(
        score=0.9, vector_score=None, keyword_score=None, reranker_score=None,
        chunk=chunk, parent_chunk=parent, filing=filing,
    )


def _make_embedder(dim: int = 1024) -> MagicMock:
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1] * dim]
    return embedder


def _make_retriever(results=None) -> MagicMock:
    retriever = MagicMock()
    retriever.retrieve.return_value = results or [_make_retrieval_result()]
    return retriever


def _make_filings_repo(tickers_in_db: list[str] | None = None) -> MagicMock:
    repo = MagicMock()
    if tickers_in_db is None:
        repo.list_ids.return_value = [uuid.uuid4()]
    else:
        tickers_upper = {t.upper() for t in tickers_in_db}
        repo.list_ids.side_effect = lambda filters: (
            [uuid.uuid4()] if filters.get("ticker") in tickers_upper else []
        )
    return repo


def _build_graph(llm, retriever, embedder, config: GenerationConfig, filings_repo=None):
    return build_graph(
        analyze_query_fn=make_analyze_query(llm, "Analyze the query.", filings_repo or _make_filings_repo()),
        hyde_expand_fn=make_hyde_expand(llm, "Write a hypothetical passage."),
        embed_queries_fn=make_embed_queries(embedder),
        retrieve_fn=make_retrieve(retriever, embedder),
        generate_fn=make_generate(llm, "Answer the question.", "Compare the companies."),
        check_hop_fn=make_check_hop(llm, config, "Decide if more retrieval is needed."),
        reflect_fn=make_reflect(llm, config, "Evaluate the answer quality."),
        config=config,
    )


def _gen_config(**overrides) -> GenerationConfig:
    data = {**VALID_GENERATION, **overrides}
    return GenerationConfig(**data)


def _make_query_plan(query_type="single", resolved_query="What was NVDA's revenue in 2024?", tasks=None) -> QueryPlan:
    return QueryPlan(
        reasoning="Test reasoning.",
        query_type=query_type,
        resolved_query=resolved_query,
        tasks=tasks if tasks is not None else [RetrievalTask(keyword_query="NVDA revenue 2024", semantic_query="What was NVIDIA's revenue in FY2024?")],
    )


# ---------------------------------------------------------------------------
# Happy path — single query, no HyDE, no reflection
# ---------------------------------------------------------------------------

def test_single_query_produces_answer():
    """analyze → retrieve → generate → END"""
    config = _gen_config(
        hyde={"enabled": False},
        reflection={"enabled": False, "max_iterations": 2},
    )
    llm = MagicMock()

    def chat_structured_side_effect(messages, schema, **kwargs):
        if schema == QueryPlan:
            return StructuredResponse(parsed=_make_query_plan(), usage=_usage())
        if schema == GenerationResponse:
            return StructuredResponse(
                parsed=GenerationResponse(answer="Revenue was $60.9B.", cited_indices=[1]),
                usage=_usage(),
            )
        raise ValueError(f"Unexpected schema: {schema}")

    llm.chat_structured.side_effect = chat_structured_side_effect

    graph = _build_graph(llm, _make_retriever(), _make_embedder(), config)
    final = graph.invoke(make_initial_state("What was NVDA's revenue in 2024?"))

    assert isinstance(final["answer"], GenerationResult)
    assert final["answer"].answer == "Revenue was $60.9B."
    assert len(final["answer"].citations) == 1


# ---------------------------------------------------------------------------
# resolved_query — stored in state and used downstream
# ---------------------------------------------------------------------------

def test_resolved_query_stored_in_state():
    """analyze sets resolved_query; downstream nodes use it instead of raw query."""
    config = _gen_config(
        hyde={"enabled": False},
        reflection={"enabled": False, "max_iterations": 2},
    )
    llm = MagicMock()

    def chat_structured_side_effect(messages, schema, **kwargs):
        if schema == QueryPlan:
            return StructuredResponse(
                parsed=_make_query_plan(resolved_query="What was NVDA's revenue in FY2024?"),
                usage=_usage(),
            )
        if schema == GenerationResponse:
            return StructuredResponse(
                parsed=GenerationResponse(answer="Revenue was $60.9B.", cited_indices=[1]),
                usage=_usage(),
            )
        raise ValueError(f"Unexpected schema: {schema}")

    llm.chat_structured.side_effect = chat_structured_side_effect

    graph = _build_graph(llm, _make_retriever(), _make_embedder(), config)
    final = graph.invoke(make_initial_state("NVDA revenue last year?"))

    assert final["resolved_query"] == "What was NVDA's revenue in FY2024?"


# ---------------------------------------------------------------------------
# Out of scope — terminates after analyze without reaching retrieve
# ---------------------------------------------------------------------------

def test_out_of_scope_query_terminates_without_answer():
    """analyze (out_of_scope) → END with canned answer; no retrieval"""
    config = _gen_config(
        hyde={"enabled": False},
        reflection={"enabled": False, "max_iterations": 2},
    )
    llm = MagicMock()
    llm.chat_structured.return_value = StructuredResponse(
        parsed=QueryPlan(
            reasoning="Not a 10-K question.",
            query_type="out_of_scope",
            resolved_query="What is the weather today?",
            tasks=[],
        ),
        usage=_usage(),
    )

    graph = _build_graph(llm, _make_retriever(), _make_embedder(), config)
    final = graph.invoke(make_initial_state("What is the weather today?"))

    assert isinstance(final["answer"], GenerationResult)
    assert "can't be answered from SEC 10-K filings" in final["answer"].answer
    assert llm.chat_structured.call_count == 1
    llm.chat.assert_not_called()


# ---------------------------------------------------------------------------
# Empty tasks — terminates after analyze with canned answer
# ---------------------------------------------------------------------------

def test_empty_tasks_terminates_without_retrieval():
    """analyze (no tasks) → END with canned answer"""
    config = _gen_config(
        hyde={"enabled": False},
        reflection={"enabled": False, "max_iterations": 2},
    )
    llm = MagicMock()
    llm.chat_structured.return_value = StructuredResponse(
        parsed=QueryPlan(
            reasoning="Could not plan tasks.",
            query_type="single",
            resolved_query="What was NVDA's revenue?",
            tasks=[],
        ),
        usage=_usage(),
    )

    graph = _build_graph(llm, _make_retriever(), _make_embedder(), config)
    final = graph.invoke(make_initial_state("What was NVDA's revenue?"))

    assert isinstance(final["answer"], GenerationResult)
    assert "trouble understanding" in final["answer"].answer
    llm.chat.assert_not_called()


def test_empty_tasks_with_hyde_skips_hyde_and_returns_canned_answer():
    """analyze (no tasks) → END — guard fires before HyDE when tasks are empty"""
    config = _gen_config(
        hyde={"enabled": True},
        reflection={"enabled": False, "max_iterations": 2},
    )
    llm = MagicMock()
    llm.chat_structured.return_value = StructuredResponse(
        parsed=QueryPlan(
            reasoning="Could not plan tasks.",
            query_type="single",
            resolved_query="What was NVDA's revenue?",
            tasks=[],
        ),
        usage=_usage(),
    )

    graph = _build_graph(llm, _make_retriever(), _make_embedder(), config)
    final = graph.invoke(make_initial_state("What was NVDA's revenue?"))

    assert isinstance(final["answer"], GenerationResult)
    assert "trouble understanding" in final["answer"].answer
    llm.chat.assert_not_called()


# ---------------------------------------------------------------------------
# Ticker not indexed — short-circuits in analyze with canned answer
# ---------------------------------------------------------------------------

def test_all_tickers_missing_returns_canned_answer():
    """analyze (ticker not in DB) → END with canned answer, no HyDE, no retrieval"""
    config = _gen_config(
        hyde={"enabled": True},
        reflection={"enabled": False, "max_iterations": 2},
    )
    llm = MagicMock()
    llm.chat_structured.return_value = StructuredResponse(
        parsed=QueryPlan(
            reasoning="Single company lookup.",
            query_type="single",
            resolved_query="What was FAKE's revenue?",
            tasks=[RetrievalTask(keyword_query="FAKE revenue", semantic_query="What was FAKE's revenue?", filter=MetadataFilter(ticker="FAKE"))],
        ),
        usage=_usage(),
    )

    graph = _build_graph(
        llm, _make_retriever(), _make_embedder(), config,
        filings_repo=_make_filings_repo(tickers_in_db=[]),
    )
    final = graph.invoke(make_initial_state("What was FAKE's revenue?"))

    assert isinstance(final["answer"], GenerationResult)
    assert "FAKE" in final["answer"].answer
    assert "don't have filings" in final["answer"].answer
    llm.chat.assert_not_called()


# ---------------------------------------------------------------------------
# HyDE — per-task hypothetical passages passed to retrieve
# ---------------------------------------------------------------------------

def test_hyde_passage_is_used_for_embedding():
    """analyze → hyde_expand → retrieve (uses task.hyde_query) → generate → END"""
    config = _gen_config(
        hyde={"enabled": True},
        reflection={"enabled": False, "max_iterations": 2},
    )
    llm = MagicMock()

    def chat_structured_side_effect(messages, schema, **kwargs):
        if schema == QueryPlan:
            return StructuredResponse(parsed=_make_query_plan(), usage=_usage())
        if schema == GenerationResponse:
            return StructuredResponse(
                parsed=GenerationResponse(answer="Revenue was $60.9B.", cited_indices=[1]),
                usage=_usage(),
            )
        raise ValueError(f"Unexpected schema: {schema}")

    llm.chat_structured.side_effect = chat_structured_side_effect
    llm.chat.return_value = LLMResponse(content="Hypothetical passage.", usage=_usage())

    embedder = _make_embedder()
    graph = _build_graph(llm, _make_retriever(), embedder, config)
    graph.invoke(make_initial_state("What was NVDA's revenue?"))

    assert llm.chat.call_count == 1
    embedded_text = embedder.embed.call_args[0][0][0]
    assert embedded_text == "Hypothetical passage."


# ---------------------------------------------------------------------------
# Reflection — high quality answer goes straight to END
# ---------------------------------------------------------------------------

def test_reflection_high_quality_ends_pipeline():
    """analyze → retrieve → generate → reflect (high) → END"""
    config = _gen_config(
        hyde={"enabled": False},
        reflection={"enabled": True, "max_iterations": 2},
    )
    llm = MagicMock()

    def chat_structured_side_effect(messages, schema, **kwargs):
        if schema == QueryPlan:
            return StructuredResponse(parsed=_make_query_plan(), usage=_usage())
        if schema == GenerationResponse:
            return StructuredResponse(
                parsed=GenerationResponse(answer="Revenue was $60.9B.", cited_indices=[1]),
                usage=_usage(),
            )
        if schema == ReflectionDecision:
            return StructuredResponse(
                parsed=ReflectionDecision(quality="high", reason="complete and grounded"),
                usage=_usage(),
            )
        raise ValueError(f"Unexpected schema: {schema}")

    llm.chat_structured.side_effect = chat_structured_side_effect

    graph = _build_graph(llm, _make_retriever(), _make_embedder(), config)
    final = graph.invoke(make_initial_state("What was NVDA's revenue?"))

    assert isinstance(final["answer"], GenerationResult)
    assert final["reflection_count"] == 1


# ---------------------------------------------------------------------------
# Reflection — low quality triggers a second retrieval then ends
# ---------------------------------------------------------------------------

def test_reflection_low_quality_triggers_extra_retrieval():
    """analyze → retrieve → generate → reflect (low) → retrieve → generate → reflect (high) → END"""
    config = _gen_config(
        hyde={"enabled": False},
        reflection={"enabled": True, "max_iterations": 2},
    )
    reflection_calls = {"count": 0}
    llm = MagicMock()

    def chat_structured_side_effect(messages, schema, **kwargs):
        if schema == QueryPlan:
            return StructuredResponse(parsed=_make_query_plan(), usage=_usage())
        if schema == GenerationResponse:
            return StructuredResponse(
                parsed=GenerationResponse(answer="Answer.", cited_indices=[1]),
                usage=_usage(),
            )
        if schema == ReflectionDecision:
            reflection_calls["count"] += 1
            if reflection_calls["count"] == 1:
                return StructuredResponse(
                    parsed=ReflectionDecision(
                        quality="low",
                        reason="missing gross margin",
                        next_task=RetrievalTask(keyword_query="NVDA gross margin 2024", semantic_query="What was NVIDIA's gross margin in FY2024?"),
                    ),
                    usage=_usage(),
                )
            return StructuredResponse(
                parsed=ReflectionDecision(quality="high", reason="now complete"),
                usage=_usage(),
            )
        raise ValueError(f"Unexpected schema: {schema}")

    llm.chat_structured.side_effect = chat_structured_side_effect

    retriever = _make_retriever()
    graph = _build_graph(llm, retriever, _make_embedder(), config)
    final = graph.invoke(make_initial_state("What was NVDA's revenue and margin?"))

    assert final["reflection_count"] == 2
    assert retriever.retrieve.call_count == 2


# ---------------------------------------------------------------------------
# Comparison — two tasks fan out to two parallel retrieve calls
# ---------------------------------------------------------------------------

def test_comparison_query_fans_out_to_parallel_retrieves():
    """analyze → [retrieve#NVDA, retrieve#AMD] (parallel) → generate → END"""
    config = _gen_config(
        hyde={"enabled": False},
        reflection={"enabled": False, "max_iterations": 2},
    )
    llm = MagicMock()

    def chat_structured_side_effect(messages, schema, **kwargs):
        if schema == QueryPlan:
            return StructuredResponse(
                parsed=_make_query_plan(
                    query_type="comparison",
                    resolved_query="Compare NVDA and AMD revenue",
                    tasks=[
                        RetrievalTask(keyword_query="revenue annual", semantic_query="What was NVIDIA's annual revenue?", filter=MetadataFilter(ticker="NVDA")),
                        RetrievalTask(keyword_query="revenue annual", semantic_query="What was AMD's annual revenue?", filter=MetadataFilter(ticker="AMD")),
                    ],
                ),
                usage=_usage(),
            )
        if schema == GenerationResponse:
            return StructuredResponse(
                parsed=GenerationResponse(answer="NVDA > AMD.", cited_indices=[1, 2]),
                usage=_usage(),
            )
        raise ValueError(f"Unexpected schema: {schema}")

    llm.chat_structured.side_effect = chat_structured_side_effect

    retriever = MagicMock()
    retriever.retrieve.side_effect = [
        [_make_retrieval_result("NVDA")],
        [_make_retrieval_result("AMD")],
    ]
    embedder = _make_embedder()

    graph = _build_graph(llm, retriever, embedder, config)
    final = graph.invoke(make_initial_state("Compare NVDA and AMD revenue"))

    assert retriever.retrieve.call_count == 2
    assert len(final["completed_results"]) == 2
    assert final["answer"].answer == "NVDA > AMD."


# ---------------------------------------------------------------------------
# Hop — check_hop enabled triggers iterative retrieval
# ---------------------------------------------------------------------------

def test_hop_enabled_triggers_extra_retrieval():
    """analyze → retrieve → check_hop (not done) → retrieve → check_hop (done) → generate → END"""
    config = _gen_config(
        hyde={"enabled": False},
        hop={"enabled": True, "max_hops": 3},
        reflection={"enabled": False, "max_iterations": 2},
    )
    hop_calls = {"count": 0}
    llm = MagicMock()

    def chat_structured_side_effect(messages, schema, **kwargs):
        if schema == QueryPlan:
            return StructuredResponse(parsed=_make_query_plan(), usage=_usage())
        if schema == HopDecision:
            hop_calls["count"] += 1
            if hop_calls["count"] == 1:
                return StructuredResponse(
                    parsed=HopDecision(
                        done=False,
                        next_task=RetrievalTask(keyword_query="NVDA data center segment revenue", semantic_query="What was NVIDIA's data center segment revenue?"),
                    ),
                    usage=_usage(),
                )
            return StructuredResponse(parsed=HopDecision(done=True), usage=_usage())
        if schema == GenerationResponse:
            return StructuredResponse(
                parsed=GenerationResponse(answer="Data center drove growth.", cited_indices=[1]),
                usage=_usage(),
            )
        raise ValueError(f"Unexpected schema: {schema}")

    llm.chat_structured.side_effect = chat_structured_side_effect

    retriever = _make_retriever()
    graph = _build_graph(llm, retriever, _make_embedder(), config)
    final = graph.invoke(make_initial_state("What drove NVDA's revenue growth?"))

    assert final["hop_count"] == 2
    assert retriever.retrieve.call_count == 2
    assert isinstance(final["answer"], GenerationResult)
    assert final["answer"].answer == "Data center drove growth."


def test_hop_disabled_skips_check_hop():
    """analyze → retrieve → generate → END (check_hop never called)"""
    config = _gen_config(
        hyde={"enabled": False},
        hop={"enabled": False, "max_hops": 3},
        reflection={"enabled": False, "max_iterations": 2},
    )
    llm = MagicMock()

    def chat_structured_side_effect(messages, schema, **kwargs):
        if schema == QueryPlan:
            return StructuredResponse(parsed=_make_query_plan(), usage=_usage())
        if schema == GenerationResponse:
            return StructuredResponse(
                parsed=GenerationResponse(answer="Revenue was $60.9B.", cited_indices=[1]),
                usage=_usage(),
            )
        raise ValueError(f"Unexpected schema: {schema}")

    llm.chat_structured.side_effect = chat_structured_side_effect

    retriever = _make_retriever()
    graph = _build_graph(llm, retriever, _make_embedder(), config)
    final = graph.invoke(make_initial_state("What was NVDA's revenue?"))

    # HopDecision was never requested
    called_schemas = [call[0][1] for call in llm.chat_structured.call_args_list]
    assert HopDecision not in called_schemas
    assert isinstance(final["answer"], GenerationResult)


# ---------------------------------------------------------------------------
# Reflection exhausted — terminates after max_iterations
# ---------------------------------------------------------------------------

def test_reflection_exhausted_terminates_after_max_iterations():
    """analyze → retrieve → generate → reflect (low) × max_iterations → END"""
    config = _gen_config(
        hyde={"enabled": False},
        reflection={"enabled": True, "max_iterations": 2},
    )
    llm = MagicMock()
    generation_calls = {"count": 0}

    def chat_structured_side_effect(messages, schema, **kwargs):
        if schema == QueryPlan:
            return StructuredResponse(parsed=_make_query_plan(), usage=_usage())
        if schema == GenerationResponse:
            generation_calls["count"] += 1
            return StructuredResponse(
                parsed=GenerationResponse(answer="Partial answer.", cited_indices=[1]),
                usage=_usage(),
            )
        if schema == ReflectionDecision:
            return StructuredResponse(
                parsed=ReflectionDecision(
                    quality="low",
                    reason="still incomplete",
                    next_task=RetrievalTask(keyword_query="more data", semantic_query="What is the additional data needed?"),
                ),
                usage=_usage(),
            )
        raise ValueError(f"Unexpected schema: {schema}")

    llm.chat_structured.side_effect = chat_structured_side_effect

    retriever = _make_retriever()
    graph = _build_graph(llm, retriever, _make_embedder(), config)
    final = graph.invoke(make_initial_state("What was NVDA's revenue?"))

    assert final["reflection_count"] == config.reflection.max_iterations
    assert generation_calls["count"] == config.reflection.max_iterations
    assert isinstance(final["answer"], GenerationResult)
