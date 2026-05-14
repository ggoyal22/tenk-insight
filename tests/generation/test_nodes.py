"""
Tests for generation nodes. All LLM and retriever calls are mocked.
"""

import uuid
from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pytest

from config.loader import GenerationConfig
from generation.nodes.analyze_query import make_analyze_query
from generation.nodes.check_hop import make_check_hop
from generation.nodes.generate import make_generate, _select_prompt, _filter_by_indices
from generation.nodes.hyde import make_hyde_expand
from generation.nodes.reflect import make_reflect
from generation.nodes.retrieve import make_retrieve, RetrieveInput
from generation.types import (
    Citation, GenerationResponse, GenerationResult, GenerationState,
    HopDecision, QueryPlan, ReflectionDecision, RetrievalTask,
)
from llm.base import LLMError
from llm.types import LLMResponse, LLMUsage, Message, StructuredResponse
from retrieval.types import MetadataFilter, RetrievalResult
from db.models import ChunkRecord, FilingRecord, ParentChunkRecord
from tests.conftest import VALID_GENERATION


# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

def _make_llm(chat_return=None, structured_return=None) -> MagicMock:
    llm = MagicMock()
    if chat_return is not None:
        llm.chat.return_value = chat_return
    if structured_return is not None:
        llm.chat_structured.return_value = structured_return
    return llm


def _make_usage() -> LLMUsage:
    return LLMUsage(input_tokens=10, output_tokens=20)


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


def _make_filing() -> FilingRecord:
    return FilingRecord(
        id=uuid.uuid4(),
        ticker="NVDA",
        company_name="NVIDIA Corporation",
        cik="0001045810",
        accession_number="0001045810-24-000029",
        form_type="10-K",
        filing_date=date(2024, 2, 21),
        source_url="https://www.sec.gov/Archives/edgar/data/1045810/",
        downloaded_at=datetime(2024, 3, 1),
        fiscal_year_end=date(2024, 1, 28),
    )


def _make_parent_chunk(filing_id) -> ParentChunkRecord:
    return ParentChunkRecord(
        id=uuid.uuid4(),
        filing_id=filing_id,
        chunk_index=0,
        section="Financial Statements",
        text="Total revenue was $60.9 billion for fiscal 2024.",
        token_count=12,
        content_hash="abc123",
        created_at=datetime(2024, 3, 1),
    )


def _make_chunk(filing_id, parent_id) -> ChunkRecord:
    return ChunkRecord(
        id=uuid.uuid4(),
        filing_id=filing_id,
        parent_chunk_id=parent_id,
        chunk_index=0,
        section="Financial Statements",
        chunk_type="child",
        text="Revenue $60.9B",
        token_count=4,
        content_hash="def456",
        created_at=datetime(2024, 3, 1),
    )


def _make_retrieval_result() -> RetrievalResult:
    filing = _make_filing()
    parent = _make_parent_chunk(filing.id)
    chunk = _make_chunk(filing.id, parent.id)
    return RetrievalResult(
        score=0.9, vector_score=None, keyword_score=None, reranker_score=None,
        chunk=chunk, parent_chunk=parent, filing=filing,
    )


def _base_state(**overrides) -> dict:
    state = {
        "query": "What was NVDA's revenue in 2024?",
        "history": [],
        "query_filter": None,
        "query_type": "single",
        "resolved_query": None,
        "pending_tasks": [],
        "completed_results": [],
        "hop_count": 0,
        "reflection_count": 0,
        "retrieval_triggered_by": "analysis",
        "answer": None,
    }
    state.update(overrides)
    return state


def _gen_config(**overrides) -> GenerationConfig:
    data = {**VALID_GENERATION, **overrides}
    return GenerationConfig(**data)


def _make_query_plan(query_type="single", tasks=None) -> QueryPlan:
    return QueryPlan(
        reasoning="Test reasoning.",
        query_type=query_type,
        resolved_query="What was NVDA's revenue in 2024?",
        tasks=tasks if tasks is not None else [RetrievalTask(keyword_query="NVDA revenue 2024", semantic_query="What was NVIDIA's revenue in FY2024?")],
    )


# ---------------------------------------------------------------------------
# analyze_query
# ---------------------------------------------------------------------------

def test_analyze_query_returns_query_type_and_resolved_query():
    plan = _make_query_plan()
    llm = _make_llm(structured_return=StructuredResponse(parsed=plan, usage=_make_usage()))

    node = make_analyze_query(llm, "Analyze the query.", _make_filings_repo())
    result = node(_base_state())

    assert result["query_type"] == "single"
    assert result["resolved_query"] == "What was NVDA's revenue in 2024?"
    assert result["retrieval_triggered_by"] == "analysis"


def test_analyze_query_passes_system_prompt():
    plan = _make_query_plan()
    llm = _make_llm(structured_return=StructuredResponse(parsed=plan, usage=_make_usage()))

    node = make_analyze_query(llm, "Analyze the query.", _make_filings_repo())
    node(_base_state())

    messages = llm.chat_structured.call_args[0][0]
    assert messages[0].role == "system"
    assert messages[0].content == "Analyze the query."


def test_analyze_query_includes_filter_in_user_message():
    plan = _make_query_plan()
    llm = _make_llm(structured_return=StructuredResponse(parsed=plan, usage=_make_usage()))

    node = make_analyze_query(llm, "Analyze the query.", _make_filings_repo())
    node(_base_state(query_filter=MetadataFilter(ticker="NVDA")))

    messages = llm.chat_structured.call_args[0][0]
    assert "NVDA" in messages[1].content


def test_analyze_query_includes_history_in_user_message():
    plan = _make_query_plan()
    llm = _make_llm(structured_return=StructuredResponse(parsed=plan, usage=_make_usage()))

    node = make_analyze_query(llm, "Analyze the query.", _make_filings_repo())
    node(_base_state(history=[Message(role="user", content="Prior question")]))

    messages = llm.chat_structured.call_args[0][0]
    assert "Prior question" in messages[1].content


def test_analyze_query_out_of_scope_sets_canned_answer():
    plan = QueryPlan(
        reasoning="Not answerable from 10-K filings.",
        query_type="out_of_scope",
        resolved_query="What is the weather?",
        tasks=[],
    )
    llm = _make_llm(structured_return=StructuredResponse(parsed=plan, usage=_make_usage()))

    node = make_analyze_query(llm, "Analyze the query.", _make_filings_repo())
    result = node(_base_state(query="What is the weather?"))

    assert isinstance(result.get("answer"), GenerationResult)
    assert "can't be answered from SEC 10-K filings" in result["answer"].answer


def test_analyze_query_empty_tasks_returns_canned_answer():
    plan = QueryPlan(
        reasoning="Answerable but no tasks produced.",
        query_type="single",
        resolved_query="What was NVDA's revenue?",
        tasks=[],
    )
    llm = _make_llm(structured_return=StructuredResponse(parsed=plan, usage=_make_usage()))

    node = make_analyze_query(llm, "Analyze the query.", _make_filings_repo())
    result = node(_base_state())

    assert isinstance(result.get("answer"), GenerationResult)
    assert "trouble understanding" in result["answer"].answer
    assert result["pending_tasks"] == []


def test_analyze_query_all_tickers_missing_returns_canned_answer():
    plan = QueryPlan(
        reasoning="Single company lookup.",
        query_type="single",
        resolved_query="What was FAKE's revenue?",
        tasks=[RetrievalTask(keyword_query="FAKE revenue", semantic_query="What was FAKE's revenue?", filter=MetadataFilter(ticker="FAKE"))],
    )
    llm = _make_llm(structured_return=StructuredResponse(parsed=plan, usage=_make_usage()))

    node = make_analyze_query(llm, "Analyze the query.", _make_filings_repo(tickers_in_db=[]))
    result = node(_base_state())

    assert isinstance(result.get("answer"), GenerationResult)
    assert "FAKE" in result["answer"].answer
    assert "don't have filings" in result["answer"].answer
    assert result["pending_tasks"] == []


def test_analyze_query_partial_ticker_missing_returns_canned_answer():
    plan = QueryPlan(
        reasoning="Comparison: NVDA in DB, AMD not.",
        query_type="comparison",
        resolved_query="Compare NVDA and AMD revenue.",
        tasks=[
            RetrievalTask(keyword_query="revenue net sales", semantic_query="What was NVIDIA's revenue?", filter=MetadataFilter(ticker="NVDA")),
            RetrievalTask(keyword_query="revenue net sales", semantic_query="What was AMD's revenue?", filter=MetadataFilter(ticker="AMD")),
        ],
    )
    llm = _make_llm(structured_return=StructuredResponse(parsed=plan, usage=_make_usage()))

    node = make_analyze_query(llm, "Analyze the query.", _make_filings_repo(tickers_in_db=["NVDA"]))
    result = node(_base_state())

    assert isinstance(result.get("answer"), GenerationResult)
    assert "NVDA" in result["answer"].answer
    assert "AMD" in result["answer"].answer
    assert "can't complete this comparison" in result["answer"].answer
    assert result["pending_tasks"] == []


def test_analyze_query_returns_pending_tasks_on_success():
    task = RetrievalTask(keyword_query="NVDA revenue 2024", semantic_query="What was NVIDIA's revenue in FY2024?", filter=MetadataFilter(ticker="NVDA"))
    plan = _make_query_plan(tasks=[task])
    llm = _make_llm(structured_return=StructuredResponse(parsed=plan, usage=_make_usage()))

    node = make_analyze_query(llm, "Analyze the query.", _make_filings_repo())
    result = node(_base_state())

    assert len(result["pending_tasks"]) == 1
    assert result["pending_tasks"][0].keyword_query == "NVDA revenue 2024"


# ---------------------------------------------------------------------------
# hyde_expand
# ---------------------------------------------------------------------------

def test_hyde_expand_populates_hyde_query_on_each_task():
    llm = _make_llm(chat_return=LLMResponse(content="Hypothetical passage.", usage=_make_usage()))
    task = RetrievalTask(keyword_query="NVDA revenue 2024", semantic_query="What was NVIDIA's revenue in FY2024?")

    node = make_hyde_expand(llm, "Write a hypothetical passage.")
    result = node(_base_state(pending_tasks=[task]))

    assert len(result["pending_tasks"]) == 1
    assert result["pending_tasks"][0].hyde_query == "Hypothetical passage."


def test_hyde_expand_sends_semantic_query_as_user_message():
    llm = _make_llm(chat_return=LLMResponse(content="passage", usage=_make_usage()))
    task = RetrievalTask(
        keyword_query="total revenues net sales fiscal year",
        semantic_query="What was NVIDIA's total revenue for fiscal year 2024?",
    )

    node = make_hyde_expand(llm, "Write a hypothetical passage.")
    node(_base_state(pending_tasks=[task]))

    messages = llm.chat.call_args[0][0]
    assert messages[-1].role == "user"
    assert "What was NVIDIA's total revenue" in messages[-1].content


def test_hyde_expand_runs_once_per_task():
    llm = _make_llm(chat_return=LLMResponse(content="passage", usage=_make_usage()))
    tasks = [
        RetrievalTask(keyword_query="NVDA revenue", semantic_query="What was NVIDIA's revenue?", filter=MetadataFilter(ticker="NVDA")),
        RetrievalTask(keyword_query="AMD revenue", semantic_query="What was AMD's revenue?", filter=MetadataFilter(ticker="AMD")),
    ]

    node = make_hyde_expand(llm, "Write a hypothetical passage.")
    result = node(_base_state(pending_tasks=tasks))

    assert llm.chat.call_count == 2
    assert len(result["pending_tasks"]) == 2
    assert all(t.hyde_query == "passage" for t in result["pending_tasks"])


def test_hyde_expand_preserves_task_order():
    responses = ["passage A", "passage B", "passage C"]
    call_count = {"n": 0}

    def chat_side_effect(messages, **kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        return LLMResponse(content=responses[idx], usage=_make_usage())

    llm = MagicMock()
    llm.chat.side_effect = chat_side_effect

    tasks = [RetrievalTask(keyword_query=f"query {i}", semantic_query=f"What is query {i}?") for i in range(3)]
    node = make_hyde_expand(llm, "Write a hypothetical passage.")
    result = node(_base_state(pending_tasks=tasks))

    passages = [t.hyde_query for t in result["pending_tasks"]]
    assert set(passages) == {"passage A", "passage B", "passage C"}
    assert len(passages) == 3


# ---------------------------------------------------------------------------
# retrieve
# ---------------------------------------------------------------------------

def test_retrieve_returns_results_wrapped_in_list():
    result = _make_retrieval_result()
    retriever = MagicMock()
    retriever.retrieve.return_value = [result]
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1] * 1024]

    node = make_retrieve(retriever, embedder)
    output = node({"task": RetrievalTask(keyword_query="NVDA revenue", semantic_query="What was NVIDIA's revenue?")})

    assert output["completed_results"] == [[result]]


def test_retrieve_uses_task_hyde_query_for_embedding_when_present():
    retriever = MagicMock()
    retriever.retrieve.return_value = []
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1] * 1024]

    task = RetrievalTask(keyword_query="NVDA revenue", semantic_query="What was NVIDIA's revenue?", hyde_query="Hypothetical passage")
    node = make_retrieve(retriever, embedder)
    node({"task": task})

    embedded_text = embedder.embed.call_args[0][0][0]
    assert embedded_text == "Hypothetical passage"


def test_retrieve_uses_semantic_query_for_embedding_when_no_hyde():
    retriever = MagicMock()
    retriever.retrieve.return_value = []
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1] * 1024]

    node = make_retrieve(retriever, embedder)
    node({"task": RetrievalTask(keyword_query="NVDA revenue", semantic_query="What was NVIDIA's revenue?")})

    embedded_text = embedder.embed.call_args[0][0][0]
    assert embedded_text == "What was NVIDIA's revenue?"


def test_retrieve_passes_filter_to_retriever():
    retriever = MagicMock()
    retriever.retrieve.return_value = []
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1] * 1024]

    f = MetadataFilter(ticker="NVDA")
    node = make_retrieve(retriever, embedder)
    node({"task": RetrievalTask(keyword_query="revenue", semantic_query="What was the revenue?", filter=f)})

    assert retriever.retrieve.call_args[1]["filters"] == f


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

def _make_gen_response(answer: str, cited_indices: list[int]) -> StructuredResponse:
    return StructuredResponse(parsed=GenerationResponse(answer=answer, cited_indices=cited_indices), usage=_make_usage())


def test_generate_returns_generation_result():
    llm = _make_llm(structured_return=_make_gen_response("Revenue was $60.9B.", [1]))
    result = _make_retrieval_result()

    node = make_generate(llm, "Answer the question.", "Compare the companies.")
    output = node(_base_state(completed_results=[[result]]))

    assert isinstance(output["answer"], GenerationResult)
    assert output["answer"].answer == "Revenue was $60.9B."


def test_generate_builds_citations_from_results():
    llm = _make_llm(structured_return=_make_gen_response("Answer.", [1]))
    result = _make_retrieval_result()

    node = make_generate(llm, "Answer the question.", "Compare the companies.")
    output = node(_base_state(completed_results=[[result]]))

    assert len(output["answer"].citations) == 1
    assert output["answer"].citations[0].ticker == "NVDA"


def test_generate_deduplicates_citations():
    llm = _make_llm(structured_return=_make_gen_response("Answer.", [1]))
    result = _make_retrieval_result()

    node = make_generate(llm, "Answer the question.", "Compare the companies.")
    output = node(_base_state(completed_results=[[result], [result]]))

    assert len(output["answer"].citations) == 1


def test_generate_orders_high_frequency_chunks_first_in_context():
    llm = _make_llm(structured_return=_make_gen_response("Answer.", []))
    filing = _make_filing()

    high_freq_parent = ParentChunkRecord(
        id=uuid.uuid4(), filing_id=filing.id, chunk_index=0, section="s1",
        text="HIGH FREQUENCY CHUNK", token_count=3, content_hash="a" * 64,
        created_at=datetime(2024, 3, 1),
    )
    low_freq_parent = ParentChunkRecord(
        id=uuid.uuid4(), filing_id=filing.id, chunk_index=1, section="s2",
        text="LOW FREQUENCY CHUNK", token_count=3, content_hash="b" * 64,
        created_at=datetime(2024, 3, 1),
    )
    high_freq_result = RetrievalResult(
        score=0.9, vector_score=None, keyword_score=None, reranker_score=None,
        chunk=_make_chunk(filing.id, high_freq_parent.id),
        parent_chunk=high_freq_parent, filing=filing,
    )
    low_freq_result = RetrievalResult(
        score=0.9, vector_score=None, keyword_score=None, reranker_score=None,
        chunk=_make_chunk(filing.id, low_freq_parent.id),
        parent_chunk=low_freq_parent, filing=filing,
    )

    node = make_generate(llm, "Answer the question.", "Compare the companies.")
    output = node(_base_state(completed_results=[[high_freq_result, low_freq_result], [high_freq_result]]))

    context = llm.chat_structured.call_args[0][0][-1].content
    assert context.index("HIGH FREQUENCY CHUNK") < context.index("LOW FREQUENCY CHUNK")


def test_generate_filters_citations_to_cited_indices():
    filing = _make_filing()
    parent1 = _make_parent_chunk(filing.id)
    parent2 = ParentChunkRecord(
        id=uuid.uuid4(), filing_id=filing.id, chunk_index=1, section="Risk Factors",
        text="Competition is intense.", token_count=3, content_hash="c" * 64,
        created_at=datetime(2024, 3, 1),
    )
    result1 = RetrievalResult(
        score=0.9, vector_score=None, keyword_score=None, reranker_score=None,
        chunk=_make_chunk(filing.id, parent1.id), parent_chunk=parent1, filing=filing,
    )
    result2 = RetrievalResult(
        score=0.8, vector_score=None, keyword_score=None, reranker_score=None,
        chunk=_make_chunk(filing.id, parent2.id), parent_chunk=parent2, filing=filing,
    )

    llm = _make_llm(structured_return=_make_gen_response("Revenue was $60.9B [1].", [1]))
    node = make_generate(llm, "Answer the question.", "Compare the companies.")
    output = node(_base_state(completed_results=[[result1, result2]]))

    assert len(output["answer"].citations) == 1
    assert output["answer"].citations[0].section == parent1.section


def test_generate_falls_back_to_all_citations_when_none_cited():
    result1 = _make_retrieval_result()
    result2 = _make_retrieval_result()

    llm = _make_llm(structured_return=_make_gen_response("I cannot determine this.", []))
    node = make_generate(llm, "Answer the question.", "Compare the companies.")
    output = node(_base_state(completed_results=[[result1, result2]]))

    assert len(output["answer"].citations) == 2


@pytest.mark.parametrize("query_type,expected_keyword", [
    ("single", "Answer the question"),
    ("comparison", "Compare the companies"),
])
def test_generate_selects_correct_prompt(query_type, expected_keyword):
    prompt = _select_prompt(query_type, qa="Answer the question", comparison="Compare the companies")
    assert expected_keyword in prompt


# ---------------------------------------------------------------------------
# check_hop
# ---------------------------------------------------------------------------

def test_check_hop_returns_empty_tasks_when_done():
    decision = HopDecision(done=True)
    llm = _make_llm(structured_return=StructuredResponse(parsed=decision, usage=_make_usage()))
    result = _make_retrieval_result()

    node = make_check_hop(llm, _gen_config(), "Decide if more retrieval is needed.")
    output = node(_base_state(completed_results=[[result]], hop_count=0))

    assert output["pending_tasks"] == []
    assert output["hop_count"] == 1
    assert output["retrieval_triggered_by"] == "check_hop"


def test_check_hop_returns_next_task_when_not_done():
    next_task = RetrievalTask(keyword_query="follow-up query", semantic_query="What is the follow-up information?")
    decision = HopDecision(done=False, next_task=next_task)
    llm = _make_llm(structured_return=StructuredResponse(parsed=decision, usage=_make_usage()))
    result = _make_retrieval_result()

    node = make_check_hop(llm, _gen_config(), "Decide if more retrieval is needed.")
    output = node(_base_state(completed_results=[[result]], hop_count=0))

    assert len(output["pending_tasks"]) == 1
    assert output["pending_tasks"][0].keyword_query == "follow-up query"


def test_check_hop_returns_empty_tasks_when_not_done_but_next_task_is_none():
    decision = HopDecision(done=False, next_task=None)
    llm = _make_llm(structured_return=StructuredResponse(parsed=decision, usage=_make_usage()))
    result = _make_retrieval_result()

    node = make_check_hop(llm, _gen_config(), "Decide if more retrieval is needed.")
    output = node(_base_state(completed_results=[[result]], hop_count=0))

    assert output["pending_tasks"] == []


# ---------------------------------------------------------------------------
# reflect
# ---------------------------------------------------------------------------

def test_reflect_returns_empty_tasks_on_high_quality():
    decision = ReflectionDecision(quality="high", reason="complete and grounded")
    llm = _make_llm(structured_return=StructuredResponse(parsed=decision, usage=_make_usage()))
    result = _make_retrieval_result()
    answer = GenerationResult(answer="Revenue was $60.9B.", citations=[])

    node = make_reflect(llm, _gen_config(), "Evaluate the answer quality.")
    output = node(_base_state(completed_results=[[result]], answer=answer, reflection_count=0))

    assert output["pending_tasks"] == []
    assert output["reflection_count"] == 1
    assert output["retrieval_triggered_by"] == "reflect"


def test_reflect_returns_next_task_on_low_quality():
    next_task = RetrievalTask(keyword_query="missing data", semantic_query="What is the missing data?")
    decision = ReflectionDecision(quality="low", reason="revenue figure missing", next_task=next_task)
    llm = _make_llm(structured_return=StructuredResponse(parsed=decision, usage=_make_usage()))
    result = _make_retrieval_result()
    answer = GenerationResult(answer="Incomplete answer.", citations=[])

    node = make_reflect(llm, _gen_config(), "Evaluate the answer quality.")
    output = node(_base_state(completed_results=[[result]], answer=answer, reflection_count=0))

    assert len(output["pending_tasks"]) == 1
    assert output["pending_tasks"][0].keyword_query == "missing data"


def test_reflect_returns_empty_tasks_when_low_quality_but_no_next_task():
    decision = ReflectionDecision(quality="low", reason="out of scope", next_task=None)
    llm = _make_llm(structured_return=StructuredResponse(parsed=decision, usage=_make_usage()))
    result = _make_retrieval_result()
    answer = GenerationResult(answer="Bad answer.", citations=[])

    node = make_reflect(llm, _gen_config(), "Evaluate the answer quality.")
    output = node(_base_state(completed_results=[[result]], answer=answer, reflection_count=0))

    assert output["pending_tasks"] == []
