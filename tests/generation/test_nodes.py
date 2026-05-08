"""
Tests for generation nodes. All LLM and retriever calls are mocked.
"""

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from config.loader import GenerationConfig
from generation.nodes.analyze import make_analyze_query
from generation.nodes.check_hop import make_check_hop
from generation.nodes.generate import make_generate, _select_prompt, _filter_by_indices
from generation.nodes.hyde import make_hyde_expand
from generation.nodes.reflect import make_reflect
from generation.nodes.retrieve import make_retrieve, RetrieveInput
from generation.types import (
    Citation, GenerationResponse, GenerationResult, GenerationState,
    HopDecision, QueryAnalysis, ReflectionDecision, RetrievalTask,
)
from llm.base import LLMError
from llm.types import LLMResponse, LLMUsage, Message, StructuredResponse
from retrieval.types import MetadataFilter, RetrievalResult
from db.models import ChunkRecord, FilingRecord, ParentChunkRecord
from tests.conftest import VALID_GENERATION

import uuid
from datetime import datetime


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
    return RetrievalResult(score=0.9, chunk=chunk, parent_chunk=parent, filing=filing)


def _base_state(**overrides) -> dict:
    state = {
        "query": "What was NVDA's revenue in 2024?",
        "history": [],
        "query_filter": None,
        "query_type": "single",
        "pending_tasks": [],
        "hyde_query": None,
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


# ---------------------------------------------------------------------------
# analyze_query
# ---------------------------------------------------------------------------

def test_analyze_query_returns_query_type_and_tasks():
    task = RetrievalTask(query="NVDA revenue 2024", filter=MetadataFilter(ticker="NVDA"))
    analysis = QueryAnalysis(query_type="single", tasks=[task])
    llm = _make_llm(structured_return=StructuredResponse(parsed=analysis, usage=_make_usage()))

    node = make_analyze_query(llm, "Analyse the query.")
    result = node(_base_state())

    assert result["query_type"] == "single"
    assert len(result["pending_tasks"]) == 1
    assert result["retrieval_triggered_by"] == "analysis"


def test_analyze_query_passes_system_prompt():
    analysis = QueryAnalysis(query_type="out_of_scope", tasks=[])
    llm = _make_llm(structured_return=StructuredResponse(parsed=analysis, usage=_make_usage()))

    node = make_analyze_query(llm, "Analyse the query.")
    node(_base_state())

    messages = llm.chat_structured.call_args[0][0]
    assert messages[0].role == "system"
    assert len(messages[0].content) > 0


def test_analyze_query_includes_filter_in_user_message():
    analysis = QueryAnalysis(query_type="single", tasks=[])
    llm = _make_llm(structured_return=StructuredResponse(parsed=analysis, usage=_make_usage()))

    node = make_analyze_query(llm, "Analyse the query.")
    node(_base_state(query_filter=MetadataFilter(ticker="NVDA")))

    messages = llm.chat_structured.call_args[0][0]
    user_msg = messages[1].content
    assert "NVDA" in user_msg


def test_analyze_query_includes_history_in_user_message():
    analysis = QueryAnalysis(query_type="single", tasks=[])
    llm = _make_llm(structured_return=StructuredResponse(parsed=analysis, usage=_make_usage()))
    history = [Message(role="user", content="Prior question")]

    node = make_analyze_query(llm, "Analyse the query.")
    node(_base_state(history=history))

    messages = llm.chat_structured.call_args[0][0]
    assert "Prior question" in messages[1].content


# ---------------------------------------------------------------------------
# hyde_expand
# ---------------------------------------------------------------------------

def test_hyde_expand_returns_hyde_query():
    llm = _make_llm(chat_return=LLMResponse(content="Hypothetical passage.", usage=_make_usage()))
    node = make_hyde_expand(llm, "Write a hypothetical passage.")
    result = node(_base_state())
    assert result["hyde_query"] == "Hypothetical passage."


def test_hyde_expand_sends_query_as_user_message():
    llm = _make_llm(chat_return=LLMResponse(content="passage", usage=_make_usage()))
    node = make_hyde_expand(llm, "Write a hypothetical passage.")
    node(_base_state(query="What was NVDA revenue?"))

    messages = llm.chat.call_args[0][0]
    assert messages[-1].role == "user"
    assert "NVDA revenue" in messages[-1].content


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
    output = node({"task": RetrievalTask(query="NVDA revenue"), "hyde_query": None})

    assert output["completed_results"] == [[result]]


def test_retrieve_uses_hyde_query_for_embedding_when_present():
    retriever = MagicMock()
    retriever.retrieve.return_value = []
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1] * 1024]

    node = make_retrieve(retriever, embedder)
    node({"task": RetrievalTask(query="NVDA revenue"), "hyde_query": "Hypothetical passage"})

    embedded_text = embedder.embed.call_args[0][0][0]
    assert embedded_text == "Hypothetical passage"


def test_retrieve_uses_task_query_for_embedding_when_no_hyde():
    retriever = MagicMock()
    retriever.retrieve.return_value = []
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1] * 1024]

    node = make_retrieve(retriever, embedder)
    node({"task": RetrievalTask(query="NVDA revenue"), "hyde_query": None})

    embedded_text = embedder.embed.call_args[0][0][0]
    assert embedded_text == "NVDA revenue"


def test_retrieve_passes_filter_to_retriever():
    retriever = MagicMock()
    retriever.retrieve.return_value = []
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1] * 1024]

    f = MetadataFilter(ticker="NVDA")
    node = make_retrieve(retriever, embedder)
    node({"task": RetrievalTask(query="revenue", filter=f), "hyde_query": None})

    assert retriever.retrieve.call_args[1]["filters"] == f


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

def _make_gen_response(answer: str, cited_indices: list[int]) -> StructuredResponse:
    return StructuredResponse(parsed=GenerationResponse(answer=answer, cited_indices=cited_indices), usage=_make_usage())


def test_generate_returns_generation_result():
    llm = _make_llm(structured_return=_make_gen_response("Revenue was $60.9B.", [1]))
    result = _make_retrieval_result()

    node = make_generate(llm, "Answer the question.", "Compare the companies.", "Analyse how.")
    output = node(_base_state(completed_results=[[result]]))

    assert isinstance(output["answer"], GenerationResult)
    assert output["answer"].answer == "Revenue was $60.9B."


def test_generate_builds_citations_from_results():
    llm = _make_llm(structured_return=_make_gen_response("Answer.", [1]))
    result = _make_retrieval_result()

    node = make_generate(llm, "Answer the question.", "Compare the companies.", "Analyse how.")
    output = node(_base_state(completed_results=[[result]]))

    assert len(output["answer"].citations) == 1
    assert output["answer"].citations[0].ticker == "NVDA"


def test_generate_deduplicates_citations():
    llm = _make_llm(structured_return=_make_gen_response("Answer.", [1]))
    result = _make_retrieval_result()

    node = make_generate(llm, "Answer the question.", "Compare the companies.", "Analyse how.")
    # Same result appearing in two groups (e.g. two retrieval hops returned same chunk)
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
        score=0.9, chunk=_make_chunk(filing.id, high_freq_parent.id),
        parent_chunk=high_freq_parent, filing=filing,
    )
    low_freq_result = RetrievalResult(
        score=0.9, chunk=_make_chunk(filing.id, low_freq_parent.id),
        parent_chunk=low_freq_parent, filing=filing,
    )

    node = make_generate(llm, "Answer the question.", "Compare the companies.", "Analyse how.")
    # high_freq_result appears in both hops, low_freq_result in only one
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
    result1 = RetrievalResult(score=0.9, chunk=_make_chunk(filing.id, parent1.id), parent_chunk=parent1, filing=filing)
    result2 = RetrievalResult(score=0.8, chunk=_make_chunk(filing.id, parent2.id), parent_chunk=parent2, filing=filing)

    # LLM only cites index 1, not 2
    llm = _make_llm(structured_return=_make_gen_response("Revenue was $60.9B [1].", [1]))
    node = make_generate(llm, "Answer the question.", "Compare the companies.", "Analyse how.")
    output = node(_base_state(completed_results=[[result1, result2]]))

    assert len(output["answer"].citations) == 1
    assert output["answer"].citations[0].section == parent1.section


def test_generate_falls_back_to_all_citations_when_none_cited():
    result1 = _make_retrieval_result()
    result2 = _make_retrieval_result()

    # LLM returns empty cited_indices
    llm = _make_llm(structured_return=_make_gen_response("I cannot determine this.", []))
    node = make_generate(llm, "Answer the question.", "Compare the companies.", "Analyse how.")
    output = node(_base_state(completed_results=[[result1, result2]]))

    assert len(output["answer"].citations) == 2


@pytest.mark.parametrize("query_type,expected_keyword", [
    ("single", "Answer the question"),
    ("multi_hop", "Answer the question"),
    ("comparison", "Compare the companies"),
    ("time_series", "Analyse how"),
])
def test_generate_selects_correct_prompt(query_type, expected_keyword):
    prompt = _select_prompt(
        query_type,
        qa="Answer the question",
        comparison="Compare the companies",
        time_series="Analyse how",
    )
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
    next_task = RetrievalTask(query="follow-up query")
    decision = HopDecision(done=False, next_task=next_task)
    llm = _make_llm(structured_return=StructuredResponse(parsed=decision, usage=_make_usage()))
    result = _make_retrieval_result()

    node = make_check_hop(llm, _gen_config(), "Decide if more retrieval is needed.")
    output = node(_base_state(completed_results=[[result]], hop_count=0))

    assert len(output["pending_tasks"]) == 1
    assert output["pending_tasks"][0].query == "follow-up query"


def test_check_hop_returns_empty_tasks_when_not_done_but_next_task_is_none():
    # LLM returns done=False but omits next_task — guard must prevent [None] in pending_tasks
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
    answer = GenerationResult(answer="Revenue was $60.9B.", citations=[], usage=_make_usage())

    node = make_reflect(llm, _gen_config(), "Evaluate the answer quality.")
    output = node(_base_state(completed_results=[[result]], answer=answer, reflection_count=0))

    assert output["pending_tasks"] == []
    assert output["reflection_count"] == 1
    assert output["retrieval_triggered_by"] == "reflect"


def test_reflect_returns_next_task_on_low_quality():
    next_task = RetrievalTask(query="missing data")
    decision = ReflectionDecision(quality="low", reason="revenue figure missing", next_task=next_task)
    llm = _make_llm(structured_return=StructuredResponse(parsed=decision, usage=_make_usage()))
    result = _make_retrieval_result()
    answer = GenerationResult(answer="Incomplete answer.", citations=[], usage=_make_usage())

    node = make_reflect(llm, _gen_config(), "Evaluate the answer quality.")
    output = node(_base_state(completed_results=[[result]], answer=answer, reflection_count=0))

    assert len(output["pending_tasks"]) == 1
    assert output["pending_tasks"][0].query == "missing data"


def test_reflect_returns_empty_tasks_when_low_quality_but_no_next_task():
    decision = ReflectionDecision(quality="low", reason="out of scope", next_task=None)
    llm = _make_llm(structured_return=StructuredResponse(parsed=decision, usage=_make_usage()))
    result = _make_retrieval_result()
    answer = GenerationResult(answer="Bad answer.", citations=[], usage=_make_usage())

    node = make_reflect(llm, _gen_config(), "Evaluate the answer quality.")
    output = node(_base_state(completed_results=[[result]], answer=answer, reflection_count=0))

    assert output["pending_tasks"] == []
