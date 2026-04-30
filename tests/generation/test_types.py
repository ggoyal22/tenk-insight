"""
Tests for generation/types.py.

Covers Pydantic model validation (the types used with chat_structured) and
basic dataclass construction (Citation, GenerationResult).
"""

from datetime import date

import pytest
from pydantic import BaseModel, ValidationError

from generation.types import (
    Citation,
    GenerationResult,
    GenerationState,
    HopDecision,
    QueryAnalysis,
    ReflectionDecision,
    RetrievalTask,
)
from llm.types import LLMUsage
from retrieval.types import MetadataFilter


# ---------------------------------------------------------------------------
# RetrievalTask
# ---------------------------------------------------------------------------

def test_retrieval_task_accepts_filter():
    f = MetadataFilter(ticker="NVDA", form_type="10-K")
    task = RetrievalTask(query="NVDA revenue", filter=f)
    assert task.query == "NVDA revenue"
    assert task.filter.ticker == "NVDA"


def test_retrieval_task_filter_defaults_to_none():
    task = RetrievalTask(query="some query")
    assert task.filter is None


def test_retrieval_task_has_json_schema():
    schema = RetrievalTask.model_json_schema()
    assert "query" in schema["properties"]


# ---------------------------------------------------------------------------
# QueryAnalysis
# ---------------------------------------------------------------------------

def test_query_analysis_valid():
    task = RetrievalTask(query="NVDA revenue 2024", filter=MetadataFilter(ticker="NVDA"))
    qa = QueryAnalysis(query_type="single", tasks=[task])
    assert qa.query_type == "single"
    assert len(qa.tasks) == 1


def test_query_analysis_rejects_invalid_query_type():
    with pytest.raises(ValidationError):
        QueryAnalysis(query_type="unknown", tasks=[])


@pytest.mark.parametrize("qt", ["single", "comparison", "time_series", "multi_hop", "out_of_scope"])
def test_query_analysis_accepts_all_valid_types(qt):
    qa = QueryAnalysis(query_type=qt, tasks=[])
    assert qa.query_type == qt


def test_query_analysis_has_json_schema():
    schema = QueryAnalysis.model_json_schema()
    assert "query_type" in schema["properties"]
    assert "tasks" in schema["properties"]


# ---------------------------------------------------------------------------
# HopDecision
# ---------------------------------------------------------------------------

def test_hop_decision_done():
    hd = HopDecision(done=True)
    assert hd.done is True
    assert hd.next_task is None


def test_hop_decision_not_done_with_task():
    task = RetrievalTask(query="follow-up query")
    hd = HopDecision(done=False, next_task=task)
    assert hd.done is False
    assert hd.next_task.query == "follow-up query"


def test_hop_decision_has_json_schema():
    schema = HopDecision.model_json_schema()
    assert "done" in schema["properties"]


# ---------------------------------------------------------------------------
# ReflectionDecision
# ---------------------------------------------------------------------------

def test_reflection_decision_high_quality():
    rd = ReflectionDecision(quality="high", reason="answer is complete and grounded")
    assert rd.quality == "high"
    assert rd.next_task is None


def test_reflection_decision_low_quality_with_task():
    task = RetrievalTask(query="missing revenue data")
    rd = ReflectionDecision(quality="low", reason="revenue figure not in context", next_task=task)
    assert rd.quality == "low"
    assert rd.next_task.query == "missing revenue data"


def test_reflection_decision_rejects_invalid_quality():
    with pytest.raises(ValidationError):
        ReflectionDecision(quality="medium", reason="ok")


def test_reflection_decision_has_json_schema():
    schema = ReflectionDecision.model_json_schema()
    assert "quality" in schema["properties"]
    assert "reason" in schema["properties"]


# ---------------------------------------------------------------------------
# Citation and GenerationResult (dataclasses)
# ---------------------------------------------------------------------------

def test_citation_fields():
    c = Citation(
        ticker="NVDA",
        company_name="NVIDIA Corporation",
        form_type="10-K",
        fiscal_year_end=date(2024, 1, 28),
        filing_date=date(2024, 2, 21),
        accession_number="0001045810-24-000029",
        source_url="https://www.sec.gov/Archives/edgar/data/1045810/000104581024000029/",
        section="Financial Statements",
        chunk_text="Total revenue was $60.9 billion for fiscal 2024.",
    )
    assert c.ticker == "NVDA"
    assert c.fiscal_year_end == date(2024, 1, 28)


def test_generation_result_fields():
    usage = LLMUsage(input_tokens=100, output_tokens=200)
    result = GenerationResult(answer="NVDA revenue was $60.9B.", citations=[], usage=usage)
    assert result.answer == "NVDA revenue was $60.9B."
    assert result.usage.input_tokens == 100


# ---------------------------------------------------------------------------
# Pydantic models are usable as chat_structured schemas
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [QueryAnalysis, HopDecision, ReflectionDecision])
def test_llm_output_types_have_json_schema(cls):
    schema = cls.model_json_schema()
    assert isinstance(schema, dict)
    assert "properties" in schema


@pytest.mark.parametrize("cls", [QueryAnalysis, HopDecision, ReflectionDecision])
def test_llm_output_types_are_pydantic_models(cls):
    assert issubclass(cls, BaseModel)
