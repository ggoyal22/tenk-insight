import operator
from dataclasses import dataclass
from datetime import date
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

from llm.types import LLMUsage, Message
from retrieval.types import MetadataFilter, RetrievalResult


# ── Field length limits for LLM-produced query strings ───────────────────────
_MAX_KEYWORD_QUERY  = 150
_MAX_SEMANTIC_QUERY = 300
_MAX_HYDE_QUERY     = 800

# ── Pydantic models — produced by the LLM via chat_structured() ──────────────
# These must be BaseModel so chat_structured() can derive a JSON Schema from
# them and parse the LLM's response back into typed objects.

class RetrievalTask(BaseModel):
    """A single retrieval request: a search query plus optional metadata filters."""
    keyword_query: str = Field(max_length=_MAX_KEYWORD_QUERY)
    semantic_query: str = Field(max_length=_MAX_SEMANTIC_QUERY)
    filter: MetadataFilter | None = None
    hyde_query: str | None = Field(default=None, max_length=_MAX_HYDE_QUERY)


class QueryPlan(BaseModel):
    """Structured output of the analyze_query node."""
    reasoning: str = Field(description="Scratchpad — think through inputs needed before planning tasks.")
    query_type: Literal["out_of_scope", "single", "comparison"]
    resolved_query: str = Field(
        description=(
            "Self-contained rewrite resolving pronouns and company name references. "
            "Normalise to ticker symbols (Apple → AAPL, NVIDIA → NVDA). "
            "Copy verbatim if already self-contained."
        )
    )
    tasks: list[RetrievalTask]


class HopDecision(BaseModel):
    """Structured output of the check_hop node."""
    reasoning: str = Field(description="Scan context for sufficiency, identify gaps, and plan next query before deciding.")
    done: bool
    next_task: RetrievalTask | None = None


class ReflectionDecision(BaseModel):
    """Structured output of the reflect node."""
    quality: Literal["high", "low"]
    reason: str
    next_task: RetrievalTask | None = None


class GenerationResponse(BaseModel):
    """Structured output of the generate node."""
    reasoning: str = Field(description="Think through query scope, granularity, chunk mapping, and completeness before writing the answer.")
    answer: str
    cited_indices: list[int]


# ── Dataclasses — assembled by node code, never produced by the LLM ──────────

@dataclass
class Citation:
    ticker: str
    company_name: str
    form_type: str
    fiscal_year_end: date | None
    filing_date: date
    accession_number: str
    source_url: str
    section: str
    chunk_text: str


@dataclass
class GenerationResult:
    answer: str
    citations: list[Citation]


def total_pipeline_usage(usages: list[LLMUsage]) -> LLMUsage:
    """Sum all LLM usage records accumulated across the pipeline."""
    return sum(usages, LLMUsage(0, 0))


# ── LangGraph state ───────────────────────────────────────────────────────────

class GenerationState(TypedDict):
    query: str
    history: list[Message]
    # Renamed from 'filter' to avoid shadowing the Python builtin.
    query_filter: MetadataFilter | None
    query_type: str
    resolved_query: str | None
    pending_tasks: list[RetrievalTask]
    # Reducer appends each retrieve node's result list — preserves which results
    # came from which task so generate can build per-source citations.
    completed_results: Annotated[list[list[RetrievalResult]], operator.add]
    # Tasks that returned zero chunks — accumulated so check_hop can avoid repeating them.
    # hyde_query is stripped before appending (it's an embedding aid, not a search query).
    failed_queries: Annotated[list[RetrievalTask], operator.add]
    # Reducer accumulates LLM usage across all nodes; generate sums it for the final total.
    pipeline_usage: Annotated[list[LLMUsage], operator.add]
    hop_count: int
    reflection_count: int
    retrieval_triggered_by: Literal["analysis", "check_hop", "reflect"]
    answer: GenerationResult | None
