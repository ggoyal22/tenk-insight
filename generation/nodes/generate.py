import logging
from collections.abc import Callable
from uuid import UUID

from llm.base import BaseLLM
from llm.types import Message
from generation.nodes._context import build_context
from generation.prompts import COMPARISON_PROMPT, QA_PROMPT, TIME_SERIES_PROMPT
from generation.types import Citation, GenerationResult, GenerationState
from retrieval.types import RetrievalResult

logger = logging.getLogger(__name__)


def make_generate(llm: BaseLLM) -> Callable[[GenerationState], dict]:
    def generate(state: GenerationState) -> dict:
        prompt = _select_prompt(state["query_type"])
        results = _deduplicate(state["completed_results"])
        context = build_context(results)

        messages = [Message(role="system", content=prompt)]
        for msg in (state.get("history") or []):
            messages.append(msg)
        messages.append(Message(
            role="user",
            content=f"Question: {state['query']}\n\nContext:\n{context}",
        ))

        response = llm.chat(messages)
        citations = _build_citations(results)

        logger.debug(
            "Answer generated (%d chars, %d citation(s)).",
            len(response.content), len(citations),
        )

        return {"answer": GenerationResult(
            answer=response.content,
            citations=citations,
            usage=response.usage,
        )}

    return generate


def _select_prompt(query_type: str) -> str:
    if query_type == "comparison":
        return COMPARISON_PROMPT
    if query_type == "time_series":
        return TIME_SERIES_PROMPT
    return QA_PROMPT  # single, multi_hop


def _deduplicate(completed_results: list[list[RetrievalResult]]) -> list[RetrievalResult]:
    """Flatten multi-hop results, deduplicate by parent_chunk.id, sort by hop frequency descending.

    Chunks retrieved across more hops appear first — mitigates Lost-in-the-Middle
    by placing the most corroborated evidence at the top of the LLM context.
    """
    frequency: dict[UUID, int] = {}
    unique: dict[UUID, RetrievalResult] = {}
    for result_group in completed_results:
        for r in result_group:
            pid = r.parent_chunk.id
            frequency[pid] = frequency.get(pid, 0) + 1
            if pid not in unique:
                unique[pid] = r
    return sorted(unique.values(), key=lambda r: frequency[r.parent_chunk.id], reverse=True)


def _build_citations(results: list[RetrievalResult]) -> list[Citation]:
    return [
        Citation(
            ticker=r.filing.ticker,
            company_name=r.filing.company_name,
            form_type=r.filing.form_type,
            fiscal_year_end=r.filing.fiscal_year_end,
            filing_date=r.filing.filing_date,
            accession_number=r.filing.accession_number,
            source_url=r.filing.source_url,
            section=r.parent_chunk.section,
            chunk_text=r.parent_chunk.text,
        )
        for r in results
    ]
