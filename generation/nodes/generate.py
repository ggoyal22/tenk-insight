import logging
from collections.abc import Callable
from uuid import UUID

from generation.nodes._stream import get_writer

from llm.base import BaseLLM
from llm.types import Message
from generation.nodes._context import build_context
from generation.types import Citation, GenerationResponse, GenerationResult, GenerationState
from retrieval.types import RetrievalResult

logger = logging.getLogger(__name__)


def make_generate(
    llm: BaseLLM,
    qa_prompt: str,
    comparison_prompt: str,
    time_series_prompt: str,
) -> Callable[[GenerationState], dict]:
    def generate(state: GenerationState) -> dict:
        get_writer()("Generating answer...")
        prompt = _select_prompt(state["query_type"], qa_prompt, comparison_prompt, time_series_prompt)
        results = _deduplicate(state["completed_results"])

        if not results:
            return {"answer": GenerationResult(
                answer="I found filings but couldn't locate relevant information. Try narrowing your question.",
                citations=[],
            )}

        context = build_context(results)

        messages = [
            Message(role="system", content=prompt),
            Message(role="user", content=f"Question: {state.get('resolved_query') or state['query']}\n\nContext:\n{context}"),
        ]

        response = llm.chat_structured(messages, GenerationResponse)
        cited_results = _filter_by_indices(results, response.parsed.cited_indices)
        citations = _build_citations(cited_results)

        logger.debug(
            "Answer generated (%d chars, %d citation(s)).",
            len(response.parsed.answer), len(citations),
        )

        return {
            "answer": GenerationResult(
                answer=response.parsed.answer,
                citations=citations,
            ),
            "pipeline_usage": [response.usage],
        }

    return generate


def _select_prompt(query_type: str, qa: str, comparison: str, time_series: str) -> str:
    if query_type == "comparison":
        return comparison
    if query_type == "time_series":
        return time_series
    return qa  # single, multi_hop


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


def _filter_by_indices(results: list[RetrievalResult], cited_indices: list[int]) -> list[RetrievalResult]:
    cited = [results[i - 1] for i in cited_indices if 1 <= i <= len(results)]
    return cited if cited else results


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
