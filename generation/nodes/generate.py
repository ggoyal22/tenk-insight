import logging
from collections.abc import Callable

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
        context = build_context(state["completed_results"])

        messages = [Message(role="system", content=prompt)]
        for msg in (state.get("history") or []):
            messages.append(msg)
        messages.append(Message(
            role="user",
            content=f"Question: {state['query']}\n\nContext:\n{context}",
        ))

        response = llm.chat(messages)
        citations = _build_citations(state["completed_results"])

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


def _build_citations(completed_results: list[list[RetrievalResult]]) -> list[Citation]:
    seen: set[tuple] = set()
    citations: list[Citation] = []
    for result_group in completed_results:
        for r in result_group:
            f = r.filing
            # Deduplicate by accession number + section — same passage cited multiple
            # times across hops should only appear once in the citation list.
            key = (f.accession_number, r.parent_chunk.section)
            if key in seen:
                continue
            seen.add(key)
            citations.append(Citation(
                ticker=f.ticker,
                company_name=f.company_name,
                form_type=f.form_type,
                fiscal_year_end=f.fiscal_year_end,
                filing_date=f.filing_date,
                accession_number=f.accession_number,
                source_url=f.source_url,
                section=r.parent_chunk.section,
                chunk_text=r.parent_chunk.text,
            ))
    return citations
