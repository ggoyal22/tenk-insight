import logging
from collections.abc import Callable
from uuid import UUID

from generation.nodes._stream import get_writer

from llm.base import BaseLLM, LLMError
from llm.types import Message
from generation.nodes._context import build_context
from generation.token_limits import MAX_TOKENS_GENERATE
from generation.types import Citation, GenerationResponse, GenerationResult, GenerationState
from retrieval.types import RetrievalResult

logger = logging.getLogger(__name__)


def make_generate(
    llm: BaseLLM,
    qa_prompt: str,
    comparison_prompt: str,
) -> Callable[[GenerationState], dict]:
    def generate(state: GenerationState) -> dict:
        get_writer()("Generating answer...")
        prompt = _select_prompt(state["query_type"], qa_prompt, comparison_prompt)
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

        try:
            response = llm.chat_structured(messages, GenerationResponse, max_tokens=MAX_TOKENS_GENERATE)
        except LLMError:
            logger.exception("generate failed.")
            return {
                "has_error": True,
                "answer": GenerationResult(answer="Something went wrong — please try again.", citations=[]),
            }

        indexed_results = _filter_by_indices(results, response.parsed.cited_indices)
        citations = _build_citations(indexed_results)

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


def _select_prompt(query_type: str, qa: str, comparison: str) -> str:
    if query_type == "comparison":
        return comparison
    return qa  # single


def _deduplicate(completed_results: list[list[RetrievalResult]]) -> list[RetrievalResult]:
    """Flatten multi-hop results, deduplicate by parent_chunk.id, sort by best child relevance score.

    For each parent, tracks the max reranker_score (or RRF score as fallback) across all
    children seen — places the most query-relevant parent context first to mitigate
    Lost-in-the-Middle.
    """
    best_score: dict[UUID, float] = {}
    unique: dict[UUID, RetrievalResult] = {}
    for result_group in completed_results:
        for r in result_group:
            pid = r.parent_chunk.id
            score = r.reranker_score if r.reranker_score is not None else r.score
            if score > best_score.get(pid, float("-inf")):
                best_score[pid] = score
            if pid not in unique:
                unique[pid] = r
    return sorted(unique.values(), key=lambda r: best_score[r.parent_chunk.id], reverse=True)


def _filter_by_indices(results: list[RetrievalResult], cited_indices: list[int]) -> list[tuple[int, RetrievalResult]]:
    """Pair each cited result with the 1-based index the model used to cite it.

    That index is what the model wrote as [N] in the answer text, so it must travel
    with the result into the citation — renumbering by list position would misalign
    the markers once uncited excerpts are dropped. Falls back to every result (in
    context order) when the model cited nothing usable.
    """
    cited = [(i, results[i - 1]) for i in cited_indices if 1 <= i <= len(results)]
    return cited if cited else list(enumerate(results, start=1))


def _build_citations(indexed_results: list[tuple[int, RetrievalResult]]) -> list[Citation]:
    return [
        Citation(
            index=i,
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
        for i, r in indexed_results
    ]
