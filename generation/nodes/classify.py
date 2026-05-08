import logging
from collections.abc import Callable

from generation.nodes._stream import get_writer
from generation.types import GenerationResult, GenerationState, QueryClassification
from llm.base import BaseLLM
from llm.types import Message

logger = logging.getLogger(__name__)


def make_classify_query(llm: BaseLLM, prompt: str) -> Callable[[GenerationState], dict]:
    def classify_query(state: GenerationState) -> dict:
        write = get_writer()
        write("Analyzing your question...")
        messages = [
            Message(role="system", content=prompt),
            Message(role="user", content=_build_user_message(state)),
        ]
        response = llm.chat_structured(messages, QueryClassification)
        classification = response.parsed

        write(f"Classified as **{classification.query_type}**")
        logger.debug(
            "Query classified as %r with resolved_query %r.",
            classification.query_type, classification.resolved_query,
        )

        base = {
            "query_type": classification.query_type,
            "resolved_query": classification.resolved_query,
            "retrieval_triggered_by": "analysis",
            "pipeline_usage": [response.usage],
        }

        if classification.query_type == "out_of_scope":
            return {**base, "answer": GenerationResult(
                answer="This question can't be answered from SEC 10-K filings.",
                citations=[],
            )}

        return base

    return classify_query


def _build_user_message(state: GenerationState) -> str:
    parts = [f"Query: {state['query']}"]

    query_filter = state.get("query_filter")
    if query_filter:
        filter_parts = []
        if query_filter.ticker:
            filter_parts.append(f"ticker={query_filter.ticker}")
        if query_filter.form_type:
            filter_parts.append(f"form_type={query_filter.form_type}")
        if query_filter.fiscal_year_end:
            filter_parts.append(f"fiscal_year_end={query_filter.fiscal_year_end}")
        if query_filter.section:
            filter_parts.append(f"section={query_filter.section}")
        if filter_parts:
            parts.append(f"Pre-applied filter: {', '.join(filter_parts)}")

    history = state.get("history") or []
    if history:
        history_lines = "\n".join(f"{m.role}: {m.content}" for m in history)
        parts.append(f"Conversation history:\n{history_lines}")

    return "\n\n".join(parts)
