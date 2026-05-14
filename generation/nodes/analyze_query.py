import logging
from collections.abc import Callable

from db.repositories.filings import FilingsRepo
from generation.nodes._stream import get_writer
from generation.types import GenerationResult, GenerationState, QueryPlan
from generation.token_limits import MAX_TOKENS_ANALYZE
from llm.base import BaseLLM
from llm.types import Message

logger = logging.getLogger(__name__)


def make_analyze_query(
    llm: BaseLLM,
    prompt: str,
    filings_repo: FilingsRepo,
) -> Callable[[GenerationState], dict]:
    def analyze_query(state: GenerationState) -> dict:
        write = get_writer()
        write("Analyzing your question...")

        messages = [
            Message(role="system", content=prompt),
            Message(role="user", content=_build_user_message(state)),
        ]
        response = llm.chat_structured(messages, QueryPlan, max_tokens=MAX_TOKENS_ANALYZE)
        plan = response.parsed

        logger.debug(
            "Query analyzed: type=%r resolved_query=%r tasks=%d reasoning=%r.",
            plan.query_type, plan.resolved_query, len(plan.tasks), plan.reasoning,
        )

        base = {
            "query_type": plan.query_type,
            "resolved_query": plan.resolved_query,
            "retrieval_triggered_by": "analysis",
            "pipeline_usage": [response.usage],
        }

        if plan.query_type == "out_of_scope":
            return {**base, "answer": GenerationResult(
                answer="This question can't be answered from SEC 10-K filings.",
                citations=[],
            )}

        if not plan.tasks:
            return {**base, "pending_tasks": [], "answer": GenerationResult(
                answer="I had trouble understanding your question. Try rephrasing it.",
                citations=[],
            )}

        tickers = [
            t.filter.ticker.upper()
            for t in plan.tasks
            if t.filter and t.filter.ticker
        ]
        if tickers:
            missing = [t for t in tickers if not filings_repo.list_ids({"ticker": t})]
            if missing:
                found = [t for t in tickers if t not in missing]
                if found:
                    msg = (
                        f"I have filings for {', '.join(found)} but not for {', '.join(missing)}"
                        f" — I can't complete this comparison."
                    )
                else:
                    msg = f"I don't have filings indexed for {', '.join(missing)}."
                return {**base, "pending_tasks": [], "answer": GenerationResult(
                    answer=msg, citations=[],
                )}

        n = len(plan.tasks)
        write(f"{n} search task{'s' if n != 1 else ''} planned")
        return {**base, "pending_tasks": plan.tasks}

    return analyze_query


def _build_user_message(state: GenerationState) -> str:
    parts = [f"Query: {state['query']}"]

    query_filter = state.get("query_filter")
    if query_filter:
        filter_parts = []
        if query_filter.ticker:
            filter_parts.append(f"ticker={query_filter.ticker}")
        if query_filter.form_type:
            filter_parts.append(f"form_type={query_filter.form_type}")
        if query_filter.fiscal_year:
            filter_parts.append(f"fiscal_year={query_filter.fiscal_year}")
        if filter_parts:
            parts.append(f"Pre-applied filter: {', '.join(filter_parts)}")

    history = state.get("history") or []
    if history:
        history_lines = "\n".join(f"{m.role}: {m.content}" for m in history)
        parts.append(f"Conversation history:\n{history_lines}")

    return "\n\n".join(parts)
