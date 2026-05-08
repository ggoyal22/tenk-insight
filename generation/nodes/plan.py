import logging
from collections.abc import Callable

from db.repositories.filings import FilingsRepo
from generation.nodes._stream import get_writer
from generation.types import GenerationResult, GenerationState, TaskPlan
from llm.base import BaseLLM
from llm.types import Message

logger = logging.getLogger(__name__)


def make_plan_tasks(
    llm: BaseLLM,
    plan_prompts: dict[str, str],
    filings_repo: FilingsRepo,
) -> Callable[[GenerationState], dict]:
    def plan_tasks(state: GenerationState) -> dict:
        write = get_writer()
        query_type = state["query_type"]
        resolved = state.get("resolved_query") or state["query"]

        messages = [
            Message(role="system", content=plan_prompts[query_type]),
            Message(role="user", content=_build_user_message(resolved, state)),
        ]
        response = llm.chat_structured(messages, TaskPlan)
        plan = response.parsed

        n = len(plan.tasks)
        write(f"{n} search task{'s' if n != 1 else ''} planned")
        logger.debug("Planned %d task(s) for query type %r.", n, query_type)

        base = {
            "pending_tasks": plan.tasks,
            "pipeline_usage": [response.usage],
        }

        if not plan.tasks:
            return {**base, "answer": GenerationResult(
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
            if len(missing) == len(tickers):
                ticker_list = ", ".join(missing)
                return {**base, "pending_tasks": [], "answer": GenerationResult(
                    answer=f"I don't have filings indexed for {ticker_list}.",
                    citations=[],
                )}

        return base

    return plan_tasks


def _build_user_message(resolved_query: str, state: GenerationState) -> str:
    parts = [f"Query: {resolved_query}"]

    query_filter = state.get("query_filter")
    if query_filter:
        filter_parts = []
        if query_filter.ticker:
            filter_parts.append(f"ticker={query_filter.ticker}")
        if query_filter.form_type:
            filter_parts.append(f"form_type={query_filter.form_type}")
        if query_filter.fiscal_year_end:
            filter_parts.append(f"fiscal_year_end={query_filter.fiscal_year_end}")
        if filter_parts:
            parts.append(f"Pre-applied filter: {', '.join(filter_parts)}")

    return "\n\n".join(parts)
