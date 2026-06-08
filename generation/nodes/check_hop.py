import logging
from collections.abc import Callable

from generation.nodes._stream import get_writer

from config.loader import GenerationConfig
from llm.base import BaseLLM, LLMError
from llm.types import Message
from generation.nodes._context import build_hop_context
from generation.token_limits import MAX_TOKENS_CHECK_HOP
from generation.types import GenerationResult, GenerationState, HopDecision, RetrievalTask

logger = logging.getLogger(__name__)


def make_check_hop(llm: BaseLLM, config: GenerationConfig, prompt: str) -> Callable[[GenerationState], dict]:
    def check_hop(state: GenerationState) -> dict:
        get_writer()("Evaluating whether more retrieval is needed...")
        context = build_hop_context([r for group in state["completed_results"] for r in group])
        messages = [
            Message(role="system", content=prompt),
            Message(
                role="user",
                content=(
                    f"Question: {state.get('resolved_query') or state['query']}\n\n"
                    f"Context retrieved so far (hop {state['hop_count'] + 1} of {config.hop.max_hops}):\n"
                    f"{context}"
                ),
            ),
        ]
        failed = state.get("failed_queries") or []
        failed_section = (
            "\n\nQueries already attempted that returned no results:\n"
            + "\n".join(
                f"- keyword: {t.keyword_query} | semantic: {t.semantic_query}"
                + (f" | filter: {t.filter}" if t.filter else "")
                for t in failed
            )
        ) if failed else ""

        messages[-1] = Message(
            role="user",
            content=messages[-1].content + failed_section,
        )

        try:
            response = llm.chat_structured(messages, HopDecision, max_tokens=MAX_TOKENS_CHECK_HOP)
        except LLMError:
            logger.exception("check_hop failed — stopping retrieval early.")
            return {
                "pending_tasks": [],
                "hop_count": state["hop_count"] + 1,
                "pipeline_usage": [],
                "has_error": True,
                "answer": GenerationResult(answer="Something went wrong — please try again.", citations=[]),
            }
        decision = response.parsed

        seen_parent_ids = list({
            r.parent_chunk.id
            for group in state["completed_results"]
            for r in group
        })
        pending_tasks = (
            [RetrievalTask.model_validate({
                **decision.next_task.model_dump(),
                "exclude_parent_ids": seen_parent_ids,
            })]
            if not decision.done and decision.next_task
            else []
        )
        logger.debug(
            "Hop decision: done=%s, hop_count=%d.", decision.done, state["hop_count"] + 1
        )

        return {
            "pending_tasks": pending_tasks,
            "hop_count": state["hop_count"] + 1,
            "retrieval_triggered_by": "check_hop",
            "pipeline_usage": [response.usage],
        }

    return check_hop
