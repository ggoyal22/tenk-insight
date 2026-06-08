import logging
from collections.abc import Callable

from generation.nodes._stream import get_writer

from config.loader import GenerationConfig
from llm.base import BaseLLM, LLMError
from llm.types import Message
from generation.nodes._context import build_context
from generation.token_limits import MAX_TOKENS_REFLECT
from generation.types import GenerationResult, GenerationState, ReflectionDecision, RetrievalTask

logger = logging.getLogger(__name__)


def make_reflect(llm: BaseLLM, config: GenerationConfig, prompt: str) -> Callable[[GenerationState], dict]:
    def reflect(state: GenerationState) -> dict:
        get_writer()("Reviewing answer quality...")
        answer = state["answer"]
        context = build_context([r for group in state["completed_results"] for r in group])
        messages = [
            Message(role="system", content=prompt),
            Message(
                role="user",
                content=(
                    f"Question: {state.get('resolved_query') or state['query']}\n\n"
                    f"Answer:\n{answer.answer}\n\n"
                    f"Context:\n{context}"
                ),
            ),
        ]
        try:
            response = llm.chat_structured(messages, ReflectionDecision, max_tokens=MAX_TOKENS_REFLECT)
        except LLMError:
            logger.exception("reflect failed — accepting current answer.")
            return {
                "pending_tasks": [],
                "reflection_count": state["reflection_count"] + 1,
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
            if decision.quality == "low" and decision.next_task
            else []
        )
        if decision.quality == "low":
            logger.warning(
                "Reflection quality=low reason=%r next_task=%r reflection_count=%d.",
                decision.reason,
                decision.next_task.keyword_query if decision.next_task else None,
                state["reflection_count"] + 1,
            )
        else:
            logger.info(
                "Reflection quality=high reflection_count=%d.",
                state["reflection_count"] + 1,
            )

        return {
            "pending_tasks": pending_tasks,
            "reflection_count": state["reflection_count"] + 1,
            "retrieval_triggered_by": "reflect",
            "pipeline_usage": [response.usage],
        }

    return reflect
