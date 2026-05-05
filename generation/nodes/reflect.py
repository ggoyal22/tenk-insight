import logging
from collections.abc import Callable

from config.loader import GenerationConfig
from llm.base import BaseLLM
from llm.types import Message
from generation.nodes._context import build_context
from generation.types import GenerationState, ReflectionDecision

logger = logging.getLogger(__name__)


def make_reflect(llm: BaseLLM, config: GenerationConfig, prompt: str) -> Callable[[GenerationState], dict]:
    def reflect(state: GenerationState) -> dict:
        answer = state["answer"]
        context = build_context([r for group in state["completed_results"] for r in group])
        messages = [
            Message(role="system", content=prompt),
            Message(
                role="user",
                content=(
                    f"Question: {state['query']}\n\n"
                    f"Answer:\n{answer.answer}\n\n"
                    f"Context:\n{context}"
                ),
            ),
        ]
        response = llm.chat_structured(messages, ReflectionDecision)
        decision = response.parsed

        pending_tasks = (
            [decision.next_task]
            if decision.quality == "low" and decision.next_task
            else []
        )
        if decision.quality == "low":
            logger.warning(
                "Reflection quality=low reason=%r next_task=%r reflection_count=%d.",
                decision.reason,
                decision.next_task.query if decision.next_task else None,
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
        }

    return reflect
