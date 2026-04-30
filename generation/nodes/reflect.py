import logging
from collections.abc import Callable

from config.loader import GenerationConfig
from llm.base import BaseLLM
from llm.types import Message
from generation.nodes._context import build_context
from generation.prompts import REFLECTION_PROMPT
from generation.types import GenerationState, ReflectionDecision

logger = logging.getLogger(__name__)


def make_reflect(llm: BaseLLM, config: GenerationConfig) -> Callable[[GenerationState], dict]:
    def reflect(state: GenerationState) -> dict:
        answer = state["answer"]
        context = build_context(state["completed_results"])
        messages = [
            Message(role="system", content=REFLECTION_PROMPT),
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
        logger.debug(
            "Reflection decision: quality=%r, reflection_count=%d.",
            decision.quality, state["reflection_count"] + 1,
        )

        return {
            "pending_tasks": pending_tasks,
            "reflection_count": state["reflection_count"] + 1,
            "retrieval_triggered_by": "reflect",
        }

    return reflect
