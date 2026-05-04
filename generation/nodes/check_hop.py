import logging
from collections.abc import Callable

from config.loader import GenerationConfig
from llm.base import BaseLLM
from llm.types import Message
from generation.nodes._context import build_context
from generation.prompts import CHECK_HOP_PROMPT
from generation.types import GenerationState, HopDecision

logger = logging.getLogger(__name__)


def make_check_hop(llm: BaseLLM, config: GenerationConfig) -> Callable[[GenerationState], dict]:
    def check_hop(state: GenerationState) -> dict:
        context = build_context([r for group in state["completed_results"] for r in group])
        messages = [
            Message(role="system", content=CHECK_HOP_PROMPT),
            Message(
                role="user",
                content=(
                    f"Question: {state['query']}\n\n"
                    f"Context retrieved so far (hop {state['hop_count'] + 1} of {config.multi_hop.max_hops}):\n"
                    f"{context}"
                ),
            ),
        ]
        response = llm.chat_structured(messages, HopDecision)
        decision = response.parsed

        pending_tasks = (
            [decision.next_task]
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
        }

    return check_hop
