import logging
from collections.abc import Callable

from llm.base import BaseLLM
from llm.types import Message
from generation.prompts import HYDE_PROMPT
from generation.types import GenerationState

logger = logging.getLogger(__name__)


def make_hyde_expand(llm: BaseLLM) -> Callable[[GenerationState], dict]:
    def hyde_expand(state: GenerationState) -> dict:
        messages = [
            Message(role="system", content=HYDE_PROMPT),
            Message(role="user", content=state["query"]),
        ]
        response = llm.chat(messages)
        logger.debug("HyDE passage generated (%d chars).", len(response.content))
        return {"hyde_query": response.content}

    return hyde_expand
