import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from generation.nodes._stream import get_writer
from generation.types import GenerationState, RetrievalTask
from generation.token_limits import MAX_TOKENS_HYDE
from llm.base import BaseLLM, LLMError
from llm.types import LLMUsage, Message

logger = logging.getLogger(__name__)


def make_hyde_expand(llm: BaseLLM, prompt: str) -> Callable[[GenerationState], dict]:
    def hyde_expand(state: GenerationState) -> dict:
        write = get_writer()
        tasks = state["pending_tasks"]
        total = len(tasks)
        write(f"Expanding {total} search {'query' if total == 1 else 'queries'}...")

        def _expand_one(task: RetrievalTask) -> tuple[RetrievalTask, LLMUsage | None]:
            messages = [
                Message(role="system", content=prompt),
                Message(role="user", content=task.semantic_query),
            ]
            try:
                response = llm.chat(messages, max_tokens=MAX_TOKENS_HYDE)
            except LLMError:
                logger.exception("hyde_expand failed for query %r — skipping expansion.", task.semantic_query)
                return task, None
            logger.debug("HyDE passage generated for query %r (%d chars).", task.semantic_query, len(response.content))
            return task.model_copy(update={"hyde_query": response.content}), response.usage

        with ThreadPoolExecutor(max_workers=total) as executor:
            results = list(executor.map(_expand_one, tasks))

        updated_tasks = [r[0] for r in results]
        usages = [r[1] for r in results if r[1] is not None]
        return {"pending_tasks": updated_tasks, "pipeline_usage": usages}

    return hyde_expand
