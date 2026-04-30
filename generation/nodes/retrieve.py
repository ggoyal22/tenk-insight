import logging
from collections.abc import Callable
from typing import TypedDict

from etl.embedder.base import Embedder
from retrieval.retriever import Retriever
from generation.types import RetrievalTask

logger = logging.getLogger(__name__)


class RetrieveInput(TypedDict):
    task: RetrievalTask
    hyde_query: str | None


def make_retrieve(retriever: Retriever, embedder: Embedder) -> Callable[[RetrieveInput], dict]:
    def retrieve(state: RetrieveInput) -> dict:
        task = state["task"]
        hyde_query = state["hyde_query"]

        # Embed hyde_query for vector search when available — the hypothetical passage
        # is closer in embedding space to relevant chunks than the raw question.
        # Keyword search always uses the raw query (handled inside Retriever).
        # Embedding errors (model failure, OOM) are intentionally not caught here —
        # they propagate up and surface as a clear graph-level failure rather than
        # silently degrading to keyword-only retrieval.
        query_to_embed = hyde_query if hyde_query else task.query
        embedding = embedder.embed([query_to_embed])[0]

        results = retriever.retrieve(
            query=task.query,
            query_embedding=embedding,
            filters=task.filter,
        )

        logger.debug(
            "Retrieved %d result(s) for query %r (filter=%s).",
            len(results), task.query, task.filter,
        )

        return {"completed_results": [results]}

    return retrieve
