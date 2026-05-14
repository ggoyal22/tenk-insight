import logging
from collections.abc import Callable
from typing import TypedDict

from generation.nodes._stream import get_writer

from etl.embedder.base import Embedder
from retrieval.retriever import Retriever
from generation.types import RetrievalTask

logger = logging.getLogger(__name__)


class RetrieveInput(TypedDict):
    task: RetrievalTask


def make_retrieve(retriever: Retriever, embedder: Embedder) -> Callable[[RetrieveInput], dict]:
    def retrieve(state: RetrieveInput) -> dict:
        task = state["task"]
        hyde_query = task.hyde_query

        write = get_writer()
        f = task.filter
        query_snippet = task.keyword_query[:50]
        if f and (f.ticker or f.form_type):
            parts = [p for p in [f.ticker, f.form_type] if p]
            if f.fiscal_year:
                parts.append(f"({f.fiscal_year})")
            write(f"Searching {' '.join(parts)} · {query_snippet}...")
        else:
            write(f"Searching: {query_snippet}...")

        # Embed hyde_query for vector search when available — the hypothetical passage
        # is closer in embedding space to relevant chunks than the raw question.
        # Keyword search always uses the raw query (handled inside Retriever).
        # Embedding errors (model failure, OOM) are intentionally not caught here —
        # they propagate up and surface as a clear graph-level failure rather than
        # silently degrading to keyword-only retrieval.
        query_to_embed = hyde_query if hyde_query else task.semantic_query
        semantic_embedding = embedder.embed([query_to_embed])[0]

        results = retriever.retrieve(
            keyword_query=task.keyword_query,
            semantic_embedding=semantic_embedding,
            rerank_query=task.semantic_query,
            filters=task.filter,
        )

        logger.debug(
            "Retrieved %d result(s) for keyword_query %r (filter=%s).",
            len(results), task.keyword_query, task.filter,
        )

        return {"completed_results": [results]}

    return retrieve
