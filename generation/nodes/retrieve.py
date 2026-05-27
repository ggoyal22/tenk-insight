import logging
from collections.abc import Callable
from typing import TypedDict

from generation.nodes._stream import get_writer

from etl.embedder.base import Embedder
from retrieval.retriever import Retriever
from generation.types import GenerationResult, RetrievalTask

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

        # Use the pre-computed embedding from embed_queries when available (batch path).
        # Falls back to single-query embedding for check_hop/reflect-triggered retrieves.
        # Keyword search always uses the raw query (handled inside Retriever).
        try:
            query_to_embed = hyde_query if hyde_query else task.semantic_query
            semantic_embedding = task.query_embedding if task.query_embedding is not None else embedder.embed([query_to_embed])[0]
            results = retriever.retrieve(
                keyword_query=task.keyword_query,
                semantic_embedding=semantic_embedding,
                rerank_query=task.semantic_query,
                filters=task.filter,
            )
        except Exception:
            logger.exception("retrieve failed for query %r.", task.keyword_query)
            return {
                "completed_results": [[]],
                "has_error": True,
                "answer": GenerationResult(answer="Something went wrong — please try again.", citations=[]),
            }

        logger.debug(
            "Retrieved %d result(s) for keyword_query %r (filter=%s).",
            len(results), task.keyword_query, task.filter,
        )

        if not results:
            return {
                "completed_results": [results],
                "failed_queries": [task.model_copy(update={"hyde_query": None, "query_embedding": None})],
            }

        for r in results:
            r.keyword_query = task.keyword_query
            r.semantic_query = task.semantic_query

        return {"completed_results": [results]}

    return retrieve
