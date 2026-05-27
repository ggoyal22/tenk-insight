import logging
from collections.abc import Callable

from etl.embedder.base import Embedder
from generation.types import GenerationState

logger = logging.getLogger(__name__)


def make_embed_queries(embedder: Embedder) -> Callable[[GenerationState], dict]:
    def embed_queries(state: GenerationState) -> dict:
        tasks = state["pending_tasks"]
        queries = [t.hyde_query if t.hyde_query else t.semantic_query for t in tasks]
        embeddings = embedder.embed(queries)
        updated_retrieval_tasks = [t.model_copy(update={"query_embedding": emb}) for t, emb in zip(tasks, embeddings)]
        logger.debug("Batch-embedded %d quer%s.", len(tasks), "y" if len(tasks) == 1 else "ies")
        return {"pending_tasks": updated_retrieval_tasks}

    return embed_queries
