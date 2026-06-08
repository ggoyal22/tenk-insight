import logging
from uuid import UUID

import numpy as np

from db.vector.base import VectorStore
from retrieval.vector.base import BaseVectorRetriever

logger = logging.getLogger(__name__)


class PgvectorRetriever(BaseVectorRetriever):
    """Vector retriever backed by pgvector.

    Two-phase retrieval:
      1. ANN search using the quantized HNSW index (fast, approximate).
      2. Exact float32 cosine rescore of the oversampled candidates (precise).

    oversample_k controls how many candidates are fetched in phase 1 before
    rescoring narrows them to top_k.
    """

    def __init__(self, vector_store: VectorStore, oversample_k: int) -> None:
        self._store = vector_store
        self._oversample_k = oversample_k

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        filing_ids: list[UUID] | None = None,
        section: str | None = None,
        exclude_parent_ids: list[UUID] | None = None,
    ) -> list[tuple[UUID, float]]:
        candidates = self._store.search(
            query_vector=query_embedding,
            top_k=self._oversample_k,
            filing_ids=filing_ids,
            section=section,
            exclude_parent_ids=exclude_parent_ids,
            include_embedding=True,
        )

        if not candidates:
            logger.debug(
                "Vector search returned no candidates (filing_ids=%s, section=%s).",
                filing_ids,
                section,
            )
            return []

        q = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            logger.warning(
                "Query embedding has zero norm — this indicates a problem with embedding "
                "generation. Returning no results."
            )
            return []
        q = q / q_norm

        rescored: list[tuple[UUID, float]] = []
        for result in candidates:
            if result.embedding is None:
                continue
            v = np.array(result.embedding, dtype=np.float32)
            v_norm = np.linalg.norm(v)
            if v_norm == 0:
                logger.warning(
                    "Candidate chunk %s has a zero-norm embedding — skipping. "
                    "Re-embed this chunk to fix it.",
                    result.chunk_id,
                )
                continue
            v = v / v_norm
            rescored.append((result.chunk_id, float(np.dot(q, v))))

        rescored.sort(key=lambda x: x[1], reverse=True)
        return rescored[:top_k]
