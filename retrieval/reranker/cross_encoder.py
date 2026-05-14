import dataclasses
import logging
from uuid import UUID

from sentence_transformers import CrossEncoder

from retrieval.reranker.base import BaseReranker
from retrieval.types import RetrievalResult

logger = logging.getLogger(__name__)


class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_name: str) -> None:
        logger.info("Loading cross-encoder model: %s", model_name)
        try:
            self._model = CrossEncoder(model_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load cross-encoder model '{model_name}'. "
                "Ensure the model name is valid and you have internet access (or a local cache). "
                f"Original error: {e}"
            ) from e

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        if not results:
            return []

        seen: set[UUID] = set()
        unique_results: list[RetrievalResult] = []
        for r in results:
            if r.parent_chunk.id not in seen:
                seen.add(r.parent_chunk.id)
                unique_results.append(r)

        pairs = [(query, r.parent_chunk.text) for r in unique_results]
        scores: list[float] = self._model.predict(pairs).tolist()

        ranked = sorted(zip(unique_results, scores), key=lambda x: x[1], reverse=True)

        return [
            dataclasses.replace(result, reranker_score=float(score))
            for result, score in ranked[:top_k]
        ]
