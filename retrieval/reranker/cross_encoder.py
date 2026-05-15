import logging
from uuid import UUID

from sentence_transformers import CrossEncoder

from db.models import ChunkRecord
from retrieval.reranker.base import BaseReranker

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
        chunks: list[ChunkRecord],
    ) -> list[tuple[UUID, float]]:
        if not chunks:
            return []

        pairs = [(query, c.text) for c in chunks]
        scores: list[float] = self._model.predict(pairs).tolist()

        ranked = sorted(zip([c.id for c in chunks], scores), key=lambda x: x[1], reverse=True)
        return ranked
