from abc import ABC, abstractmethod
from uuid import UUID

from db.models import ChunkRecord


class BaseReranker(ABC):
    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: list[ChunkRecord],
    ) -> list[tuple[UUID, float]]:
        """Score child chunks by relevance against query.

        Returns (chunk_id, score) pairs sorted by score descending.
        """
        ...
