from abc import ABC, abstractmethod
from dataclasses import dataclass
from uuid import UUID


@dataclass
class SearchResult:
    chunk_id: UUID
    score: float        # similarity score in [0, 1] — higher is more similar
    metadata: dict


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, chunk_id: UUID, embedding: list[float], metadata: dict) -> None: ...

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[SearchResult]: ...

    @abstractmethod
    def delete_embedding(self, chunk_id: UUID) -> bool: ...
