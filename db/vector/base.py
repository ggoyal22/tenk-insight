from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from uuid import UUID


@dataclass
class SearchResult:
    chunk_id: UUID
    score: float                            # similarity score in [0, 1] — higher is more similar
    metadata: dict = field(default_factory=dict)
    embedding: list[float] | None = None   # populated only when include_embedding=True


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, chunk_id: UUID, embedding: list[float], metadata: dict) -> None: ...

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int,
        filing_ids: list[UUID] | None = None,
        section: str | None = None,
        exclude_parent_ids: list[UUID] | None = None,
        include_embedding: bool = False,
    ) -> list[SearchResult]:
        """Return top-k results by vector similarity.

        filing_ids and section are optional pre-filters applied before ranking.
        exclude_parent_ids drops chunks belonging to those parent chunks before
        ranking; pass None or an empty list to skip.
        When include_embedding=True, each SearchResult carries the full float32
        embedding for downstream exact rescoring.
        """
        ...

    @abstractmethod
    def delete_embedding(self, chunk_id: UUID) -> bool: ...
