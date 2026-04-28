from abc import ABC, abstractmethod
from uuid import UUID


class BaseVectorRetriever(ABC):
    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        filing_ids: list[UUID] | None = None,
        section: str | None = None,
    ) -> list[tuple[UUID, float]]:
        """Return (chunk_id, score) pairs ordered by relevance descending.

        filing_ids and section are pre-filters; pass None to skip filtering.
        top_k is the number of results to return after any internal rescoring.
        """
        ...
