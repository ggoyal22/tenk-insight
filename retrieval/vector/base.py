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
        exclude_parent_ids: list[UUID] | None = None,
    ) -> list[tuple[UUID, float]]:
        """Return (chunk_id, score) pairs ordered by relevance descending.

        filing_ids and section are pre-filters; pass None to skip filtering.
        exclude_parent_ids drops chunks belonging to those parent chunks before
        ranking; pass None or an empty list to skip.
        top_k is the number of results to return after any internal rescoring.
        """
        ...
