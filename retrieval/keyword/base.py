from abc import ABC, abstractmethod
from uuid import UUID


class BaseKeywordRetriever(ABC):
    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int,
        filing_ids: list[UUID] | None = None,
        section: str | None = None,
        exclude_parent_ids: list[UUID] | None = None,
    ) -> list[tuple[UUID, float]]:
        """Return (chunk_id, score) pairs ordered by relevance descending.

        filing_ids and section are pre-filters; pass None to skip filtering.
        exclude_parent_ids drops chunks belonging to those parent chunks before
        ranking; pass None or an empty list to skip.
        Score semantics are implementation-defined (e.g. ts_rank for FTS, BM25 score).
        Only the relative ordering matters for RRF fusion downstream.
        """
        ...
