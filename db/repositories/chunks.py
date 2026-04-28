from abc import abstractmethod
from uuid import UUID

from db.client.base import Transaction
from db.models import ChunkRecord
from db.repositories.base import RelationalRepository


class ChunksRepo(RelationalRepository[ChunkRecord]):
    @abstractmethod
    def get_by_filing_id(self, filing_id: UUID, tx: Transaction | None = None) -> list[ChunkRecord]: ...

    @abstractmethod
    def get_unembedded(self, limit: int = 100) -> list[ChunkRecord]: ...

    @abstractmethod
    def exists_by_content_hash(self, content_hash: str) -> bool: ...

    @abstractmethod
    def insert_many(self, records: list[ChunkRecord], tx: Transaction | None = None) -> None: ...

    @abstractmethod
    def keyword_search(
        self,
        query: str,
        top_k: int,
        filing_ids: list[UUID] | None = None,
        section: str | None = None,
    ) -> list[tuple[UUID, float]]:
        """Full-text search over chunk text using tsvector/tsquery.

        Returns (chunk_id, ts_rank score) pairs ordered by relevance descending.
        filing_ids and section are optional pre-filters applied before ranking.
        """
        ...
