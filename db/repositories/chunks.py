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
