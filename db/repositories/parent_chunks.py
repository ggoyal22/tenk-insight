from abc import abstractmethod
from uuid import UUID

from db.models import ParentChunkRecord
from db.repositories.base import RelationalRepository


class ParentChunksRepo(RelationalRepository[ParentChunkRecord]):
    @abstractmethod
    def get_by_filing_id(self, filing_id: UUID) -> list[ParentChunkRecord]: ...

    @abstractmethod
    def exists_by_content_hash(self, content_hash: str) -> bool: ...

    @abstractmethod
    def insert_many(self, records: list[ParentChunkRecord]) -> None: ...
