from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from uuid import UUID

T = TypeVar("T")


class RelationalRepository(ABC, Generic[T]):
    @abstractmethod
    def insert(self, record: T) -> UUID: ...

    @abstractmethod
    def get_by_id(self, id: UUID) -> T | None: ...

    @abstractmethod
    def update(self, id: UUID, updates: dict) -> T | None: ...

    @abstractmethod
    def delete(self, id: UUID) -> bool: ...

    @abstractmethod
    def exists(self, id: UUID) -> bool: ...

    @abstractmethod
    def list(self, filters: dict | None, limit: int, offset: int) -> list[T]: ...
