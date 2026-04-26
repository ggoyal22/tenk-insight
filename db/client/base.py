from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Generator, Protocol, runtime_checkable


@runtime_checkable
class DatabaseConnection(Protocol):
    def cursor(self) -> Any: ...
    def commit(self) -> None: ...
    def rollback(self) -> None: ...
    def close(self) -> None: ...


class Transaction(ABC):
    """Opaque handle for an in-progress transaction.

    Passed to repository methods to enlist them in the caller's transaction.
    Internal structure is backend-specific; callers must not inspect it.
    """
    pass


class DatabaseClient(ABC):
    @abstractmethod
    def get_connection(self) -> DatabaseConnection: ...

    @abstractmethod
    def release_connection(self, conn: DatabaseConnection) -> None: ...

    @abstractmethod
    def health_check(self) -> bool: ...

    @abstractmethod
    def close(self) -> None: ...

    @contextmanager
    def connection(self) -> Generator[DatabaseConnection, None, None]:
        conn = self.get_connection()
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        finally:
            self.release_connection(conn)

    @contextmanager
    @abstractmethod
    def transaction(self) -> Generator[Transaction, None, None]:
        """Start a transaction and yield an opaque handle.

        Commits on clean exit, rolls back on exception, and releases the
        connection in both cases. Pass the yielded handle to repository methods
        to enlist them in this transaction.
        """
        ...
