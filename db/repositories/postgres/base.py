import dataclasses
import logging
from abc import abstractmethod
from typing import Generic, TypeVar
from uuid import UUID

from db.client.base import DatabaseClient
from db.repositories.base import RelationalRepository

logger = logging.getLogger(__name__)

T = TypeVar("T")


class PostgresRepository(RelationalRepository[T], Generic[T]):
    def __init__(self, client: DatabaseClient) -> None:
        self._client = client

    # ── Abstract — concrete repos must provide ────────────────────────────────

    @property
    @abstractmethod
    def _auto_columns(self) -> set[str]: ...

    @property
    @abstractmethod
    def _table(self) -> str: ...

    @property
    @abstractmethod
    def _model_class(self) -> type: ...

    @property
    @abstractmethod
    def _updatable_columns(self) -> set[str]: ...

    # ── Derived from model_class — no duplication across concrete repos ───────

    @property
    def _columns(self) -> list[str]:
        return [f.name for f in dataclasses.fields(self._model_class)]

    def _row_to_model(self, row: tuple, columns: list[str]) -> T:
        return self._model_class(**dict(zip(columns, row)))

    def _model_to_params(self, record: T) -> tuple:
        data = dataclasses.asdict(record)
        return tuple(data[c] for c in self._columns if c not in self._auto_columns)

    # ── Shared plumbing ───────────────────────────────────────────────────────

    def _execute_returning(self, sql: str, params: tuple = ()) -> list[tuple]:
        """Run a query that returns rows — SELECT, INSERT RETURNING, UPDATE RETURNING."""
        with self._client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                results = cur.fetchall() if cur.description else []
            conn.commit()
            return results

    def _execute_rowcount(self, sql: str, params: tuple = ()) -> int:
        """Run a statement that returns affected row count — DELETE, UPDATE without RETURNING."""
        with self._client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                count = cur.rowcount
            conn.commit()
            return count

    def _execute_many(self, sql: str, params_list: list[tuple]) -> None:
        """Run a bulk write statement — INSERT of multiple rows."""
        with self._client.connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, params_list)
            conn.commit()

    # ── RelationalRepository implementation ───────────────────────────────────

    def insert(self, record: T) -> UUID:
        cols = [c for c in self._columns if c not in self._auto_columns]
        col_clause = ", ".join(cols)
        placeholders = ", ".join(["%s"] * len(cols))
        sql = f"INSERT INTO {self._table} ({col_clause}) VALUES ({placeholders}) RETURNING id"
        rows = self._execute_returning(sql, self._model_to_params(record))
        if not rows:
            raise RuntimeError(f"INSERT INTO {self._table} returned no rows — this is unexpected.")
        return UUID(str(rows[0][0]))

    def get_by_id(self, id: UUID) -> T | None:
        col_clause = ", ".join(self._columns)
        sql = f"SELECT {col_clause} FROM {self._table} WHERE id = %s"
        rows = self._execute_returning(sql, (str(id),))
        if not rows:
            return None
        return self._row_to_model(rows[0], self._columns)

    def update(self, id: UUID, updates: dict) -> T | None:
        invalid = updates.keys() - self._updatable_columns
        if invalid:
            raise ValueError(
                f"Cannot update non-updatable columns on {self._table}: {invalid}. "
                f"Allowed: {self._updatable_columns}"
            )
        set_clause = ", ".join(f"{k} = %s" for k in updates)
        col_clause = ", ".join(self._columns)
        sql = f"UPDATE {self._table} SET {set_clause} WHERE id = %s RETURNING {col_clause}"
        params = (*updates.values(), str(id))
        rows = self._execute_returning(sql, params)
        if not rows:
            return None
        return self._row_to_model(rows[0], self._columns)

    def delete(self, id: UUID) -> bool:
        sql = f"DELETE FROM {self._table} WHERE id = %s"
        return self._execute_rowcount(sql, (str(id),)) > 0

    def exists(self, id: UUID) -> bool:
        sql = f"SELECT EXISTS(SELECT 1 FROM {self._table} WHERE id = %s)"
        rows = self._execute_returning(sql, (str(id),))
        return rows[0][0]

    def list(self, filters: dict | None = None, limit: int = 100, offset: int = 0) -> list[T]:
        col_clause = ", ".join(self._columns)
        sql = f"SELECT {col_clause} FROM {self._table}"
        params: tuple = ()

        if filters:
            invalid = filters.keys() - set(self._columns)
            if invalid:
                raise ValueError(
                    f"Invalid filter columns for {self._table}: {invalid}. "
                    f"Allowed: {self._columns}"
                )
            where_clause = " AND ".join(f"{k} = %s" for k in filters)
            sql += f" WHERE {where_clause}"
            params = (*filters.values(),)

        sql += " LIMIT %s OFFSET %s"
        params = (*params, limit, offset)

        rows = self._execute_returning(sql, params)
        return [self._row_to_model(row, self._columns) for row in rows]
