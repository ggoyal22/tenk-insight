from uuid import UUID

from db.client.base import DatabaseClient
from db.models import ParentChunkRecord
from db.repositories.postgres.base import PostgresRepository


class ParentChunksRepository(PostgresRepository[ParentChunkRecord]):
    def __init__(self, client: DatabaseClient) -> None:
        super().__init__(client)

    @property
    def _auto_columns(self) -> set[str]:
        return {"id", "created_at", "updated_at"}

    @property
    def _table(self) -> str:
        return "parent_chunks"

    @property
    def _model_class(self) -> type:
        return ParentChunkRecord

    @property
    def _updatable_columns(self) -> set[str]:
        return {"section"}

    # ── ParentChunks-specific queries ─────────────────────────────────────────

    def get_by_filing_id(self, filing_id: UUID) -> list[ParentChunkRecord]:
        col_clause = ", ".join(self._columns)
        sql = (
            f"SELECT {col_clause} FROM {self._table} "
            f"WHERE filing_id = %s ORDER BY chunk_index ASC"
        )
        rows = self._execute_returning(sql, (str(filing_id),))
        return [self._row_to_model(row, self._columns) for row in rows]

    def exists_by_content_hash(self, content_hash: str) -> bool:
        sql = f"SELECT EXISTS(SELECT 1 FROM {self._table} WHERE content_hash = %s)"
        rows = self._execute_returning(sql, (content_hash,))
        return rows[0][0]

    def insert_many(self, records: list[ParentChunkRecord]) -> None:
        cols = [c for c in self._columns if c not in self._auto_columns]
        col_clause = ", ".join(cols)
        placeholders = ", ".join(["%s"] * len(cols))
        sql = f"INSERT INTO {self._table} ({col_clause}) VALUES ({placeholders})"
        params_list = [self._model_to_params(r) for r in records]
        self._execute_many(sql, params_list)
