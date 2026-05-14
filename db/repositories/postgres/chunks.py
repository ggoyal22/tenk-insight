from uuid import UUID

from db.client.base import DatabaseClient, Transaction
from db.models import ChunkRecord
from db.repositories.chunks import ChunksRepo
from db.repositories.postgres.base import PostgresRepository


class PostgresChunksRepository(ChunksRepo, PostgresRepository[ChunkRecord]):
    def __init__(self, client: DatabaseClient) -> None:
        super().__init__(client)

    @property
    def _auto_columns(self) -> set[str]:
        return {"id", "created_at", "updated_at"}

    @property
    def _table(self) -> str:
        return "chunks"

    @property
    def _model_class(self) -> type:
        return ChunkRecord

    @property
    def _updatable_columns(self) -> set[str]:
        return {"section", "embedding", "embedding_model", "embedded_at"}

    # ── Chunks-specific queries ───────────────────────────────────────────────

    def get_by_filing_id(
        self, filing_id: UUID, tx: Transaction | None = None
    ) -> list[ChunkRecord]:
        col_clause = ", ".join(self._columns)
        sql = (
            f"SELECT {col_clause} FROM {self._table} "
            f"WHERE filing_id = %s ORDER BY chunk_index ASC"
        )
        rows = self._execute_returning(sql, (str(filing_id),), tx=tx)
        return [self._row_to_model(row, self._columns) for row in rows]

    def get_unembedded(self, limit: int = 100) -> list[ChunkRecord]:
        col_clause = ", ".join(self._columns)
        sql = (
            f"SELECT {col_clause} FROM {self._table} "
            f"WHERE embedded_at IS NULL ORDER BY created_at ASC LIMIT %s"
        )
        rows = self._execute_returning(sql, (limit,))
        return [self._row_to_model(row, self._columns) for row in rows]

    def exists_by_content_hash(self, content_hash: str) -> bool:
        sql = f"SELECT EXISTS(SELECT 1 FROM {self._table} WHERE content_hash = %s)"
        rows = self._execute_returning(sql, (content_hash,))
        return rows[0][0]

    def insert_many(
        self, records: list[ChunkRecord], tx: Transaction | None = None
    ) -> None:
        cols = [c for c in self._columns if c not in self._auto_columns]
        col_clause = ", ".join(cols)
        placeholders = ", ".join(["%s"] * len(cols))
        sql = f"INSERT INTO {self._table} ({col_clause}) VALUES ({placeholders})"
        params_list = [self._model_to_params(r) for r in records]
        self._execute_many(sql, params_list, tx=tx)

    def get_by_ids_no_embedding(self, ids: list[UUID]) -> list[ChunkRecord]:
        if not ids:
            return []
        cols = [c for c in self._columns if c != "embedding"]
        col_clause = ", ".join(cols)
        sql = f"SELECT {col_clause} FROM {self._table} WHERE id = ANY(%s::uuid[])"
        rows = self._execute_returning(sql, ([str(i) for i in ids],))
        return [self._row_to_model(row, cols) for row in rows]

    # ── Keyword search (FTS implementation) ──────────────────────────────────
    # Uses PostgreSQL tsvector/tsquery. A future Pg_bm25ChunksRepository would
    # subclass this and override keyword_search with pg_bm25 SQL.

    _QUERY_FN: dict[str, str] = {
        "standard": "plainto_tsquery",
        "phrase":   "phraseto_tsquery",
        "web":      "websearch_to_tsquery",
    }

    def keyword_search(
        self,
        query: str,
        top_k: int,
        filing_ids: list[UUID] | None = None,
        section: str | None = None,
        query_mode: str = "web",
    ) -> list[tuple[UUID, float]]:
        query_fn = self._QUERY_FN.get(query_mode, "websearch_to_tsquery")

        where_parts = [f"search_vector @@ {query_fn}('english', %s)"]
        params: list = [query]

        if filing_ids is not None:
            where_parts.append("filing_id = ANY(%s::uuid[])")
            params.append([str(fid) for fid in filing_ids])

        if section is not None:
            where_parts.append("section = %s")
            params.append(section)

        where = " AND ".join(where_parts)
        # query appears twice: once in WHERE tsquery and once for ts_rank scoring
        sql = f"""
            SELECT id, ts_rank(search_vector, {query_fn}('english', %s)) AS score
            FROM {self._table}
            WHERE {where}
            ORDER BY score DESC
            LIMIT %s
        """
        all_params = [query] + params + [top_k]
        rows = self._execute_returning(sql, tuple(all_params))
        return [(UUID(str(row[0])), float(row[1])) for row in rows]
