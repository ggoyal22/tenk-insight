import logging
from typing import Literal
from uuid import UUID

from db.client.base import DatabaseClient
from db.vector.base import SearchResult, VectorStore

logger = logging.getLogger(__name__)

# Maps config distance function name to pgvector operator and ops class
_DISTANCE_OPERATOR = {
    "cosine": "<=>",
    "l2":     "<->",
    "dot":    "<#>",
}


class PgvectorStore(VectorStore):
    def __init__(
        self,
        client: DatabaseClient,
        similarity_threshold: float,
        distance_function: Literal["cosine", "l2", "dot"],
        table: str = "chunks",
        embedding_col: str = "embedding",
    ) -> None:
        self._client = client
        self._similarity_threshold = similarity_threshold
        self._operator = _DISTANCE_OPERATOR[distance_function]
        self._table = table
        self._embedding_col = embedding_col

    def upsert(self, chunk_id: UUID, embedding: list[float], metadata: dict) -> None:
        embedding_model = metadata.get("embedding_model")
        if not embedding_model:
            raise ValueError(
                f"metadata must include 'embedding_model' when upserting an embedding "
                f"(chunk_id={chunk_id}). The DB enforces that embedding and embedding_model "
                f"are always set together."
            )
        sql = f"""
            UPDATE {self._table}
            SET {self._embedding_col} = %s, embedding_model = %s, embedded_at = NOW()
            WHERE id = %s
        """
        with self._client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (embedding, embedding_model, str(chunk_id)))
            conn.commit()

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        # pgvector's operators return distance (lower = more similar), so we
        # convert to similarity score: score = 1 - distance
        where_clauses = [f"1 - (embedding {self._operator} %s) >= %s"]
        params: list = [query_vector, self._similarity_threshold]

        if filters:
            for col, val in filters.items():
                where_clauses.append(f"{col} = %s")
                params.append(val)

        where = " AND ".join(where_clauses)
        sql = f"""
            SELECT id, 1 - ({self._embedding_col} {self._operator} %s) AS score,
                   filing_id, section, chunk_type
            FROM {self._table}
            WHERE {where}
            ORDER BY {self._embedding_col} {self._operator} %s ASC
            LIMIT %s
        """
        # query_vector appears 3 times: score calc, WHERE threshold, ORDER BY
        params = [query_vector] + params + [query_vector, top_k]

        with self._client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        return [
            SearchResult(
                chunk_id=UUID(str(row[0])),
                score=float(row[1]),
                metadata={"filing_id": row[2], "section": row[3], "chunk_type": row[4]},
            )
            for row in rows
        ]

    def delete_embedding(self, chunk_id: UUID) -> bool:
        sql = f"""
            UPDATE {self._table}
            SET {self._embedding_col} = NULL, embedding_model = NULL, embedded_at = NULL
            WHERE id = %s
        """
        with self._client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (str(chunk_id),))
                deleted = cur.rowcount > 0
            conn.commit()
        return deleted
