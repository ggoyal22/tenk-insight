import logging
from typing import Literal
from uuid import UUID

from db.client.base import DatabaseClient
from db.vector.base import SearchResult, VectorStore

logger = logging.getLogger(__name__)

_DISTANCE_OPERATOR = {
    "cosine": "<=>",
    "l2":     "<->",
    "dot":    "<#>",
}

# Maps (distance_function, quantization) to pgvector HNSW ops class.
# Kept here as a reference mirror of db/setup.py — both must stay in sync.
_HNSW_OPS_CLASS: dict[tuple[str, str], str] = {
    ("cosine", "none"):    "vector_cosine_ops",
    ("cosine", "halfvec"): "halfvec_cosine_ops",
    ("cosine", "scalar"):  "int8_cosine_ops",
    ("l2",     "none"):    "vector_l2_ops",
    ("l2",     "halfvec"): "halfvec_l2_ops",
    ("dot",    "none"):    "vector_ip_ops",
    ("dot",    "halfvec"): "halfvec_ip_ops",
}


class PgvectorStore(VectorStore):
    def __init__(
        self,
        client: DatabaseClient,
        similarity_threshold: float,
        distance_function: Literal["cosine", "l2", "dot"],
        embedding_dimension: int,
        quantization: Literal["none", "halfvec", "scalar"] = "halfvec",
        table: str = "chunks",
    ) -> None:
        self._client = client
        self._similarity_threshold = similarity_threshold
        self._operator = _DISTANCE_OPERATOR[distance_function]
        self._dim = embedding_dimension
        self._quantization = quantization
        self._table = table

        # Pre-compute the SQL fragments that depend on quantization so they
        # are not recomputed on every search call.
        if quantization == "halfvec":
            self._emb_expr = f"embedding::halfvec({embedding_dimension})"
            self._q_cast = f"::halfvec({embedding_dimension})"
        elif quantization == "scalar":
            self._emb_expr = f"embedding::int8({embedding_dimension})"
            self._q_cast = f"::int8({embedding_dimension})"
        else:  # none — use raw float32; cast needed so psycopg2 list binds as vector not numeric[]
            self._emb_expr = "embedding"
            self._q_cast = f"::vector({embedding_dimension})"

    def upsert(self, chunk_id: UUID, embedding: list[float], metadata: dict) -> None:
        embedding_model = metadata.get("embedding_model")
        if not embedding_model:
            raise ValueError(
                f"metadata must include 'embedding_model' when upserting an embedding "
                f"(chunk_id={chunk_id})."
            )
        sql = f"""
            UPDATE {self._table}
            SET embedding = %s, embedding_model = %s, embedded_at = NOW()
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
        filing_ids: list[UUID] | None = None,
        section: str | None = None,
        include_embedding: bool = False,
    ) -> list[SearchResult]:
        op = self._operator
        e = self._emb_expr
        qc = self._q_cast

        # Build WHERE clause
        # Score = 1 - distance; filter rows below similarity threshold
        where_parts = [f"1 - ({e} {op} %s{qc}) >= %s"]
        where_params: list = [query_vector, self._similarity_threshold]

        if filing_ids is not None:
            where_parts.append("filing_id = ANY(%s::uuid[])")
            where_params.append([str(fid) for fid in filing_ids])

        if section is not None:
            where_parts.append("section = %s")
            where_params.append(section)

        where = " AND ".join(where_parts)

        extra_col = ", embedding" if include_embedding else ""
        sql = f"""
            SELECT id,
                   1 - ({e} {op} %s{qc}) AS score,
                   filing_id, section, chunk_type{extra_col}
            FROM {self._table}
            WHERE {where}
            ORDER BY {e} {op} %s{qc} ASC
            LIMIT %s
        """
        # query_vector appears three times: SELECT score, WHERE score, ORDER BY
        all_params = [query_vector] + where_params + [query_vector, top_k]

        with self._client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, all_params)
                rows = cur.fetchall()

        results = []
        for row in rows:
            raw_emb = row[5] if include_embedding and len(row) > 5 else None
            if raw_emb is None:
                embedding = None
            elif isinstance(raw_emb, str):
                embedding = [float(x.strip()) for x in raw_emb[1:-1].split(",") if x.strip()]
            else:
                embedding = [float(x) for x in raw_emb]
            results.append(SearchResult(
                chunk_id=UUID(str(row[0])),
                score=float(row[1]),
                metadata={"filing_id": row[2], "section": row[3], "chunk_type": row[4]},
                embedding=embedding,
            ))
        return results

    def delete_embedding(self, chunk_id: UUID) -> bool:
        sql = f"""
            UPDATE {self._table}
            SET embedding = NULL, embedding_model = NULL, embedded_at = NULL
            WHERE id = %s
        """
        with self._client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (str(chunk_id),))
                deleted = cur.rowcount > 0
            conn.commit()
        return deleted
