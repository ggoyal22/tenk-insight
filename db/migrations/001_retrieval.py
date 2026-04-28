"""
Migration 001 — Add HNSW vector index and tsvector column for hybrid retrieval.

Applies three changes to an existing database:
  1. Drop the existing HNSW index on chunks.embedding.
  2. Create a new HNSW index using the quantization configured in config.yaml.
  3. Add search_vector (generated tsvector) column + GIN index for keyword search.

Safe to run on a live database. Idempotent — checks whether already applied before acting.

Usage:
    python db/migrations/001_retrieval.py
"""

import sys
from pathlib import Path

import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.loader import load_config  # noqa: E402  (sys.path must be set first)
from db.setup import _hnsw_index_col   # noqa: E402


def _already_applied(conn) -> bool:
    """Return True if search_vector column exists — indicates migration was applied."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'chunks' AND column_name = 'search_vector'
            )
        """)
        return cur.fetchone()[0]


def run() -> None:
    config = load_config()
    db = config.database

    # Direct psycopg2 connection — migrations are DDL (CREATE/DROP/ALTER), which sit
    # outside the application's DML abstraction layer. Config-driven connection params
    # are the same; only the DDL execution bypasses the repo layer.
    conn = psycopg2.connect(
        host=db.host,
        port=db.port,
        dbname=db.name,
        user=db.user,
        password=db.password.get_secret_value(),
    )
    try:
        if _already_applied(conn):
            print("Migration 001 already applied — skipping.")
            return

        vi = config.vector_index
        index_col = _hnsw_index_col(config)

        with conn.cursor() as cur:
            cur.execute("DROP INDEX IF EXISTS idx_chunks_embedding")

            cur.execute(f"""
                CREATE INDEX idx_chunks_embedding
                    ON chunks USING hnsw ({index_col})
                    WITH (m = {vi.hnsw_m}, ef_construction = {vi.hnsw_ef_construction})
            """)

            cur.execute("""
                ALTER TABLE chunks
                    ADD COLUMN search_vector tsvector
                    GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
            """)

            cur.execute("""
                CREATE INDEX idx_chunks_search_vector ON chunks USING gin(search_vector)
            """)

        conn.commit()
        print(f"Migration 001 applied successfully to '{db.name}'.")

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


if __name__ == "__main__":
    run()
