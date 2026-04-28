"""
Database setup for the SEC EDGAR RAG system.

Reads config and substitutes placeholders in db/schema.template.sql,
then executes the DDL against the configured PostgreSQL database.

Usage:
    python db/setup.py

The schema uses CREATE IF NOT EXISTS throughout. If the schema already
exists, a message is printed and the script exits without making changes.
"""

import sys
from pathlib import Path

import psycopg2

# Insert project root so this script is runnable directly: python db/setup.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.loader import AppConfig, ensure_directories, load_config  # noqa: E402

_SCHEMA_TEMPLATE = Path(__file__).resolve().parent / "schema.template.sql"

# Maps (distance_function, quantization) to the correct pgvector HNSW ops class.
_HNSW_OPS_CLASS: dict[tuple[str, str], str] = {
    ("cosine", "none"):    "vector_cosine_ops",
    ("cosine", "halfvec"): "halfvec_cosine_ops",
    ("cosine", "scalar"):  "int8_cosine_ops",
    ("l2",     "none"):    "vector_l2_ops",
    ("l2",     "halfvec"): "halfvec_l2_ops",
    ("dot",    "none"):    "vector_ip_ops",
    ("dot",    "halfvec"): "halfvec_ip_ops",
}


def _hnsw_index_col(config: AppConfig) -> str:
    """Return the HNSW index column expression for the configured quantization."""
    vi = config.vector_index
    q = config.retrieval.vector_search.quantization
    ops = _HNSW_OPS_CLASS[(vi.distance_function, q)]
    if q == "none":
        return f"embedding {ops}"
    dim = config.embedding.dimension
    cast = "halfvec" if q == "halfvec" else "int8"
    return f"(embedding::{cast}({dim})) {ops}"


def _substitute(template: str, config: AppConfig) -> str:
    """Substitute schema template placeholders with values from config.

    Uses str.replace() rather than str.format() to avoid conflicts with
    SQL regex patterns that contain literal braces (e.g. '^[0-9a-f]{64}$').
    """
    vi = config.vector_index
    return (
        template
        .replace("{embedding_dimension}", str(config.embedding.dimension))
        .replace("{hnsw_index_col}",      _hnsw_index_col(config))
        .replace("{hnsw_m}",              str(vi.hnsw_m))
        .replace("{hnsw_ef_construction}", str(vi.hnsw_ef_construction))
    )


def _schema_exists(conn) -> bool:
    """Return True if the schema has already been created (filings table exists)."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = 'filings'
            )
        """)
        return cur.fetchone()[0]


def run() -> None:
    """Substitute the schema template and execute DDL against the configured database."""
    config = load_config()
    ensure_directories(config)

    sql = _substitute(_SCHEMA_TEMPLATE.read_text(), config)

    db = config.database
    conn = psycopg2.connect(
        host=db.host,
        port=db.port,
        dbname=db.name,
        user=db.user,
        password=db.password.get_secret_value(),
    )
    try:
        if _schema_exists(conn):
            print(f"Schema already exists in '{db.name}' — no changes made.")
            return

        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
        print(f"Schema created successfully in '{db.name}' on {db.host}:{db.port}.")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    run()
