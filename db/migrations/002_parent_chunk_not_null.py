"""
Migration 002 — Enforce NOT NULL on chunks.parent_chunk_id and switch to ON DELETE CASCADE.

Applies two changes to an existing database:
  1. Add NOT NULL constraint to chunks.parent_chunk_id.
  2. Replace the ON DELETE SET NULL foreign key with ON DELETE CASCADE.

Safe to run on a live database. Idempotent — checks whether already applied before acting.
Fails fast with a clear message if any chunks have a NULL parent_chunk_id (indicates
incomplete ingestion that must be resolved before the constraint can be enforced).

Usage:
    python db/migrations/002_parent_chunk_not_null.py
"""

import sys
from pathlib import Path

import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.loader import load_config  # noqa: E402


def _already_applied(conn) -> bool:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT is_nullable
            FROM information_schema.columns
            WHERE table_name = 'chunks' AND column_name = 'parent_chunk_id'
        """)
        row = cur.fetchone()
        return row is not None and row[0] == "NO"


def _check_no_nulls(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM chunks WHERE parent_chunk_id IS NULL")
        count = cur.fetchone()[0]
    if count > 0:
        raise RuntimeError(
            f"Migration 002 cannot proceed: {count} chunk(s) have NULL parent_chunk_id. "
            "Re-run the ingestion pipeline to fix orphaned chunks before applying this migration."
        )


def _find_fk_constraint(conn) -> str | None:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT conname FROM pg_constraint
            WHERE conrelid = 'chunks'::regclass
              AND contype = 'f'
              AND confrelid = 'parent_chunks'::regclass
        """)
        row = cur.fetchone()
    return row[0] if row else None


def run() -> None:
    config = load_config()
    db = config.database

    conn = psycopg2.connect(
        host=db.host,
        port=db.port,
        dbname=db.name,
        user=db.user,
        password=db.password.get_secret_value(),
    )
    try:
        if _already_applied(conn):
            print("Migration 002 already applied — skipping.")
            return

        _check_no_nulls(conn)

        fk_name = _find_fk_constraint(conn)

        with conn.cursor() as cur:
            if fk_name:
                cur.execute(f"ALTER TABLE chunks DROP CONSTRAINT {fk_name}")

            cur.execute("ALTER TABLE chunks ALTER COLUMN parent_chunk_id SET NOT NULL")

            cur.execute("""
                ALTER TABLE chunks
                    ADD CONSTRAINT chunks_parent_chunk_id_fkey
                    FOREIGN KEY (parent_chunk_id) REFERENCES parent_chunks (id) ON DELETE CASCADE
            """)

        conn.commit()
        print(f"Migration 002 applied successfully to '{db.name}'.")

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


if __name__ == "__main__":
    run()
