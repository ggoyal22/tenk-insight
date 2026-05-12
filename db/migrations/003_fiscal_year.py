"""
Migration 003 — Add fiscal_year integer column to filings.

Adds a fiscal_year INT column that stores the fiscal year label (e.g. 2024)
as declared by the company in their SEC filing. This is distinct from
fiscal_year_end DATE, which is the exact period end date.

Backfills existing rows using EXTRACT(YEAR FROM fiscal_year_end) as a
best-effort approximation. This is accurate for most companies but may be
off by one for companies whose fiscal year ends in January or February and
whose label differs from the calendar year of the end date. Such rows will
be corrected on re-ingest.

Safe to run on a live database. Idempotent.

Usage:
    python db/migrations/003_fiscal_year.py
"""

import sys
from pathlib import Path

import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.loader import load_config  # noqa: E402


def _already_applied(conn) -> bool:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'filings' AND column_name = 'fiscal_year'
        """)
        return cur.fetchone() is not None


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
            print("Migration 003 already applied — skipping.")
            return

        with conn.cursor() as cur:
            cur.execute("ALTER TABLE filings ADD COLUMN fiscal_year INT")

            cur.execute("""
                UPDATE filings
                SET fiscal_year = EXTRACT(YEAR FROM fiscal_year_end)::int
                WHERE fiscal_year_end IS NOT NULL
            """)

            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_filings_fiscal_year ON filings (fiscal_year)"
            )

        conn.commit()
        print(f"Migration 003 applied successfully to '{db.name}'.")

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


if __name__ == "__main__":
    run()
