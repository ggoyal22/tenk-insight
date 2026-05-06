import logging
from contextlib import contextmanager
from typing import Generator

import psycopg2
import psycopg2.extras
from psycopg2 import pool

from config.loader import DatabaseConfig
from db.client.base import DatabaseClient, DatabaseConnection, Transaction

logger = logging.getLogger(__name__)


class PostgresTransaction(Transaction):
    """Wraps a live psycopg2 connection. Accessible within the postgres package only."""

    def __init__(self, conn: DatabaseConnection) -> None:
        self.conn = conn


class PostgresClient(DatabaseClient):
    def __init__(self, config: DatabaseConfig) -> None:
        psycopg2.extras.register_uuid()
        self._config = config
        self._pool = pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=config.pool_size,
            host=config.host,
            port=config.port,
            dbname=config.name,
            user=config.user,
            password=config.password.get_secret_value(),
        )
        logger.info(
            "PostgresClient initialised (pool_size=%d, db=%s, host=%s:%d)",
            config.pool_size,
            config.name,
            config.host,
            config.port,
        )

    def get_connection(self) -> DatabaseConnection:
        conn = self._pool.getconn()
        try:
            conn.cursor().execute("SELECT 1")
        except Exception:
            self._pool.putconn(conn, close=True)
            conn = self._pool.getconn()
        return conn

    def release_connection(self, conn: DatabaseConnection) -> None:
        # psycopg2 expects its own connection type; cast is safe — we only ever
        # store psycopg2 connections in this pool.
        self._pool.putconn(conn)  # type: ignore[arg-type]

    def health_check(self) -> bool:
        try:
            with self.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            return True
        except Exception:
            logger.exception("PostgresClient health check failed")
            return False

    def close(self) -> None:
        self._pool.closeall()
        logger.info("PostgresClient pool closed")

    @contextmanager
    def transaction(self) -> Generator[PostgresTransaction, None, None]:
        conn = self.get_connection()
        try:
            yield PostgresTransaction(conn)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self.release_connection(conn)
