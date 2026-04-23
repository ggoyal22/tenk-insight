import logging

import psycopg2
from psycopg2 import pool

from config.loader import DatabaseConfig
from db.client.base import DatabaseClient, DatabaseConnection

logger = logging.getLogger(__name__)


class PostgresClient(DatabaseClient):
    def __init__(self, config: DatabaseConfig) -> None:
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
        return self._pool.getconn()

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
