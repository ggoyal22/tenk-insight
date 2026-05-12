import logging

from db.client.base import DatabaseClient
from db.repositories.feedback import FeedbackRepo

logger = logging.getLogger(__name__)


class PostgresFeedbackRepo(FeedbackRepo):
    def __init__(self, client: DatabaseClient) -> None:
        self._client = client

    def create_table(self) -> None:
        sql = """
            CREATE TABLE IF NOT EXISTS query_feedback (
                id         UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
                query      TEXT        NOT NULL,
                answer     TEXT        NOT NULL,
                rating     BOOLEAN     NOT NULL,
                comment    TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """
        with self._client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()

    def insert_feedback(
        self,
        query: str,
        answer: str,
        rating: bool,
        comment: str | None,
    ) -> None:
        sql = """
            INSERT INTO query_feedback (query, answer, rating, comment)
            VALUES (%s, %s, %s, %s)
        """
        with self._client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (query, answer, rating, comment))
            conn.commit()
        logger.debug("Feedback stored: rating=%r query=%r", rating, query[:60])
