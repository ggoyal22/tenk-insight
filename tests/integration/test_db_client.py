import pytest

from tests.conftest import truncate_tables


@pytest.mark.usefixtures("truncate_tables")
class TestPostgresClient:
    def test_health_check_returns_true(self, db_client):
        assert db_client.health_check() is True

    def test_get_and_release_connection(self, db_client):
        conn = db_client.get_connection()
        assert conn is not None
        db_client.release_connection(conn)

    def test_connection_context_manager_commits(self, db_client):
        with db_client.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                result = cur.fetchone()
        assert result == (1,)

    def test_connection_context_manager_rolls_back_on_exception(self, db_client):
        with pytest.raises(Exception):
            with db_client.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                raise RuntimeError("forced error")

        # pool connection still usable after rollback
        assert db_client.health_check() is True
