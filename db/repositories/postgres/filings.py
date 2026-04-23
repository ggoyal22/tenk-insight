from db.client.base import DatabaseClient
from db.models import FilingRecord
from db.repositories.postgres.base import PostgresRepository


class FilingsRepository(PostgresRepository[FilingRecord]):
    def __init__(self, client: DatabaseClient) -> None:
        super().__init__(client)

    @property
    def _auto_columns(self) -> set[str]:
        return {"id", "updated_at"}

    @property
    def _table(self) -> str:
        return "filings"

    @property
    def _model_class(self) -> type:
        return FilingRecord

    @property
    def _updatable_columns(self) -> set[str]:
        return {"fiscal_year_end", "sic_code"}

    # ── Filings-specific queries ──────────────────────────────────────────────

    def get_by_accession_number(self, accession_number: str) -> FilingRecord | None:
        col_clause = ", ".join(self._columns)
        sql = f"SELECT {col_clause} FROM {self._table} WHERE accession_number = %s"
        rows = self._execute_returning(sql, (accession_number,))
        if not rows:
            return None
        return self._row_to_model(rows[0], self._columns)

    def get_by_ticker(self, ticker: str) -> list[FilingRecord]:
        col_clause = ", ".join(self._columns)
        sql = f"SELECT {col_clause} FROM {self._table} WHERE ticker = %s ORDER BY filing_date DESC"
        rows = self._execute_returning(sql, (ticker.upper(),))
        return [self._row_to_model(row, self._columns) for row in rows]

    def exists_by_accession_number(self, accession_number: str) -> bool:
        sql = f"SELECT EXISTS(SELECT 1 FROM {self._table} WHERE accession_number = %s)"
        rows = self._execute_returning(sql, (accession_number,))
        return rows[0][0]
