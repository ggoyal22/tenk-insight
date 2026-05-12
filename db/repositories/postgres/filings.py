from uuid import UUID

from db.client.base import DatabaseClient
from db.models import FilingRecord
from db.repositories.filings import FilingsRepo
from db.repositories.postgres.base import PostgresRepository


class PostgresFilingsRepository(FilingsRepo, PostgresRepository[FilingRecord]):
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
        return {"fiscal_year_end", "fiscal_year", "sic_code"}

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

    def list_indexed_summary(self) -> list[tuple[str, str, list[int]]]:
        sql = """
            SELECT
                ticker,
                company_name,
                array_agg(
                    DISTINCT fiscal_year
                    ORDER BY fiscal_year DESC
                )
            FROM filings
            GROUP BY ticker, company_name
            ORDER BY ticker
        """
        rows = self._execute_returning(sql, ())
        return [(row[0], row[1], list(row[2])) for row in rows]

    def exists_by_accession_number(self, accession_number: str) -> bool:
        sql = f"SELECT EXISTS(SELECT 1 FROM {self._table} WHERE accession_number = %s)"
        rows = self._execute_returning(sql, (accession_number,))
        return rows[0][0]

    def list_ids(self, filters: dict | None = None) -> list[UUID]:
        sql = f"SELECT id FROM {self._table}"
        params: tuple = ()

        if filters:
            invalid = filters.keys() - set(self._columns)
            if invalid:
                raise ValueError(
                    f"Invalid filter columns for {self._table}: {invalid}. "
                    f"Allowed: {self._columns}"
                )
            where_clause = " AND ".join(f"{k} = %s" for k in filters)
            sql += f" WHERE {where_clause}"
            params = tuple(filters.values())

        rows = self._execute_returning(sql, params)
        return [UUID(str(row[0])) for row in rows]
