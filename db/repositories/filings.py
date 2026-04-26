from abc import abstractmethod

from db.models import FilingRecord
from db.repositories.base import RelationalRepository


class FilingsRepo(RelationalRepository[FilingRecord]):
    @abstractmethod
    def get_by_accession_number(self, accession_number: str) -> FilingRecord | None: ...

    @abstractmethod
    def get_by_ticker(self, ticker: str) -> list[FilingRecord]: ...

    @abstractmethod
    def exists_by_accession_number(self, accession_number: str) -> bool: ...
