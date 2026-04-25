from abc import ABC, abstractmethod
from pathlib import Path

from db.models import FilingRecord


class FilingNotFoundError(Exception):
    """Raised when no filing matches the requested ticker, form type, and year."""


class Downloader(ABC):
    @abstractmethod
    def fetch(self, ticker: str, form_type: str, year: int) -> tuple[FilingRecord, Path]:
        """Fetch a filing from the source, save raw content to disk, and return
        the filing metadata alongside the path to the saved file.

        Args:
            ticker:    stock ticker, e.g. "NVDA"
            form_type: SEC form type, e.g. "10-K"
            year:      fiscal year, e.g. 2024

        Returns:
            (FilingRecord, Path) — filing metadata and path to raw file on disk

        Raises:
            FilingNotFoundError: if no filing matches the given parameters
        """
        ...
