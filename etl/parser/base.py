from abc import ABC, abstractmethod
from pathlib import Path

from etl.types import ParsedSection


class Parser(ABC):
    def _validate_raw_file(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Raw filing not found: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"Raw filing is empty: {path}")

    @abstractmethod
    def parse(self, raw_path: Path) -> list[ParsedSection]:
        """Parse a raw filing file into a list of typed, ordered sections.

        Args:
            raw_path: path to the raw filing file saved by the downloader

        Returns:
            list of ParsedSection ordered by section_order
        """
        ...
