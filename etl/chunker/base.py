from abc import ABC, abstractmethod

from db.models import FilingRecord
from etl.types import ChildChunk, ParsedSection, ParentChunk


class Chunker(ABC):
    def _validate_sections(self, sections: list[ParsedSection]) -> None:
        if not sections:
            raise ValueError("Cannot chunk an empty list of sections.")

    @abstractmethod
    def chunk(
        self, sections: list[ParsedSection], filing: FilingRecord
    ) -> tuple[list[ParentChunk], list[ChildChunk]]:
        """Split parsed sections into parent and child chunks.

        Args:
            sections: ordered list of ParsedSection from the parser
            filing:   filing metadata used to build contextual prefix on child text

        Returns:
            (parent_chunks, child_chunks) — both lists ordered by filing_chunk_index
        """
        ...
