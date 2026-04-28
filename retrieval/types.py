from dataclasses import dataclass
from datetime import date
from uuid import UUID

from db.models import ChunkRecord, FilingRecord, ParentChunkRecord


@dataclass
class MetadataFilter:
    ticker: str | None = None
    form_type: str | None = None
    fiscal_year_end: date | None = None
    section: str | None = None


@dataclass
class RetrievalResult:
    score: float
    chunk: ChunkRecord
    parent_chunk: ParentChunkRecord
    filing: FilingRecord
