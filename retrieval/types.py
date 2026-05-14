from dataclasses import dataclass
from uuid import UUID

from db.models import ChunkRecord, FilingRecord, ParentChunkRecord


@dataclass
class MetadataFilter:
    ticker: str | None = None
    form_type: str | None = None
    fiscal_year: int | None = None
    section: str | None = None


@dataclass
class RetrievalResult:
    score: float
    vector_score: float | None
    keyword_score: float | None
    reranker_score: float | None
    chunk: ChunkRecord
    parent_chunk: ParentChunkRecord
    filing: FilingRecord
