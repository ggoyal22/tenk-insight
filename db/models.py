from dataclasses import dataclass
from datetime import date, datetime
from uuid import UUID


@dataclass
class FilingRecord:
    id: UUID
    ticker: str
    company_name: str
    cik: str
    accession_number: str
    form_type: str
    filing_date: date
    source_url: str
    downloaded_at: datetime
    fiscal_year_end: date | None = None
    sic_code: str | None = None
    updated_at: datetime | None = None


@dataclass
class ParentChunkRecord:
    id: UUID
    filing_id: UUID
    chunk_index: int
    section: str
    text: str
    token_count: int
    content_hash: str
    created_at: datetime
    updated_at: datetime | None = None


@dataclass
class ChunkRecord:
    id: UUID
    filing_id: UUID
    chunk_index: int
    section: str
    chunk_type: str
    text: str
    token_count: int
    content_hash: str
    created_at: datetime
    parent_chunk_id: UUID | None = None
    page_number: int | None = None
    embedding: list[float] | None = None
    embedding_model: str | None = None
    embedded_at: datetime | None = None
    updated_at: datetime | None = None
