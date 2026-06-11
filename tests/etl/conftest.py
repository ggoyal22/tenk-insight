from datetime import date, datetime, timezone
from uuid import uuid4

import pytest

from config.loader import ChunkingConfig
from db.models import FilingRecord
from etl.chunker.recursive import RecursiveChunker


@pytest.fixture
def filing() -> FilingRecord:
    return FilingRecord(
        id=uuid4(),
        ticker="NVDA",
        company_name="NVIDIA Corporation",
        cik="1045810",
        accession_number="0001045810-24-000001",
        form_type="10-K",
        filing_date=date(2024, 2, 21),
        fiscal_year_end=date(2024, 1, 28),
        source_url="https://www.sec.gov/test",
        downloaded_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def chunking_config() -> ChunkingConfig:
    return ChunkingConfig(
        child_chunk_size=256,
        child_chunk_overlap=32,
        parent_chunk_size=1024,
        parent_chunk_overlap=64,
    )


@pytest.fixture(scope="module")
def chunker() -> RecursiveChunker:
    # Module-scoped so tiktoken's cl100k_base encoding is loaded once per module.
    config = ChunkingConfig(
        child_chunk_size=256,
        child_chunk_overlap=32,
        parent_chunk_size=1024,
        parent_chunk_overlap=64,
    )
    return RecursiveChunker(config)
