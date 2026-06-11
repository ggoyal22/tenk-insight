from contextlib import contextmanager
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, call
from uuid import UUID, uuid4

import psycopg2.errors
import pytest

from db.models import ChunkRecord, FilingRecord
from etl.loader import Loader
from etl.types import ChildChunk, ParentChunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _filing(accession: str = "0001045810-24-000001") -> FilingRecord:
    return FilingRecord(
        id=uuid4(),
        ticker="NVDA",
        company_name="NVIDIA Corporation",
        cik="1045810",
        accession_number=accession,
        form_type="10-K",
        filing_date=date(2024, 2, 21),
        fiscal_year_end=date(2024, 1, 28),
        source_url="https://www.sec.gov/test",
        downloaded_at=datetime.now(timezone.utc),
    )


def _parent(idx: int = 0) -> ParentChunk:
    return ParentChunk(
        section_name="Item 1",
        content_type="narrative",
        text=f"parent text {idx}",
        token_count=3,
        filing_chunk_index=idx,
    )


def _child(idx: int = 0, parent_idx: int = 0, with_embedding: bool = True) -> ChildChunk:
    c = ChildChunk(
        section_name="Item 1",
        content_type="narrative",
        text=f"child text {idx}",
        token_count=3,
        filing_chunk_index=idx,
        parent_chunk_index=parent_idx,
    )
    if with_embedding:
        c.embedding = [0.1] * 4
        c.embedding_model = "test-model"
    return c


def _chunk_record(filing_id: UUID, chunk_index: int) -> ChunkRecord:
    return ChunkRecord(
        id=uuid4(),
        filing_id=filing_id,
        parent_chunk_id=uuid4(),
        chunk_index=chunk_index,
        section="Item 1",
        chunk_type="narrative",
        text=f"child text {chunk_index}",
        token_count=3,
        content_hash="a" * 64,
        created_at=datetime.now(timezone.utc),
    )


def _make_loader() -> tuple[Loader, MagicMock, MagicMock, MagicMock, MagicMock, MagicMock]:
    db_client = MagicMock()
    filings_repo = MagicMock()
    parent_chunks_repo = MagicMock()
    chunks_repo = MagicMock()
    vector_store = MagicMock()
    loader = Loader(db_client, filings_repo, parent_chunks_repo, chunks_repo, vector_store)
    return loader, db_client, filings_repo, parent_chunks_repo, chunks_repo, vector_store


def _install_transaction(db_client: MagicMock) -> MagicMock:
    """Make db_client.transaction() a no-op context manager, yielding a tx mock."""
    tx = MagicMock()

    @contextmanager
    def fake_transaction(*args, **kwargs):
        yield tx

    db_client.transaction.side_effect = fake_transaction
    return tx


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_load_skips_fully_ingested_filing():
    loader, db_client, filings_repo, _, chunks_repo, _ = _make_loader()
    existing_id = uuid4()
    existing = MagicMock()
    existing.id = existing_id

    filings_repo.get_by_accession_number.return_value = existing

    chunk = _chunk_record(existing_id, 0)
    chunk.embedded_at = datetime.now(timezone.utc)  # fully embedded
    chunks_repo.get_by_filing_id.return_value = [chunk]

    result = loader.load(_filing(), [_parent()], [_child()])

    assert result == existing_id
    filings_repo.insert.assert_not_called()


def test_load_reembeds_when_chunks_have_missing_embeddings():
    loader, db_client, filings_repo, _, chunks_repo, vector_store = _make_loader()
    existing_id = uuid4()
    existing = MagicMock()
    existing.id = existing_id

    filings_repo.get_by_accession_number.return_value = existing

    missing_chunk = _chunk_record(existing_id, 0)
    missing_chunk.embedded_at = None  # not yet embedded
    chunks_repo.get_by_filing_id.return_value = [missing_chunk]

    child = _child(idx=0, with_embedding=True)
    result = loader.load(_filing(), [_parent()], [child])

    assert result == existing_id
    vector_store.upsert.assert_called_once_with(
        missing_chunk.id,
        child.embedding,
        {"embedding_model": child.embedding_model},
    )


def test_load_deletes_orphan_filing_and_reingests():
    loader, db_client, filings_repo, parent_chunks_repo, chunks_repo, vector_store = _make_loader()
    tx = _install_transaction(db_client)

    orphan = MagicMock()
    orphan.id = uuid4()
    filing_uuid = uuid4()

    # First call: orphan exists; after delete, second call returns None (new filing)
    filings_repo.get_by_accession_number.side_effect = [orphan, None]
    chunks_repo.get_by_filing_id.return_value = []  # 0 chunks → orphan
    filings_repo.insert.return_value = filing_uuid

    child = _child(0)
    inserted = _chunk_record(filing_uuid, 0)
    chunks_repo.get_by_filing_id.side_effect = [[], [inserted]]

    loader.load(_filing(), [_parent()], [child])

    filings_repo.delete.assert_called_once_with(orphan.id)
    filings_repo.insert.assert_called_once()


def test_load_full_ingestion_inserts_all_records_and_upserts_embeddings():
    loader, db_client, filings_repo, parent_chunks_repo, chunks_repo, vector_store = _make_loader()
    tx = _install_transaction(db_client)

    filing_uuid = uuid4()
    filings_repo.get_by_accession_number.return_value = None
    filings_repo.insert.return_value = filing_uuid

    parents = [_parent(0), _parent(1)]
    children = [_child(0, parent_idx=0), _child(1, parent_idx=1)]
    inserted_records = [_chunk_record(filing_uuid, 0), _chunk_record(filing_uuid, 1)]
    chunks_repo.get_by_filing_id.return_value = inserted_records

    loader.load(_filing(), parents, children)

    filings_repo.insert.assert_called_once()
    assert parent_chunks_repo.insert.call_count == 2
    chunks_repo.insert_many.assert_called_once()
    assert vector_store.upsert.call_count == 2


def test_load_skips_upsert_for_children_without_embedding():
    loader, db_client, filings_repo, parent_chunks_repo, chunks_repo, vector_store = _make_loader()
    _install_transaction(db_client)

    filing_uuid = uuid4()
    filings_repo.get_by_accession_number.return_value = None
    filings_repo.insert.return_value = filing_uuid

    child_no_emb = _child(0, with_embedding=False)
    inserted = _chunk_record(filing_uuid, 0)
    chunks_repo.get_by_filing_id.return_value = [inserted]

    loader.load(_filing(), [_parent()], [child_no_emb])

    vector_store.upsert.assert_not_called()


def test_load_handles_concurrent_unique_violation():
    loader, db_client, filings_repo, _, chunks_repo, _ = _make_loader()
    _install_transaction(db_client)

    existing_id = uuid4()
    existing = MagicMock()
    existing.id = existing_id

    filings_repo.get_by_accession_number.side_effect = [None, existing]
    filings_repo.insert.side_effect = psycopg2.errors.UniqueViolation()

    result = loader.load(_filing(), [_parent()], [_child()])

    assert result == existing_id
