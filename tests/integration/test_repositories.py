from datetime import date, datetime, timezone
from uuid import UUID, uuid4

import pytest

from db.factory import create_chunks_repo, create_filings_repo, create_parent_chunks_repo
from db.models import ChunkRecord, FilingRecord, ParentChunkRecord


def _filing(accession_number: str = "0001045810-24-000001") -> FilingRecord:
    return FilingRecord(
        id=uuid4(),
        ticker="NVDA",
        company_name="NVIDIA Corporation",
        cik="1045810",
        accession_number=accession_number,
        form_type="10-K",
        filing_date=date(2024, 2, 21),
        source_url="https://www.sec.gov/Archives/edgar/data/1045810/000104581024000001",
        downloaded_at=datetime.now(timezone.utc),
    )


def _parent_chunk(filing_id: UUID, chunk_index: int = 0) -> ParentChunkRecord:
    return ParentChunkRecord(
        id=uuid4(),
        filing_id=filing_id,
        chunk_index=chunk_index,
        section="Item 1",
        text="NVIDIA is a technology company.",
        token_count=6,
        content_hash="a" * 64,
        created_at=datetime.now(timezone.utc),
    )


def _chunk(filing_id: UUID, parent_chunk_id: UUID, chunk_index: int = 0) -> ChunkRecord:
    return ChunkRecord(
        id=uuid4(),
        filing_id=filing_id,
        parent_chunk_id=parent_chunk_id,
        chunk_index=chunk_index,
        section="Item 1",
        chunk_type="narrative",
        text="NVIDIA makes GPUs.",
        token_count=4,
        content_hash="b" * 64,
        created_at=datetime.now(timezone.utc),
    )


# ── FilingsRepository ─────────────────────────────────────────────────────────

@pytest.mark.usefixtures("truncate_tables")
class TestFilingsRepository:
    @pytest.fixture
    def repo(self, db_client):
        return create_filings_repo(db_client)

    def test_insert_and_get_by_id(self, repo):
        record = _filing()
        id_ = repo.insert(record)
        fetched = repo.get_by_id(id_)
        assert fetched is not None
        assert fetched.ticker == "NVDA"
        assert fetched.accession_number == record.accession_number

    def test_get_by_id_returns_none_for_missing(self, repo):
        assert repo.get_by_id(uuid4()) is None

    def test_exists(self, repo):
        record = _filing()
        id_ = repo.insert(record)
        assert repo.exists(id_) is True
        assert repo.exists(uuid4()) is False

    def test_get_by_accession_number(self, repo):
        record = _filing()
        repo.insert(record)
        fetched = repo.get_by_accession_number(record.accession_number)
        assert fetched is not None
        assert fetched.ticker == "NVDA"

    def test_get_by_accession_number_returns_none_for_missing(self, repo):
        assert repo.get_by_accession_number("0000000000-00-000000") is None

    def test_exists_by_accession_number(self, repo):
        record = _filing()
        repo.insert(record)
        assert repo.exists_by_accession_number(record.accession_number) is True
        assert repo.exists_by_accession_number("0000000000-00-000000") is False

    def test_get_by_ticker(self, repo):
        repo.insert(_filing("0001045810-24-000001"))
        repo.insert(_filing("0001045810-23-000001"))
        results = repo.get_by_ticker("NVDA")
        assert len(results) == 2
        assert all(r.ticker == "NVDA" for r in results)

    def test_update(self, repo):
        id_ = repo.insert(_filing())
        updated = repo.update(id_, {"sic_code": "3674"})
        assert updated is not None
        assert updated.sic_code == "3674"

    def test_update_rejects_non_updatable_columns(self, repo):
        id_ = repo.insert(_filing())
        with pytest.raises(ValueError, match="non-updatable"):
            repo.update(id_, {"ticker": "AMD"})

    def test_delete(self, repo):
        id_ = repo.insert(_filing())
        assert repo.delete(id_) is True
        assert repo.exists(id_) is False

    def test_delete_returns_false_for_missing(self, repo):
        assert repo.delete(uuid4()) is False


# ── ParentChunksRepository ────────────────────────────────────────────────────

@pytest.mark.usefixtures("truncate_tables")
class TestParentChunksRepository:
    @pytest.fixture
    def filing_id(self, db_client):
        return create_filings_repo(db_client).insert(_filing())

    @pytest.fixture
    def repo(self, db_client):
        return create_parent_chunks_repo(db_client)

    def test_insert_and_get_by_id(self, repo, filing_id):
        record = _parent_chunk(filing_id)
        id_ = repo.insert(record)
        fetched = repo.get_by_id(id_)
        assert fetched is not None
        assert fetched.section == "Item 1"

    def test_get_by_filing_id(self, repo, filing_id):
        repo.insert(_parent_chunk(filing_id, chunk_index=0))
        repo.insert(_parent_chunk(filing_id, chunk_index=1))
        results = repo.get_by_filing_id(filing_id)
        assert len(results) == 2
        assert results[0].chunk_index == 0
        assert results[1].chunk_index == 1

    def test_exists_by_content_hash(self, repo, filing_id):
        record = _parent_chunk(filing_id)
        repo.insert(record)
        assert repo.exists_by_content_hash(record.content_hash) is True
        assert repo.exists_by_content_hash("z" * 64) is False

    def test_insert_many(self, repo, filing_id):
        records = [_parent_chunk(filing_id, chunk_index=i) for i in range(3)]
        repo.insert_many(records)
        results = repo.get_by_filing_id(filing_id)
        assert len(results) == 3


# ── ChunksRepository ──────────────────────────────────────────────────────────

@pytest.mark.usefixtures("truncate_tables")
class TestChunksRepository:
    @pytest.fixture
    def filing_id(self, db_client):
        return create_filings_repo(db_client).insert(_filing())

    @pytest.fixture
    def parent_chunk_id(self, db_client, filing_id):
        return create_parent_chunks_repo(db_client).insert(_parent_chunk(filing_id))

    @pytest.fixture
    def repo(self, db_client):
        return create_chunks_repo(db_client)

    def test_insert_and_get_by_id(self, repo, filing_id, parent_chunk_id):
        record = _chunk(filing_id, parent_chunk_id)
        id_ = repo.insert(record)
        fetched = repo.get_by_id(id_)
        assert fetched is not None
        assert fetched.chunk_type == "narrative"

    def test_get_by_filing_id(self, repo, filing_id, parent_chunk_id):
        repo.insert(_chunk(filing_id, parent_chunk_id, chunk_index=0))
        repo.insert(_chunk(filing_id, parent_chunk_id, chunk_index=1))
        results = repo.get_by_filing_id(filing_id)
        assert len(results) == 2

    def test_get_unembedded(self, repo, filing_id, parent_chunk_id):
        repo.insert(_chunk(filing_id, parent_chunk_id, chunk_index=0))
        repo.insert(_chunk(filing_id, parent_chunk_id, chunk_index=1))
        results = repo.get_unembedded()
        assert len(results) == 2
        assert all(r.embedded_at is None for r in results)

    def test_exists_by_content_hash(self, repo, filing_id, parent_chunk_id):
        record = _chunk(filing_id, parent_chunk_id)
        repo.insert(record)
        assert repo.exists_by_content_hash(record.content_hash) is True
        assert repo.exists_by_content_hash("z" * 64) is False

    def test_insert_many(self, repo, filing_id, parent_chunk_id):
        records = [_chunk(filing_id, parent_chunk_id, chunk_index=i) for i in range(3)]
        repo.insert_many(records)
        results = repo.get_by_filing_id(filing_id)
        assert len(results) == 3

    def test_delete(self, repo, filing_id, parent_chunk_id):
        id_ = repo.insert(_chunk(filing_id, parent_chunk_id))
        assert repo.delete(id_) is True
        assert repo.exists(id_) is False
