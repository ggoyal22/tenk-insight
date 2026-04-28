from datetime import date, datetime, timezone
from uuid import uuid4

import pytest

from db.factory import create_chunks_repo, create_filings_repo, create_parent_chunks_repo
from db.models import ChunkRecord, FilingRecord, ParentChunkRecord
from db.vector.pgvector import PgvectorStore


def _filing() -> FilingRecord:
    return FilingRecord(
        id=uuid4(),
        ticker="NVDA",
        company_name="NVIDIA Corporation",
        cik="1045810",
        accession_number="0001045810-24-000001",
        form_type="10-K",
        filing_date=date(2024, 2, 21),
        source_url="https://www.sec.gov/Archives/edgar/data/1045810/000104581024000001",
        downloaded_at=datetime.now(timezone.utc),
    )


def _parent_chunk(filing_id, chunk_index: int = 0) -> ParentChunkRecord:
    return ParentChunkRecord(
        id=uuid4(),
        filing_id=filing_id,
        chunk_index=chunk_index,
        section="Item 1",
        text="NVIDIA is a technology company.",
        token_count=6,
        content_hash="a" * 63 + str(chunk_index),
        created_at=datetime.now(timezone.utc),
    )


def _chunk(filing_id, parent_chunk_id, chunk_index: int = 0) -> ChunkRecord:
    return ChunkRecord(
        id=uuid4(),
        filing_id=filing_id,
        parent_chunk_id=parent_chunk_id,
        chunk_index=chunk_index,
        section="Item 1",
        chunk_type="narrative",
        text="NVIDIA makes GPUs.",
        token_count=4,
        content_hash="b" * 63 + str(chunk_index),
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def vector_store(db_client):
    return PgvectorStore(
        client=db_client,
        similarity_threshold=0.0,   # low threshold so test embeddings match
        distance_function="cosine",
        embedding_dimension=1024,
        quantization="none",        # no halfvec cast in tests
    )


@pytest.mark.usefixtures("truncate_tables")
class TestPgvectorStore:
    @pytest.fixture
    def chunk_id(self, db_client):
        filing_id = create_filings_repo(db_client).insert(_filing())
        parent_chunk_id = create_parent_chunks_repo(db_client).insert(_parent_chunk(filing_id))
        return create_chunks_repo(db_client).insert(_chunk(filing_id, parent_chunk_id))

    def test_upsert_sets_embedding(self, vector_store, db_client, chunk_id):
        embedding = [0.1] * 1024
        vector_store.upsert(chunk_id, embedding, {"embedding_model": "BAAI/bge-large-en-v1.5"})

        fetched = create_chunks_repo(db_client).get_by_id(chunk_id)
        assert fetched is not None
        assert fetched.embedded_at is not None
        assert fetched.embedding_model == "BAAI/bge-large-en-v1.5"

    def test_search_returns_results(self, vector_store, db_client, chunk_id):
        embedding = [0.1] * 1024
        vector_store.upsert(chunk_id, embedding, {"embedding_model": "BAAI/bge-large-en-v1.5"})

        results = vector_store.search(query_vector=embedding, top_k=5)
        assert len(results) >= 1
        assert results[0].chunk_id == chunk_id
        assert 0.0 <= results[0].score <= 1.0

    def test_search_respects_top_k(self, vector_store, db_client):
        filing_id = create_filings_repo(db_client).insert(_filing())
        parent_chunk_id = create_parent_chunks_repo(db_client).insert(_parent_chunk(filing_id))
        chunks_repo = create_chunks_repo(db_client)
        embedding = [0.1] * 1024
        for i in range(5):
            cid = chunks_repo.insert(_chunk(filing_id, parent_chunk_id, chunk_index=i))
            vector_store.upsert(cid, embedding, {"embedding_model": "BAAI/bge-large-en-v1.5"})

        results = vector_store.search(query_vector=embedding, top_k=3)
        assert len(results) <= 3

    def test_delete_embedding_clears_embedding(self, vector_store, db_client, chunk_id):
        embedding = [0.1] * 1024
        vector_store.upsert(chunk_id, embedding, {"embedding_model": "BAAI/bge-large-en-v1.5"})
        assert vector_store.delete_embedding(chunk_id) is True

        fetched = create_chunks_repo(db_client).get_by_id(chunk_id)
        assert fetched is not None
        assert fetched.embedding is None
        assert fetched.embedded_at is None

    def test_delete_embedding_returns_false_for_missing(self, vector_store):
        assert vector_store.delete_embedding(uuid4()) is False
