"""Integration tests for retrieval DB operations — requires a live test database."""

from datetime import date, datetime, timezone
from uuid import UUID, uuid4

import pytest

from config.loader import RetrievalConfig
from db.factory import create_chunks_repo, create_filings_repo, create_parent_chunks_repo
from db.models import ChunkRecord, FilingRecord, ParentChunkRecord
from db.vector.pgvector import PgvectorStore
from retrieval.fusion.rrf import RRFFusion
from retrieval.keyword.postgres_fts import PostgresFTSRetriever
from retrieval.retriever import Retriever
from retrieval.types import MetadataFilter
from retrieval.vector.pgvector import PgvectorRetriever
from tests.conftest import VALID_RETRIEVAL


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
        section="Item 1A",
        text="NVIDIA faces supply chain risks in semiconductor manufacturing.",
        token_count=10,
        content_hash="a" * 63 + str(chunk_index),
        created_at=datetime.now(timezone.utc),
    )


def _chunk(
    filing_id: UUID,
    parent_chunk_id: UUID,
    chunk_index: int = 0,
    text: str = "NVIDIA GPU supply chain risk.",
) -> ChunkRecord:
    return ChunkRecord(
        id=uuid4(),
        filing_id=filing_id,
        parent_chunk_id=parent_chunk_id,
        chunk_index=chunk_index,
        section="Item 1A",
        chunk_type="narrative",
        text=text,
        token_count=6,
        content_hash="b" * 63 + str(chunk_index),
        created_at=datetime.now(timezone.utc),
    )


# ── get_by_ids ────────────────────────────────────────────────────────────────

@pytest.mark.usefixtures("truncate_tables")
class TestGetByIds:
    def test_chunks_get_by_ids(self, db_client):
        filings_repo = create_filings_repo(db_client)
        parents_repo = create_parent_chunks_repo(db_client)
        chunks_repo = create_chunks_repo(db_client)

        filing_id = filings_repo.insert(_filing())
        parent_id = parents_repo.insert(_parent_chunk(filing_id))
        id1 = chunks_repo.insert(_chunk(filing_id, parent_id, chunk_index=0))
        id2 = chunks_repo.insert(_chunk(filing_id, parent_id, chunk_index=1))
        chunks_repo.insert(_chunk(filing_id, parent_id, chunk_index=2))  # not requested

        results = chunks_repo.get_by_ids([id1, id2])
        assert len(results) == 2
        assert {r.id for r in results} == {id1, id2}

    def test_chunks_get_by_ids_empty(self, db_client):
        assert create_chunks_repo(db_client).get_by_ids([]) == []

    def test_chunks_get_by_ids_missing(self, db_client):
        results = create_chunks_repo(db_client).get_by_ids([uuid4()])
        assert results == []

    def test_filings_get_by_ids(self, db_client):
        repo = create_filings_repo(db_client)
        id1 = repo.insert(_filing("0001045810-24-000001"))
        id2 = repo.insert(_filing("0001045810-23-000001"))
        repo.insert(_filing("0001045810-22-000001"))  # not requested

        results = repo.get_by_ids([id1, id2])
        assert len(results) == 2
        assert {r.id for r in results} == {id1, id2}

    def test_parent_chunks_get_by_ids(self, db_client):
        filings_repo = create_filings_repo(db_client)
        parents_repo = create_parent_chunks_repo(db_client)

        filing_id = filings_repo.insert(_filing())
        id1 = parents_repo.insert(_parent_chunk(filing_id, chunk_index=0))
        id2 = parents_repo.insert(_parent_chunk(filing_id, chunk_index=1))

        results = parents_repo.get_by_ids([id1, id2])
        assert len(results) == 2
        assert {r.id for r in results} == {id1, id2}


# ── keyword_search ────────────────────────────────────────────────────────────

@pytest.mark.usefixtures("truncate_tables")
class TestKeywordSearch:
    @pytest.fixture
    def setup(self, db_client):
        filings_repo = create_filings_repo(db_client)
        parents_repo = create_parent_chunks_repo(db_client)
        chunks_repo = create_chunks_repo(db_client)

        filing_id = filings_repo.insert(_filing())
        parent_id = parents_repo.insert(_parent_chunk(filing_id))
        chunks_repo.insert(_chunk(filing_id, parent_id, chunk_index=0, text="NVIDIA GPU supply chain risk."))
        chunks_repo.insert(_chunk(filing_id, parent_id, chunk_index=1, text="Revenue from data center grew significantly."))
        chunks_repo.insert(_chunk(filing_id, parent_id, chunk_index=2, text="Automotive segment faces headwinds."))
        return filing_id, chunks_repo

    def test_returns_matching_chunks(self, setup):
        filing_id, chunks_repo = setup
        results = chunks_repo.keyword_search("GPU supply chain", top_k=10)
        assert len(results) >= 1
        # first result should be the GPU supply chain chunk
        assert isinstance(results[0][0], UUID)
        assert isinstance(results[0][1], float)

    def test_no_match_returns_empty(self, setup):
        _, chunks_repo = setup
        results = chunks_repo.keyword_search("quantum computing blockchain", top_k=10)
        assert results == []

    def test_filing_id_filter(self, db_client, setup):
        filing_id, chunks_repo = setup
        filings_repo = create_filings_repo(db_client)

        # Insert a second filing with identical text
        other_filing_id = filings_repo.insert(_filing("0001045810-23-000001"))
        other_parent_id = create_parent_chunks_repo(db_client).insert(_parent_chunk(other_filing_id))
        chunks_repo.insert(_chunk(other_filing_id, other_parent_id, chunk_index=0, text="NVIDIA GPU supply chain risk."))

        results_filtered = chunks_repo.keyword_search(
            "GPU supply chain", top_k=10, filing_ids=[filing_id]
        )
        result_ids = [r[0] for r in results_filtered]
        chunks = chunks_repo.get_by_ids(result_ids)
        assert all(c.filing_id == filing_id for c in chunks)

    def test_respects_top_k(self, setup):
        _, chunks_repo = setup
        results = chunks_repo.keyword_search("NVIDIA", top_k=1)
        assert len(results) <= 1

    def test_scores_descending(self, setup):
        _, chunks_repo = setup
        results = chunks_repo.keyword_search("GPU supply chain revenue", top_k=10)
        if len(results) > 1:
            scores = [s for _, s in results]
            assert scores == sorted(scores, reverse=True)

    def test_exclude_parent_ids(self, db_client):
        filings_repo = create_filings_repo(db_client)
        parents_repo = create_parent_chunks_repo(db_client)
        chunks_repo = create_chunks_repo(db_client)

        filing_id = filings_repo.insert(_filing())
        parent_a = parents_repo.insert(_parent_chunk(filing_id, chunk_index=0))
        parent_b = parents_repo.insert(_parent_chunk(filing_id, chunk_index=1))
        chunks_repo.insert(_chunk(filing_id, parent_a, chunk_index=0, text="NVIDIA GPU supply chain risk."))
        chunks_repo.insert(_chunk(filing_id, parent_b, chunk_index=1, text="NVIDIA GPU supply chain shortage."))

        results = chunks_repo.keyword_search("GPU supply chain", top_k=10, exclude_parent_ids=[parent_a])
        chunks = chunks_repo.get_by_ids([r[0] for r in results])
        assert chunks
        assert all(c.parent_chunk_id != parent_a for c in chunks)
        assert any(c.parent_chunk_id == parent_b for c in chunks)


# ── Retriever end-to-end ──────────────────────────────────────────────────────

def _make_retriever(db_client) -> tuple[Retriever, object, object, object]:
    """Wire up a full Retriever stack against the test DB. Returns (retriever, repos...)."""
    config = RetrievalConfig(**VALID_RETRIEVAL)
    filings_repo = create_filings_repo(db_client)
    parents_repo = create_parent_chunks_repo(db_client)
    chunks_repo = create_chunks_repo(db_client)
    vector_store = PgvectorStore(
        client=db_client,
        similarity_threshold=0.0,
        distance_function="cosine",
        embedding_dimension=1024,
        quantization="none",
    )
    retriever = Retriever(
        config=config,
        fusion=RRFFusion(k=60),
        chunks_repo=chunks_repo,
        parent_chunks_repo=parents_repo,
        filings_repo=filings_repo,
        vector_retriever=PgvectorRetriever(vector_store, config.vector_search.oversample_k),
        keyword_retriever=PostgresFTSRetriever(chunks_repo, "web"),
    )
    return retriever, filings_repo, parents_repo, chunks_repo, vector_store


@pytest.mark.usefixtures("truncate_tables")
class TestRetrieverEndToEnd:
    def test_retrieve_returns_enriched_results(self, db_client):
        retriever, filings_repo, parents_repo, chunks_repo, vector_store = _make_retriever(db_client)

        filing_id = filings_repo.insert(_filing())
        parent_id = parents_repo.insert(_parent_chunk(filing_id))
        chunk_id = chunks_repo.insert(_chunk(filing_id, parent_id, text="NVIDIA GPU supply chain risk."))
        embedding = [0.1] * 1024
        vector_store.upsert(chunk_id, embedding, {"embedding_model": "test-model"})

        results = retriever.retrieve(keyword_query="GPU supply chain", semantic_embedding=embedding)

        assert len(results) >= 1
        r = results[0]
        assert r.chunk.id == chunk_id
        assert r.parent_chunk.id == parent_id
        assert r.filing.id == filing_id
        assert 0.0 <= r.score <= 1.0

    def test_retrieve_with_ticker_filter(self, db_client):
        retriever, filings_repo, parents_repo, chunks_repo, vector_store = _make_retriever(db_client)

        # NVDA filing with embedded chunk
        nvda_filing_id = filings_repo.insert(_filing("0001045810-24-000001"))
        nvda_parent_id = parents_repo.insert(_parent_chunk(nvda_filing_id))
        nvda_chunk_id = chunks_repo.insert(_chunk(nvda_filing_id, nvda_parent_id, text="NVIDIA GPU supply chain."))
        embedding = [0.1] * 1024
        vector_store.upsert(nvda_chunk_id, embedding, {"embedding_model": "test-model"})

        # AMD filing with embedded chunk — same text so keyword search would normally return it
        amd_filing = FilingRecord(
            id=uuid4(), ticker="AMD", company_name="Advanced Micro Devices",
            cik="2488", accession_number="0000002488-24-000001", form_type="10-K",
            filing_date=date(2024, 2, 1),
            source_url="https://www.sec.gov/Archives/edgar/data/2488/000000248824000001",
            downloaded_at=datetime.now(timezone.utc),
        )
        amd_filing_id = filings_repo.insert(amd_filing)
        amd_parent_id = parents_repo.insert(_parent_chunk(amd_filing_id))
        amd_chunk_id = chunks_repo.insert(_chunk(amd_filing_id, amd_parent_id, text="AMD GPU supply chain."))
        vector_store.upsert(amd_chunk_id, embedding, {"embedding_model": "test-model"})

        results = retriever.retrieve(
            keyword_query="GPU supply chain",
            semantic_embedding=embedding,
            filters=MetadataFilter(ticker="nvda"),  # lowercase — tests normalization
        )

        assert len(results) >= 1
        assert all(r.filing.ticker == "NVDA" for r in results)
        assert all(r.chunk.id != amd_chunk_id for r in results)

    def test_retrieve_excludes_seen_parents(self, db_client):
        retriever, filings_repo, parents_repo, chunks_repo, vector_store = _make_retriever(db_client)

        filing_id = filings_repo.insert(_filing())
        parent_a = parents_repo.insert(_parent_chunk(filing_id, chunk_index=0))
        parent_b = parents_repo.insert(_parent_chunk(filing_id, chunk_index=1))
        chunk_a = chunks_repo.insert(_chunk(filing_id, parent_a, chunk_index=0, text="NVIDIA GPU supply chain risk."))
        chunk_b = chunks_repo.insert(_chunk(filing_id, parent_b, chunk_index=1, text="NVIDIA GPU supply chain shortage."))
        embedding = [0.1] * 1024
        vector_store.upsert(chunk_a, embedding, {"embedding_model": "test-model"})
        vector_store.upsert(chunk_b, embedding, {"embedding_model": "test-model"})

        # Baseline — both parents are retrievable.
        baseline = retriever.retrieve(keyword_query="GPU supply chain", semantic_embedding=embedding)
        assert {parent_a, parent_b} <= {r.parent_chunk.id for r in baseline}

        # Excluding parent_a must drop it from both the vector and keyword legs.
        results = retriever.retrieve(
            keyword_query="GPU supply chain",
            semantic_embedding=embedding,
            exclude_parent_ids=[parent_a],
        )
        assert results
        assert all(r.parent_chunk.id != parent_a for r in results)
        assert any(r.parent_chunk.id == parent_b for r in results)
        assert chunk_a not in {r.chunk.id for r in results}

    def test_retrieve_on_empty_db_returns_empty(self, db_client):
        retriever, *_ = _make_retriever(db_client)
        results = retriever.retrieve(keyword_query="GPU supply chain", semantic_embedding=[0.1] * 1024)
        assert results == []
