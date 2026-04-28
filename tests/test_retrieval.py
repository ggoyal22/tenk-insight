"""Unit tests for retrieval components — no database required."""

from datetime import date, datetime, timezone
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from config.loader import FusionConfig, KeywordSearchConfig, RetrievalConfig, RerankingConfig, VectorSearchConfig
from db.models import ChunkRecord, FilingRecord, ParentChunkRecord
from retrieval.fusion.rrf import RRFFusion
from retrieval.retriever import Retriever
from retrieval.types import MetadataFilter


def _ids(*n: int) -> list[UUID]:
    """Generate deterministic UUIDs from integers for readable test assertions."""
    return [UUID(int=i) for i in n]


class TestRRFFusion:
    def setup_method(self):
        self.fusion = RRFFusion(k=60)

    def test_single_list_passthrough(self):
        ids = _ids(1, 2, 3)
        results = self.fusion.fuse([(ids[0], 0.9), (ids[1], 0.8), (ids[2], 0.7)])
        result_ids = [cid for cid, _ in results]
        assert result_ids == ids

    def test_two_lists_merged_and_deduplicated(self):
        a, b, c, d = _ids(1, 2, 3, 4)
        # a appears in both lists — should accumulate higher RRF score
        list1 = [(a, 0.9), (b, 0.8)]
        list2 = [(a, 0.7), (c, 0.6), (d, 0.5)]
        results = self.fusion.fuse(list1, list2)
        result_ids = [cid for cid, _ in results]
        # a appears in both lists so should rank first
        assert result_ids[0] == a
        # all unique IDs present
        assert set(result_ids) == {a, b, c, d}

    def test_rrf_score_formula(self):
        k = 60
        fusion = RRFFusion(k=k)
        chunk_id = _ids(1)[0]
        # Single list, rank 0: expected score = 1 / (k + 1)
        results = fusion.fuse([(chunk_id, 0.5)])
        _, score = results[0]
        assert abs(score - 1.0 / (k + 1)) < 1e-9

    def test_empty_list_skipped(self):
        ids = _ids(1, 2)
        results = self.fusion.fuse([(ids[0], 0.9), (ids[1], 0.8)], [])
        assert len(results) == 2

    def test_all_empty_returns_empty(self):
        assert self.fusion.fuse([], []) == []

    def test_scores_descending(self):
        ids = _ids(1, 2, 3)
        results = self.fusion.fuse([(ids[0], 0.9), (ids[1], 0.8), (ids[2], 0.7)])
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)


# ── Retriever mode tests ──────────────────────────────────────────────────────

def _filing_record(id: UUID) -> FilingRecord:
    return FilingRecord(
        id=id, ticker="NVDA", company_name="NVIDIA Corporation",
        cik="1045810", accession_number="0001045810-24-000001", form_type="10-K",
        filing_date=date(2024, 2, 21),
        source_url="https://www.sec.gov/Archives/edgar/data/1045810/000104581024000001",
        downloaded_at=datetime.now(timezone.utc),
    )


def _parent_record(id: UUID, filing_id: UUID) -> ParentChunkRecord:
    return ParentChunkRecord(
        id=id, filing_id=filing_id, chunk_index=0, section="Item 1",
        text="Parent text.", token_count=3, content_hash="a" * 64,
        created_at=datetime.now(timezone.utc),
    )


def _chunk_record(id: UUID, filing_id: UUID, parent_id: UUID) -> ChunkRecord:
    return ChunkRecord(
        id=id, filing_id=filing_id, parent_chunk_id=parent_id,
        chunk_index=0, section="Item 1", chunk_type="narrative",
        text="Chunk text.", token_count=2, content_hash="b" * 64,
        created_at=datetime.now(timezone.utc),
    )


def _make_repos(chunk_id: UUID, parent_id: UUID, filing_id: UUID):
    """Return mocked chunks, parent_chunks, and filings repos pre-populated with one record each."""
    filing = _filing_record(filing_id)
    parent = _parent_record(parent_id, filing_id)
    chunk = _chunk_record(chunk_id, filing_id, parent_id)

    chunks_repo = MagicMock()
    chunks_repo.get_by_ids.return_value = [chunk]

    parents_repo = MagicMock()
    parents_repo.get_by_ids.return_value = [parent]

    filings_repo = MagicMock()
    filings_repo.get_by_ids.return_value = [filing]
    filings_repo.list_ids.return_value = None  # no filtering

    return chunks_repo, parents_repo, filings_repo


class TestRetrieverModes:
    def test_vector_only_skips_keyword_search(self):
        chunk_id, parent_id, filing_id = uuid4(), uuid4(), uuid4()
        chunks_repo, parents_repo, filings_repo = _make_repos(chunk_id, parent_id, filing_id)

        config = RetrievalConfig(
            vector_search=VectorSearchConfig(enabled=True, oversample_k=5, similarity_threshold=0.0),
            keyword_search=KeywordSearchConfig(enabled=False),
            fusion=FusionConfig(top_k=5),
            reranking=RerankingConfig(enabled=False),
            final_top_k=5,
        )

        vector_retriever = MagicMock()
        vector_retriever.search.return_value = [(chunk_id, 0.9)]

        retriever = Retriever(
            config=config, fusion=RRFFusion(),
            chunks_repo=chunks_repo, parent_chunks_repo=parents_repo, filings_repo=filings_repo,
            vector_retriever=vector_retriever,
            keyword_retriever=None,
        )
        results = retriever.retrieve(query="GPU", query_embedding=[0.1] * 8)

        vector_retriever.search.assert_called_once()
        assert len(results) == 1
        assert results[0].chunk.id == chunk_id

    def test_keyword_only_skips_vector_search(self):
        chunk_id, parent_id, filing_id = uuid4(), uuid4(), uuid4()
        chunks_repo, parents_repo, filings_repo = _make_repos(chunk_id, parent_id, filing_id)

        config = RetrievalConfig(
            vector_search=VectorSearchConfig(enabled=False),
            keyword_search=KeywordSearchConfig(enabled=True, top_k=5),
            fusion=FusionConfig(top_k=5),
            reranking=RerankingConfig(enabled=False),
            final_top_k=5,
        )

        keyword_retriever = MagicMock()
        keyword_retriever.search.return_value = [(chunk_id, 0.85)]

        retriever = Retriever(
            config=config, fusion=RRFFusion(),
            chunks_repo=chunks_repo, parent_chunks_repo=parents_repo, filings_repo=filings_repo,
            vector_retriever=None,
            keyword_retriever=keyword_retriever,
        )
        results = retriever.retrieve(query="GPU", query_embedding=[0.1] * 8)

        keyword_retriever.search.assert_called_once()
        assert len(results) == 1
        assert results[0].chunk.id == chunk_id

    def test_results_trimmed_to_final_top_k_when_reranker_disabled(self):
        # Insert 3 chunks, set final_top_k=2 — result must be capped at 2
        ids = [uuid4() for _ in range(3)]
        filing_id, parent_id = uuid4(), uuid4()
        filing = _filing_record(filing_id)
        parent = _parent_record(parent_id, filing_id)
        chunks = [_chunk_record(cid, filing_id, parent_id) for cid in ids]

        chunks_repo = MagicMock()
        chunks_repo.get_by_ids.return_value = chunks
        parents_repo = MagicMock()
        parents_repo.get_by_ids.return_value = [parent]
        filings_repo = MagicMock()
        filings_repo.get_by_ids.return_value = [filing]
        filings_repo.list_ids.return_value = None

        config = RetrievalConfig(
            vector_search=VectorSearchConfig(enabled=True, oversample_k=5, similarity_threshold=0.0),
            keyword_search=KeywordSearchConfig(enabled=False),
            fusion=FusionConfig(top_k=5),
            reranking=RerankingConfig(enabled=False),
            final_top_k=2,
        )

        vector_retriever = MagicMock()
        vector_retriever.search.return_value = [(ids[0], 0.9), (ids[1], 0.8), (ids[2], 0.7)]

        retriever = Retriever(
            config=config, fusion=RRFFusion(),
            chunks_repo=chunks_repo, parent_chunks_repo=parents_repo, filings_repo=filings_repo,
            vector_retriever=vector_retriever,
        )
        results = retriever.retrieve(query="GPU", query_embedding=[0.1] * 8)

        assert len(results) == 2
