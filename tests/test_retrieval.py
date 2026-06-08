"""Unit tests for retrieval components — no database required."""

from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest

from config.loader import (
    FusionConfig, KeywordSearchConfig, RetrievalConfig, RerankingConfig,
    SectionRetryConfig, VectorSearchConfig,
)
from db.models import ChunkRecord, FilingRecord, ParentChunkRecord
from retrieval.fusion.rrf import RRFFusion
from retrieval.reranker.cross_encoder import CrossEncoderReranker
from retrieval.retriever import Retriever
from retrieval.types import MetadataFilter, RetrievalResult


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
    chunks_repo.get_by_ids_no_embedding.return_value = [chunk]

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
        results = retriever.retrieve(keyword_query="GPU", semantic_embedding=[0.1] * 8)

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
        results = retriever.retrieve(keyword_query="GPU", semantic_embedding=[0.1] * 8)

        keyword_retriever.search.assert_called_once()
        assert len(results) == 1
        assert results[0].chunk.id == chunk_id

    def test_results_trimmed_to_final_top_k_when_reranker_disabled(self):
        # 3 chunks each with a distinct parent — final_top_k=2 caps at 2
        ids = [uuid4() for _ in range(3)]
        filing_id = uuid4()
        parent_ids = [uuid4() for _ in range(3)]
        filing = _filing_record(filing_id)
        parents = [_parent_record(pid, filing_id) for pid in parent_ids]
        chunks = [_chunk_record(cid, filing_id, pid) for cid, pid in zip(ids, parent_ids)]

        chunks_repo = MagicMock()
        chunks_repo.get_by_ids_no_embedding.return_value = chunks
        parents_repo = MagicMock()
        parents_repo.get_by_ids.return_value = parents
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
        results = retriever.retrieve(keyword_query="GPU", semantic_embedding=[0.1] * 8)

        assert len(results) == 2

    def test_sibling_chunks_from_same_parent_not_deduplicated(self):
        # 2 chunks sharing one parent — retriever now returns both; generate node deduplicates
        chunk_id_a, chunk_id_b = uuid4(), uuid4()
        filing_id = uuid4()
        shared_parent_id = uuid4()
        filing = _filing_record(filing_id)
        parent = _parent_record(shared_parent_id, filing_id)
        chunk_a = _chunk_record(chunk_id_a, filing_id, shared_parent_id)
        chunk_b = _chunk_record(chunk_id_b, filing_id, shared_parent_id)

        chunks_repo = MagicMock()
        chunks_repo.get_by_ids_no_embedding.return_value = [chunk_a, chunk_b]
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
            final_top_k=5,
        )

        vector_retriever = MagicMock()
        vector_retriever.search.return_value = [(chunk_id_a, 0.9), (chunk_id_b, 0.8)]

        retriever = Retriever(
            config=config, fusion=RRFFusion(),
            chunks_repo=chunks_repo, parent_chunks_repo=parents_repo, filings_repo=filings_repo,
            vector_retriever=vector_retriever,
        )
        results = retriever.retrieve(keyword_query="GPU", semantic_embedding=[0.1] * 8)

        assert len(results) == 2
        assert results[0].chunk.id == chunk_id_a
        assert results[1].chunk.id == chunk_id_b
        assert results[0].parent_chunk.id == results[1].parent_chunk.id == shared_parent_id


# ── CrossEncoderReranker tests ────────────────────────────────────────────────

class TestCrossEncoderReranker:
    @patch("retrieval.reranker.cross_encoder.CrossEncoder")
    def test_rerank_sorts_by_score_descending(self, MockCE):
        mock_model = MagicMock()
        mock_model.predict.return_value.tolist.return_value = [0.3, 0.9, 0.6]
        MockCE.return_value = mock_model

        reranker = CrossEncoderReranker("mock-model")
        filing_id, parent_id = uuid4(), uuid4()
        chunks = [_chunk_record(uuid4(), filing_id, parent_id) for _ in range(3)]
        ranked = reranker.rerank("test query", chunks)

        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)

    @patch("retrieval.reranker.cross_encoder.CrossEncoder")
    def test_rerank_returns_correct_chunk_id_and_score(self, MockCE):
        mock_model = MagicMock()
        mock_model.predict.return_value.tolist.return_value = [0.8]
        MockCE.return_value = mock_model

        reranker = CrossEncoderReranker("mock-model")
        filing_id, parent_id = uuid4(), uuid4()
        chunk = _chunk_record(uuid4(), filing_id, parent_id)
        ranked = reranker.rerank("test query", [chunk])

        assert len(ranked) == 1
        cid, score = ranked[0]
        assert cid == chunk.id
        assert score == 0.8

    @patch("retrieval.reranker.cross_encoder.CrossEncoder")
    def test_rerank_scores_child_text(self, MockCE):
        mock_model = MagicMock()
        mock_model.predict.return_value.tolist.return_value = [0.7]
        MockCE.return_value = mock_model

        reranker = CrossEncoderReranker("mock-model")
        filing_id, parent_id = uuid4(), uuid4()
        chunk = _chunk_record(uuid4(), filing_id, parent_id)
        reranker.rerank("test query", [chunk])

        pairs = mock_model.predict.call_args[0][0]
        assert pairs == [("test query", chunk.text)]

    @patch("retrieval.reranker.cross_encoder.CrossEncoder")
    def test_rerank_empty_returns_empty(self, MockCE):
        MockCE.return_value = MagicMock()
        reranker = CrossEncoderReranker("mock-model")
        assert reranker.rerank("query", []) == []


def _result(reranker_score: float) -> RetrievalResult:
    fid, pid, cid = uuid4(), uuid4(), uuid4()
    return RetrievalResult(
        score=0.0, vector_score=None, keyword_score=None, reranker_score=reranker_score,
        chunk=_chunk_record(cid, fid, pid),
        parent_chunk=_parent_record(pid, fid),
        filing=_filing_record(fid),
    )


class TestSectionRetry:
    """Retrieval widens an over-narrow section filter by re-searching without it and merging."""

    def _retriever(self, *, enabled: bool = True, min_top_score: float = 3.0) -> Retriever:
        config = RetrievalConfig(
            vector_search=VectorSearchConfig(enabled=True, oversample_k=5, similarity_threshold=0.0),
            keyword_search=KeywordSearchConfig(enabled=False),
            fusion=FusionConfig(top_k=5),
            reranking=RerankingConfig(enabled=True, top_k=5),
            section_retry=SectionRetryConfig(enabled=enabled, min_top_score=min_top_score),
            final_top_k=5,
        )
        return Retriever(
            config=config, fusion=RRFFusion(),
            chunks_repo=MagicMock(), parent_chunks_repo=MagicMock(), filings_repo=MagicMock(),
            vector_retriever=MagicMock(), keyword_retriever=None,
            reranker=MagicMock(),
        )

    def test_widens_and_merges_when_top_below_floor(self):
        r = self._retriever()
        low, high = _result(1.0), _result(5.0)
        with patch.object(r, "_search", side_effect=[[low], [high]]) as m:
            results = r.retrieve(
                keyword_query="x", semantic_embedding=[0.1] * 8,
                filters=MetadataFilter(section="Item 1"),
            )
        assert m.call_count == 2
        assert m.call_args_list[1].args[2].section is None  # second search drops the section
        assert [x.reranker_score for x in results] == [5.0, 1.0]  # merged, ordered by score

    def test_no_widen_when_top_meets_floor(self):
        r = self._retriever()
        good = _result(4.0)
        with patch.object(r, "_search", side_effect=[[good]]) as m:
            results = r.retrieve(
                keyword_query="x", semantic_embedding=[0.1] * 8,
                filters=MetadataFilter(section="Item 1"),
            )
        assert m.call_count == 1
        assert results == [good]

    def test_disabled_never_widens(self):
        r = self._retriever(enabled=False)
        low = _result(1.0)
        with patch.object(r, "_search", side_effect=[[low]]) as m:
            results = r.retrieve(
                keyword_query="x", semantic_embedding=[0.1] * 8,
                filters=MetadataFilter(section="Item 1"),
            )
        assert m.call_count == 1
        assert results == [low]

    def test_no_widen_without_section_filter(self):
        r = self._retriever()
        low = _result(1.0)  # below floor, but no section to relax
        with patch.object(r, "_search", side_effect=[[low]]) as m:
            results = r.retrieve(
                keyword_query="x", semantic_embedding=[0.1] * 8,
                filters=MetadataFilter(ticker="NVDA"),
            )
        assert m.call_count == 1
        assert results == [low]
