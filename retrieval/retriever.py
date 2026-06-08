import logging
from dataclasses import replace
from uuid import UUID

from config.loader import RetrievalConfig
from db.models import ChunkRecord
from db.repositories.chunks import ChunksRepo
from db.repositories.filings import FilingsRepo
from db.repositories.parent_chunks import ParentChunksRepo
from retrieval.fusion.base import BaseFusion
from retrieval.keyword.base import BaseKeywordRetriever
from retrieval.reranker.base import BaseReranker
from retrieval.types import MetadataFilter, RetrievalResult
from retrieval.vector.base import BaseVectorRetriever

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(
        self,
        config: RetrievalConfig,
        fusion: BaseFusion,
        chunks_repo: ChunksRepo,
        parent_chunks_repo: ParentChunksRepo,
        filings_repo: FilingsRepo,
        vector_retriever: BaseVectorRetriever | None = None,
        keyword_retriever: BaseKeywordRetriever | None = None,
        reranker: BaseReranker | None = None,
    ) -> None:
        self._config = config
        self._vector = vector_retriever
        self._keyword = keyword_retriever
        self._fusion = fusion
        self._reranker = reranker
        self._chunks = chunks_repo
        self._parents = parent_chunks_repo
        self._filings = filings_repo

    def retrieve(
        self,
        keyword_query: str,
        semantic_embedding: list[float],
        filters: MetadataFilter | None = None,
        rerank_query: str | None = None,
        exclude_parent_ids: list[UUID] | None = None,
    ) -> list[RetrievalResult]:
        results = self._search(
            keyword_query, semantic_embedding, filters, rerank_query, exclude_parent_ids
        )

        # A section filter that even the best chunk can't clear (or that returns
        # nothing) was likely too narrow to reach the answer — companies vary in
        # which 10-K item holds a given figure. Search again without the section
        # (ticker and fiscal year are kept) and merge both sets, letting the
        # reranker arbitrate across them. Merging never discards a strong
        # section-filtered result, so widening can only help.
        if self._section_too_restrictive(filters, results):
            widened = replace(filters, section=None)
            relaxed = self._search(
                keyword_query, semantic_embedding, widened, rerank_query, exclude_parent_ids
            )
            if relaxed:
                results = self._merge(results, relaxed)

        return results

    def _merge(
        self, primary: list[RetrievalResult], extra: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        """Combine two reranked result sets into a single top-k ordered by reranker score.

        Reranker scores are per-(query, chunk) independent, so taking the top-k by score
        over the union of two already-top-k sets yields the true top-k over their union.
        Deduplicates by chunk id, keeping the higher-scored occurrence.
        """
        best: dict[UUID, RetrievalResult] = {}
        for r in (*primary, *extra):
            current = best.get(r.chunk.id)
            if current is None or (r.reranker_score or 0.0) > (current.reranker_score or 0.0):
                best[r.chunk.id] = r
        return sorted(
            best.values(), key=lambda r: r.reranker_score or 0.0, reverse=True
        )[: self._config.reranking.top_k]

    def _section_too_restrictive(
        self, filters: MetadataFilter | None, results: list[RetrievalResult]
    ) -> bool:
        """True when a section filter was applied but the results fail the relevance floor.

        The floor is a reranker-score concept, so this only applies when reranking is
        active. An empty result set counts as failing — it is the strongest signal the
        section excluded the relevant content.
        """
        if not self._config.section_retry.enabled:
            return False
        if not (self._reranker and self._config.reranking.enabled):
            return False

        section = filters.section if filters else None
        if section:
            section = section.strip().rstrip(".")
            if section.lower() == "null":
                section = None
        if not section:
            return False

        if not results:
            return True
        scores = [r.reranker_score for r in results if r.reranker_score is not None]
        return bool(scores) and max(scores) < self._config.section_retry.min_top_score

    def _search(
        self,
        keyword_query: str,
        semantic_embedding: list[float],
        filters: MetadataFilter | None = None,
        rerank_query: str | None = None,
        exclude_parent_ids: list[UUID] | None = None,
    ) -> list[RetrievalResult]:
        filing_ids = self._resolve_filing_ids(filters)
        raw_section = filters.section if filters else None
        section = raw_section.strip().rstrip(".") if raw_section else None
        if section and section.lower() == "null":
            section = None

        ranked_lists: list[list[tuple[UUID, float]]] = []
        vector_scores: dict[UUID, float] = {}
        keyword_scores: dict[UUID, float] = {}

        if self._vector:
            vector_results = self._vector.search(
                query_embedding=semantic_embedding,
                top_k=self._config.vector_search.oversample_k,
                filing_ids=filing_ids,
                section=section,
                exclude_parent_ids=exclude_parent_ids,
            )
            if vector_results:
                vector_scores = {cid: s for cid, s in vector_results}
                ranked_lists.append(vector_results)

        if self._keyword:
            keyword_results = self._keyword.search(
                query=keyword_query,
                top_k=self._config.keyword_search.top_k,
                filing_ids=filing_ids,
                section=section,
                exclude_parent_ids=exclude_parent_ids,
            )
            if keyword_results:
                keyword_scores = {cid: s for cid, s in keyword_results}
                ranked_lists.append(keyword_results)

        if not ranked_lists:
            logger.warning("Both vector and keyword search returned no results.")
            return []

        fused = self._fusion.fuse(*ranked_lists)
        fused = fused[: self._config.fusion.top_k]

        chunks = self._chunks.get_by_ids_no_embedding([cid for cid, _ in fused])

        reranking_active = bool(self._reranker and self._config.reranking.enabled)
        reranked: list[tuple[UUID, float]] | None = None

        if reranking_active:
            reranked = self._reranker.rerank(rerank_query or keyword_query, chunks)

        results = self._enrich(fused, vector_scores, keyword_scores, chunks, reranked=reranked)
        top_k = self._config.reranking.top_k if reranking_active else self._config.final_top_k
        return results[:top_k]

    def _resolve_filing_ids(self, filters: MetadataFilter | None) -> list[UUID] | None:
        if not filters or not self._config.metadata_filtering.enabled:
            return None

        filing_filters: dict = {}
        if filters.ticker:
            filing_filters["ticker"] = filters.ticker.upper()
        if filters.form_type:
            filing_filters["form_type"] = filters.form_type
        if filters.fiscal_year:
            filing_filters["fiscal_year"] = filters.fiscal_year

        if not filing_filters:
            return None

        filing_ids = self._filings.list_ids(filing_filters)
        if not filing_ids:
            logger.warning("No filings found matching filters: %s", filing_filters)
            return []

        return filing_ids

    def _enrich(
        self,
        fused: list[tuple[UUID, float]],
        vector_scores: dict[UUID, float],
        keyword_scores: dict[UUID, float],
        chunks: list[ChunkRecord],
        reranked: list[tuple[UUID, float]] | None = None,
    ) -> list[RetrievalResult]:
        """Assemble RetrievalResult objects from pre-fetched chunks, parent chunks, and filings.

        chunks are fetched once in retrieve() and passed in to avoid a duplicate DB call.
        When reranked is provided, iteration follows reranker order so parent dedup
        keeps the highest-reranker-scored child per parent. RRF scores from fused
        are preserved in RetrievalResult.score.
        """
        if not fused:
            return []

        chunk_ids = [cid for cid, _ in fused]
        rrf_score_map = {cid: score for cid, score in fused}
        reranker_score_map = {cid: score for cid, score in reranked} if reranked else {}

        chunk_map = {c.id: c for c in chunks}

        parent_ids = [c.parent_chunk_id for c in chunks]
        parent_map = {p.id: p for p in self._parents.get_by_ids(parent_ids)}

        filing_ids = list({c.filing_id for c in chunks})
        filing_map = {f.id: f for f in self._filings.get_by_ids(filing_ids)}

        iteration_order = [cid for cid, _ in reranked] if reranked else chunk_ids

        results: list[RetrievalResult] = []
        for chunk_id in iteration_order:
            chunk = chunk_map.get(chunk_id)
            if chunk is None:
                logger.warning("Chunk %s not found during enrichment — skipping.", chunk_id)
                continue

            filing = filing_map.get(chunk.filing_id)
            if filing is None:
                logger.warning("Filing %s not found for chunk %s — skipping.", chunk.filing_id, chunk_id)
                continue

            parent = parent_map.get(chunk.parent_chunk_id)
            if parent is None:
                raise RuntimeError(
                    f"Chunk {chunk_id} references parent {chunk.parent_chunk_id} which was not found. "
                    "This indicates incomplete ingestion — re-run the ingestion pipeline."
                )

            results.append(RetrievalResult(
                score=rrf_score_map[chunk_id],
                vector_score=vector_scores.get(chunk_id),
                keyword_score=keyword_scores.get(chunk_id),
                reranker_score=reranker_score_map.get(chunk_id),
                chunk=chunk,
                parent_chunk=parent,
                filing=filing,
            ))

        return results
