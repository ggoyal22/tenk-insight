import logging
from uuid import UUID

from config.loader import RetrievalConfig
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
        query: str,
        query_embedding: list[float],
        filters: MetadataFilter | None = None,
    ) -> list[RetrievalResult]:
        filing_ids = self._resolve_filing_ids(filters)
        section = filters.section if filters else None

        ranked_lists: list[list[tuple[UUID, float]]] = []

        if self._vector:
            vector_results = self._vector.search(
                query_embedding=query_embedding,
                top_k=self._config.vector_search.oversample_k,
                filing_ids=filing_ids,
                section=section,
            )
            if vector_results:
                ranked_lists.append(vector_results)

        if self._keyword:
            keyword_results = self._keyword.search(
                query=query,
                top_k=self._config.keyword_search.top_k,
                filing_ids=filing_ids,
                section=section,
            )
            if keyword_results:
                ranked_lists.append(keyword_results)

        if not ranked_lists:
            logger.warning("Both vector and keyword search returned no results.")
            return []

        fused = self._fusion.fuse(*ranked_lists)
        fused = fused[: self._config.fusion.top_k]

        results = self._enrich(fused)

        if self._reranker and self._config.reranking.enabled:
            return self._reranker.rerank(query, results, self._config.reranking.top_k)

        return results[: self._config.final_top_k]

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

    def _enrich(self, fused: list[tuple[UUID, float]]) -> list[RetrievalResult]:
        """Batch-fetch chunks, parent chunks, and filings; assemble RetrievalResult objects."""
        if not fused:
            return []

        chunk_ids = [cid for cid, _ in fused]
        score_map = {cid: score for cid, score in fused}

        chunks = self._chunks.get_by_ids(chunk_ids)
        chunk_map = {c.id: c for c in chunks}

        parent_ids = [c.parent_chunk_id for c in chunks]
        parent_map = {p.id: p for p in self._parents.get_by_ids(parent_ids)}

        filing_ids = list({c.filing_id for c in chunks})
        filing_map = {f.id: f for f in self._filings.get_by_ids(filing_ids)}

        results: list[RetrievalResult] = []
        for chunk_id in chunk_ids:
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
                score=score_map[chunk_id],
                chunk=chunk,
                parent_chunk=parent,
                filing=filing,
            ))

        return results
