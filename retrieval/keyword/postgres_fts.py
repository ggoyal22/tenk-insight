import logging
from uuid import UUID

from db.repositories.chunks import ChunksRepo
from retrieval.keyword.base import BaseKeywordRetriever

logger = logging.getLogger(__name__)


class PostgresFTSRetriever(BaseKeywordRetriever):
    def __init__(self, chunks_repo: ChunksRepo, query_mode: str = "web") -> None:
        self._repo = chunks_repo
        self._query_mode = query_mode

    def search(
        self,
        query: str,
        top_k: int,
        filing_ids: list[UUID] | None = None,
        section: str | None = None,
    ) -> list[tuple[UUID, float]]:
        results = self._repo.keyword_search(
            query=query,
            top_k=top_k,
            filing_ids=filing_ids,
            section=section,
            query_mode=self._query_mode,
        )

        if not results:
            logger.debug(
                "Keyword search returned no results for query %r "
                "(filing_ids=%s, section=%s).",
                query,
                filing_ids,
                section,
            )

        return results
