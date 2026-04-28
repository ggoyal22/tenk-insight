from abc import ABC, abstractmethod

from retrieval.types import RetrievalResult


class BaseReranker(ABC):
    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Rerank results and return the top_k most relevant.

        Scores in the returned RetrievalResult objects reflect the reranker's
        own relevance signal, replacing the upstream RRF scores.
        """
        ...
