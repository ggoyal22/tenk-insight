from uuid import UUID

from retrieval.fusion.base import BaseFusion


class RRFFusion(BaseFusion):
    """Reciprocal Rank Fusion.

    Merges ranked lists by assigning each result a score of 1 / (k + rank)
    and summing across lists. Results appearing in multiple lists accumulate
    higher scores. k=60 is the empirically validated default from Cormack et al.,
    "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning
    Methods" (SIGIR 2009).
    """

    def __init__(self, k: int = 60) -> None:
        self._k = k

    def fuse(self, *ranked_lists: list[tuple[UUID, float]]) -> list[tuple[UUID, float]]:
        scores: dict[UUID, float] = {}
        for ranked_list in ranked_lists:
            assert all(
                ranked_list[i][1] >= ranked_list[i + 1][1]
                for i in range(len(ranked_list) - 1)
            ), "Each input list must be ordered by score descending — RRF assigns ranks by position."
            for rank, (chunk_id, _) in enumerate(ranked_list):
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (self._k + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
