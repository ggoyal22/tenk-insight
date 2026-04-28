from abc import ABC, abstractmethod
from uuid import UUID


class BaseFusion(ABC):
    @abstractmethod
    def fuse(self, *ranked_lists: list[tuple[UUID, float]]) -> list[tuple[UUID, float]]:
        """Merge ranked result lists into a single ranked list.

        Each input list contains (chunk_id, score) pairs ordered by relevance
        descending. Scores across lists are not assumed to be on the same scale.
        Returns a merged list in descending order of the fused score.
        """
        ...
