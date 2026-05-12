from abc import ABC, abstractmethod


class FeedbackRepo(ABC):
    @abstractmethod
    def create_table(self) -> None: ...

    @abstractmethod
    def insert_feedback(
        self,
        query: str,
        answer: str,
        rating: bool,
        comment: str | None,
    ) -> None: ...
