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
    ) -> str:
        """Persist a rating and return the new row's id so a comment can be
        attached later."""
        ...

    @abstractmethod
    def update_comment(self, feedback_id: str, comment: str) -> None: ...
