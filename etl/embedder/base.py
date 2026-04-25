from abc import ABC, abstractmethod


class Embedder(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimension — must match config.embedding.dimension and the DB schema."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Embedding model identifier — written to child chunks for traceability."""
        ...

    def _validate_texts(self, texts: list[str]) -> None:
        if not texts:
            raise ValueError("Cannot embed an empty list of texts.")

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into dense vectors.

        Args:
            texts: list of text strings to embed

        Returns:
            list of embedding vectors, same order as input
        """
        ...
