import logging

from sentence_transformers import SentenceTransformer

from config.loader import EmbeddingConfig
from etl.embedder.base import Embedder

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, config: EmbeddingConfig, child_chunk_size: int) -> None:
        self._model = SentenceTransformer(config.model, device=config.device)
        self._model_name = config.model
        self._batch_size = config.batch_size
        self._dim = config.dimension

        max_seq = self._model.max_seq_length
        if child_chunk_size > max_seq:
            raise ValueError(
                f"child_chunk_size ({child_chunk_size}) exceeds model max_seq_length "
                f"({max_seq}) for {config.model!r}. Reduce child_chunk_size in config.yaml."
            )

        actual_dim = self._model.get_sentence_embedding_dimension()
        if actual_dim != config.dimension:
            raise ValueError(
                f"Model {config.model!r} produces {actual_dim}-d embeddings but "
                f"config.embedding.dimension is {config.dimension}. Update config.yaml."
            )

        logger.info(
            "Loaded embedding model %r — dim=%d, max_seq=%d, device=%s",
            config.model, actual_dim, max_seq, config.device,
        )

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._validate_texts(texts)
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # required for cosine similarity with bge models
        )
        logger.info("Embedded %d texts → shape %s", len(texts), vectors.shape)
        return vectors.tolist()
