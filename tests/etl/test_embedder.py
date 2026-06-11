from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from config.loader import EmbeddingConfig
from etl.embedder.sentence_transformer import SentenceTransformerEmbedder


def _config(dimension: int = 1024) -> EmbeddingConfig:
    return EmbeddingConfig(
        model="BAAI/bge-large-en-v1.5",
        dimension=dimension,
        batch_size=32,
        device="cpu",
    )


def _make_embedder(dimension: int = 1024, max_seq: int = 512, child_chunk_size: int = 256):
    """Build a SentenceTransformerEmbedder with a mocked underlying model."""
    with patch("etl.embedder.sentence_transformer.SentenceTransformer") as mock_cls:
        mock_model = MagicMock()
        mock_model.max_seq_length = max_seq
        mock_model.get_sentence_embedding_dimension.return_value = dimension
        mock_cls.return_value = mock_model
        embedder = SentenceTransformerEmbedder(_config(dimension), child_chunk_size)
    # Attach the mock model so tests can configure encode() behaviour.
    embedder._model = mock_model
    return embedder


def test_embed_returns_vectors_as_list():
    embedder = _make_embedder()
    vectors = np.array([[0.1] * 1024, [0.2] * 1024])
    embedder._model.encode.return_value = vectors

    result = embedder.embed(["text one", "text two"])

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == pytest.approx([0.1] * 1024)


def test_embed_raises_on_empty_list():
    embedder = _make_embedder()
    with pytest.raises(ValueError, match="empty"):
        embedder.embed([])


def test_dimension_property_matches_config():
    embedder = _make_embedder(dimension=768)
    assert embedder.dimension == 768


def test_model_name_property_matches_config():
    embedder = _make_embedder()
    assert embedder.model_name == "BAAI/bge-large-en-v1.5"


def test_init_raises_when_dim_mismatches_config():
    with patch("etl.embedder.sentence_transformer.SentenceTransformer") as mock_cls:
        mock_model = MagicMock()
        mock_model.max_seq_length = 512
        mock_model.get_sentence_embedding_dimension.return_value = 768  # ≠ config dimension 1024
        mock_cls.return_value = mock_model

        with pytest.raises(ValueError, match="dimension"):
            SentenceTransformerEmbedder(_config(dimension=1024), 256)


def test_init_raises_when_child_chunk_size_exceeds_max_seq():
    with patch("etl.embedder.sentence_transformer.SentenceTransformer") as mock_cls:
        mock_model = MagicMock()
        mock_model.max_seq_length = 128
        mock_model.get_sentence_embedding_dimension.return_value = 1024
        mock_cls.return_value = mock_model

        with pytest.raises(ValueError, match="max_seq_length"):
            SentenceTransformerEmbedder(_config(), child_chunk_size=256)  # 256 > 128
