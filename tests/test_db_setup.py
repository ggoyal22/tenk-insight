"""
Tests for db/setup.py.

_substitute() is pure Python with no external dependencies — tested fully here.
_schema_exists() and run() require a live database — skipped when .env is absent.
"""

import re
from pathlib import Path

import pytest

from config.loader import (
    AppConfig,
    ChunkingConfig,
    DatabaseConfig,
    EdgarConfig,
    EmbeddingConfig,
    LoggingConfig,
    RetrievalConfig,
    VectorIndexConfig,
    VectorStoreConfig,
)
from db.setup import _substitute
from tests.conftest import (
    VALID_CHUNKING,
    VALID_DATABASE,
    VALID_EDGAR,
    VALID_EMBEDDING,
    VALID_LOGGING,
    VALID_RETRIEVAL,
    VALID_VECTOR_INDEX,
    VALID_VECTOR_STORE,
)

_SCHEMA_TEMPLATE = Path(__file__).resolve().parent.parent / "db" / "schema.template.sql"


def _make_config(**vector_index_overrides) -> AppConfig:
    """Build a minimal AppConfig for testing _substitute."""
    return AppConfig(
        environment="test",
        edgar=EdgarConfig(**VALID_EDGAR),
        database=DatabaseConfig(**VALID_DATABASE),
        vector_store=VectorStoreConfig(**VALID_VECTOR_STORE),
        embedding=EmbeddingConfig(**VALID_EMBEDDING),
        chunking=ChunkingConfig(**VALID_CHUNKING),
        vector_index=VectorIndexConfig(**{**VALID_VECTOR_INDEX, **vector_index_overrides}),
        retrieval=RetrievalConfig(**VALID_RETRIEVAL),
        logging=LoggingConfig(**VALID_LOGGING),
    )


# Minimal template that contains all four placeholders for focused unit tests
_MINIMAL_TEMPLATE = (
    "VECTOR({embedding_dimension}) "
    "hnsw (embedding {hnsw_ops_class}) "
    "m = {hnsw_m}, ef_construction = {hnsw_ef_construction}"
)


# ---------------------------------------------------------------------------
# _substitute — placeholder substitution
# ---------------------------------------------------------------------------

def test_substitute_embedding_dimension():
    result = _substitute(_MINIMAL_TEMPLATE, _make_config())
    assert "VECTOR(1024)" in result


def test_substitute_hnsw_ops_class_cosine():
    result = _substitute(_MINIMAL_TEMPLATE, _make_config(distance_function="cosine"))
    assert "vector_cosine_ops" in result


def test_substitute_hnsw_ops_class_l2():
    result = _substitute(_MINIMAL_TEMPLATE, _make_config(distance_function="l2"))
    assert "vector_l2_ops" in result


def test_substitute_hnsw_m():
    result = _substitute(_MINIMAL_TEMPLATE, _make_config(hnsw_m=32))
    assert "m = 32" in result


def test_substitute_hnsw_ef_construction():
    result = _substitute(_MINIMAL_TEMPLATE, _make_config(hnsw_ef_construction=128))
    assert "ef_construction = 128" in result


def test_substitute_preserves_sql_regex_braces():
    # SQL regex patterns like '^[0-9a-f]{64}$' must survive substitution unchanged.
    # Using str.format() would raise KeyError on {64} — str.replace() avoids this.
    template = "CHECK (content_hash ~ '^[0-9a-f]{64}$') VECTOR({embedding_dimension})"
    result = _substitute(template, _make_config())
    assert "'^[0-9a-f]{64}$'" in result
    assert "VECTOR(1024)" in result


def test_substitute_resolves_all_named_placeholders_in_real_template():
    # Read the real template and verify no named placeholders remain after substitution.
    template = _SCHEMA_TEMPLATE.read_text()
    result = _substitute(template, _make_config())
    # Named placeholders follow {snake_case_word} — all must be resolved
    remaining = re.findall(r"\{[a-z_]+\}", result)
    assert remaining == [], f"Unresolved placeholders after substitution: {remaining}"
