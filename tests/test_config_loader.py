"""
Tests for config/loader.py.

Validation tests construct Pydantic models directly — no file I/O needed.
The cache test uses the real config/config.yaml and .env files.
"""

import os
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from config.loader import (
    AppConfig,
    ChunkingConfig,
    DatabaseConfig,
    EdgarConfig,
    EmbeddingConfig,
    EvaluationConfig,
    EvaluatorConfig,
    ExtractorConfig,
    GenerationConfig,
    JudgeLLMConfig,
    LLMConfig,
    LoggingConfig,
    ResultsConfig,
    RetrievalConfig,
    VectorIndexConfig,
    _require_env,
    _require_env_int,
    load_config,
    load_eval_config,
)
from tests.conftest import (
    VALID_CHUNKING,
    VALID_DATABASE,
    VALID_EDGAR,
    VALID_EMBEDDING,
    VALID_EVALUATION,
    VALID_GENERATION,
    VALID_LLM,
    VALID_LOGGING,
    VALID_RETRIEVAL,
    VALID_VECTOR_INDEX,
)

# These tests drive the real loader against config/config.yaml, so they need the
# settings the application requires at startup. Those come either from a local
# .env or from variables exported directly, which is how Docker and CI supply
# them. ENVIRONMENT stands in for the whole set: it is required, has no default,
# and anything that configures the app sets it. When it is present but another
# required variable is missing, the loader raises and the test fails, which is
# the intended signal.
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
_requires_real_config = pytest.mark.skipif(
    not _ENV_FILE.exists() and not os.environ.get("ENVIRONMENT"),
    reason="no .env file and no config environment — skipping integration test"
)


# ---------------------------------------------------------------------------
# Happy-path — confirm baseline constants are accepted
# ---------------------------------------------------------------------------

def test_retrieval_valid_config():
    config = RetrievalConfig(**VALID_RETRIEVAL)
    assert config.final_top_k == VALID_RETRIEVAL["final_top_k"]
    assert config.vector_search.similarity_threshold == VALID_RETRIEVAL["vector_search"]["similarity_threshold"]


def test_embedding_valid_config():
    config = EmbeddingConfig(**VALID_EMBEDDING)
    assert config.model == VALID_EMBEDDING["model"]
    assert config.device == VALID_EMBEDDING["device"]


def test_chunking_valid_config():
    config = ChunkingConfig(**VALID_CHUNKING)
    assert config.child_chunk_size == VALID_CHUNKING["child_chunk_size"]
    assert config.parent_chunk_size == VALID_CHUNKING["parent_chunk_size"]


def test_vector_index_valid_config():
    config = VectorIndexConfig(**VALID_VECTOR_INDEX)
    assert config.type == VALID_VECTOR_INDEX["type"]
    assert config.distance_function == VALID_VECTOR_INDEX["distance_function"]


def test_database_valid_config():
    config = DatabaseConfig(**VALID_DATABASE)
    assert config.host == VALID_DATABASE["host"]
    assert config.pool_size == VALID_DATABASE["pool_size"]


def test_edgar_valid_config():
    config = EdgarConfig(**VALID_EDGAR)
    assert config.tickers == VALID_EDGAR["tickers"]


# ---------------------------------------------------------------------------
# RetrievalConfig validation
# ---------------------------------------------------------------------------

def test_retrieval_rejects_negative_top_k():
    with pytest.raises(ValidationError):
        RetrievalConfig(**{**VALID_RETRIEVAL, "final_top_k": -1})


def test_retrieval_rejects_zero_top_k():
    with pytest.raises(ValidationError):
        RetrievalConfig(**{**VALID_RETRIEVAL, "final_top_k": 0})


def test_retrieval_rejects_similarity_threshold_above_one():
    overrides = {**VALID_RETRIEVAL, "vector_search": {**VALID_RETRIEVAL["vector_search"], "similarity_threshold": 1.5}}
    with pytest.raises(ValidationError):
        RetrievalConfig(**overrides)


def test_retrieval_rejects_negative_similarity_threshold():
    overrides = {**VALID_RETRIEVAL, "vector_search": {**VALID_RETRIEVAL["vector_search"], "similarity_threshold": -0.1}}
    with pytest.raises(ValidationError):
        RetrievalConfig(**overrides)


# ---------------------------------------------------------------------------
# EmbeddingConfig validation
# ---------------------------------------------------------------------------

def test_embedding_rejects_invalid_device():
    # Valid values: "cpu", "cuda", "mps"
    with pytest.raises(ValidationError):
        EmbeddingConfig(**{**VALID_EMBEDDING, "device": "gpu"})


def test_embedding_rejects_zero_dimension():
    with pytest.raises(ValidationError):
        EmbeddingConfig(**{**VALID_EMBEDDING, "dimension": 0})


def test_embedding_rejects_zero_batch_size():
    with pytest.raises(ValidationError):
        EmbeddingConfig(**{**VALID_EMBEDDING, "batch_size": 0})


# ---------------------------------------------------------------------------
# ChunkingConfig validation
# ---------------------------------------------------------------------------

def test_chunking_rejects_zero_child_chunk_size():
    with pytest.raises(ValidationError):
        ChunkingConfig(**{**VALID_CHUNKING, "child_chunk_size": 0})


def test_chunking_rejects_zero_parent_chunk_size():
    with pytest.raises(ValidationError):
        ChunkingConfig(**{**VALID_CHUNKING, "parent_chunk_size": 0})


def test_chunking_rejects_child_overlap_equal_to_chunk_size():
    with pytest.raises(ValidationError):
        ChunkingConfig(**{**VALID_CHUNKING, "child_chunk_overlap": VALID_CHUNKING["child_chunk_size"]})


def test_chunking_rejects_child_overlap_greater_than_chunk_size():
    with pytest.raises(ValidationError):
        ChunkingConfig(**{**VALID_CHUNKING, "child_chunk_overlap": VALID_CHUNKING["child_chunk_size"] + 1})


def test_chunking_rejects_parent_overlap_equal_to_chunk_size():
    with pytest.raises(ValidationError):
        ChunkingConfig(**{**VALID_CHUNKING, "parent_chunk_overlap": VALID_CHUNKING["parent_chunk_size"]})


def test_chunking_rejects_parent_overlap_greater_than_chunk_size():
    with pytest.raises(ValidationError):
        ChunkingConfig(**{**VALID_CHUNKING, "parent_chunk_overlap": VALID_CHUNKING["parent_chunk_size"] + 1})


# ---------------------------------------------------------------------------
# LoggingConfig validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("invalid_level", ["VERBOSE", "TRACE", "info", "debug", ""])
def test_logging_rejects_invalid_level(invalid_level):
    with pytest.raises(ValidationError):
        LoggingConfig(level=invalid_level)


# ---------------------------------------------------------------------------
# VectorIndexConfig validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("invalid_type", ["flat", "annoy", "ivfflat", ""])
def test_vector_index_rejects_invalid_type(invalid_type):
    with pytest.raises(ValidationError):
        VectorIndexConfig(**{**VALID_VECTOR_INDEX, "type": invalid_type})


@pytest.mark.parametrize("invalid_fn", ["euclidean", "inner_product", ""])
def test_vector_index_rejects_invalid_distance_function(invalid_fn):
    with pytest.raises(ValidationError):
        VectorIndexConfig(**{**VALID_VECTOR_INDEX, "distance_function": invalid_fn})


# ---------------------------------------------------------------------------
# DatabaseConfig validation
# ---------------------------------------------------------------------------

def test_database_rejects_zero_port():
    with pytest.raises(ValidationError):
        DatabaseConfig(**{**VALID_DATABASE, "port": 0})


def test_database_rejects_port_above_max():
    with pytest.raises(ValidationError):
        DatabaseConfig(**{**VALID_DATABASE, "port": 65536})


def test_database_rejects_zero_pool_size():
    with pytest.raises(ValidationError):
        DatabaseConfig(**{**VALID_DATABASE, "pool_size": 0})


def test_database_rejects_empty_password():
    with pytest.raises(ValidationError):
        DatabaseConfig(**{**VALID_DATABASE, "password": ""})


def test_database_password_is_secret_str():
    config = DatabaseConfig(**VALID_DATABASE)
    assert "test-password-not-real" not in str(config)
    assert "test-password-not-real" not in repr(config)
    assert config.password.get_secret_value() == "test-password-not-real"


# ---------------------------------------------------------------------------
# EdgarConfig validation
# ---------------------------------------------------------------------------

def test_edgar_rejects_pre_edgar_year():
    with pytest.raises(ValidationError):
        EdgarConfig(**{**VALID_EDGAR, "years": [1992]})


def test_edgar_rejects_future_year():
    with pytest.raises(ValidationError):
        EdgarConfig(**{**VALID_EDGAR, "years": [date.today().year + 1]})


def test_edgar_normalizes_tickers_to_uppercase():
    config = EdgarConfig(**{**VALID_EDGAR, "tickers": ["nvda", " msft "]})
    assert config.tickers == ["NVDA", "MSFT"]


def test_edgar_rejects_empty_tickers():
    with pytest.raises(ValidationError):
        EdgarConfig(**{**VALID_EDGAR, "tickers": []})


def test_edgar_rejects_empty_form_types():
    with pytest.raises(ValidationError):
        EdgarConfig(**{**VALID_EDGAR, "form_types": []})


def test_edgar_rejects_empty_years():
    with pytest.raises(ValidationError):
        EdgarConfig(**{**VALID_EDGAR, "years": []})


# ---------------------------------------------------------------------------
# AppConfig validation
# ---------------------------------------------------------------------------

def test_app_config_rejects_invalid_environment():
    with pytest.raises(ValidationError):
        AppConfig(
            environment="staging",
            edgar=EdgarConfig(**VALID_EDGAR),
            database=DatabaseConfig(**VALID_DATABASE),
            embedding=EmbeddingConfig(**VALID_EMBEDDING),
            chunking=ChunkingConfig(**VALID_CHUNKING),
            vector_index=VectorIndexConfig(**VALID_VECTOR_INDEX),
            retrieval=RetrievalConfig(**VALID_RETRIEVAL),
            llm=LLMConfig(**VALID_LLM),
            generation=GenerationConfig(**VALID_GENERATION),
            logging=LoggingConfig(**VALID_LOGGING),
        )


# ---------------------------------------------------------------------------
# LLMConfig validation
# ---------------------------------------------------------------------------

def test_llm_valid_config():
    config = LLMConfig(**VALID_LLM)
    assert config.provider == VALID_LLM["provider"]
    assert config.model == VALID_LLM["model"]
    assert config.base_url == VALID_LLM["base_url"]
    assert config.api_key is None


def test_llm_api_key_is_secret_str():
    config = LLMConfig(**{**VALID_LLM, "api_key": "sk-secret-key"})
    assert "sk-secret-key" not in str(config)
    assert "sk-secret-key" not in repr(config)
    assert config.api_key.get_secret_value() == "sk-secret-key"


def test_llm_accepts_none_base_url():
    config = LLMConfig(**{**VALID_LLM, "base_url": None})
    assert config.base_url is None


@pytest.mark.parametrize("invalid_provider", ["anthropic", "cohere", "vllm", "claude", ""])
def test_llm_rejects_invalid_provider(invalid_provider):
    with pytest.raises(ValidationError):
        LLMConfig(**{**VALID_LLM, "provider": invalid_provider})


def test_llm_rejects_temperature_above_max():
    with pytest.raises(ValidationError):
        LLMConfig(**{**VALID_LLM, "temperature": 2.1})


def test_llm_rejects_negative_temperature():
    with pytest.raises(ValidationError):
        LLMConfig(**{**VALID_LLM, "temperature": -0.1})


def test_llm_rejects_zero_max_tokens():
    with pytest.raises(ValidationError):
        LLMConfig(**{**VALID_LLM, "max_tokens": 0})


def test_llm_rejects_zero_timeout():
    with pytest.raises(ValidationError):
        LLMConfig(**{**VALID_LLM, "timeout": 0})


# ---------------------------------------------------------------------------
# GenerationConfig validation
# ---------------------------------------------------------------------------

def test_generation_valid_config():
    config = GenerationConfig(**VALID_GENERATION)
    assert config.hyde.enabled == VALID_GENERATION["hyde"]["enabled"]
    assert config.reflection.max_iterations == VALID_GENERATION["reflection"]["max_iterations"]
    assert config.hop.max_hops == VALID_GENERATION["hop"]["max_hops"]


def test_generation_defaults_are_sensible():
    config = GenerationConfig()
    assert config.hyde.enabled is True
    assert config.reflection.enabled is True
    assert config.reflection.max_iterations > 0
    assert config.hop.max_hops > 0


def test_generation_rejects_zero_max_iterations():
    with pytest.raises(ValidationError):
        GenerationConfig(**{**VALID_GENERATION, "reflection": {"enabled": True, "max_iterations": 0}})


def test_generation_rejects_zero_max_hops():
    with pytest.raises(ValidationError):
        GenerationConfig(**{**VALID_GENERATION, "hop": {"enabled": False, "max_hops": 0}})


# ---------------------------------------------------------------------------
# _require_env validation
# ---------------------------------------------------------------------------

def test_require_env_rejects_empty_string(monkeypatch):
    monkeypatch.setenv("TEST_EMPTY_VAR", "")
    with pytest.raises(RuntimeError, match="is not set or is empty"):
        _require_env("TEST_EMPTY_VAR")


def test_require_env_rejects_missing_key():
    with pytest.raises(RuntimeError, match="is not set or is empty"):
        _require_env("TEST_NONEXISTENT_VAR_XYZ")


def test_require_env_rejects_whitespace_only(monkeypatch):
    monkeypatch.setenv("TEST_WHITESPACE_VAR", "   ")
    with pytest.raises(RuntimeError, match="is not set or is empty"):
        _require_env("TEST_WHITESPACE_VAR")


def test_require_env_int_rejects_non_integer(monkeypatch):
    monkeypatch.setenv("TEST_BAD_INT_VAR", "abc")
    with pytest.raises(RuntimeError, match="must be an integer"):
        _require_env_int("TEST_BAD_INT_VAR")


# ---------------------------------------------------------------------------
# Cache behaviour
# ---------------------------------------------------------------------------

@_requires_real_config
def test_load_config_returns_same_instance():
    # patch.dict restores os.environ to its original state after the test,
    # preventing load_dotenv side effects from leaking into subsequent tests
    load_config.cache_clear()
    try:
        with patch.dict(os.environ, {}, clear=False):
            config_a = load_config()
            config_b = load_config()
            assert config_a is config_b  # same object in memory, not just equal values
            assert config_a.embedding.model == "BAAI/bge-large-en-v1.5"
    finally:
        load_config.cache_clear()


# ---------------------------------------------------------------------------
# EvaluationConfig validation
# ---------------------------------------------------------------------------

def test_evaluation_valid_config(monkeypatch):
    monkeypatch.setenv("PHOENIX_DB_PATH", "/tmp/test.db")
    config = EvaluationConfig(**VALID_EVALUATION)
    assert config.extractor.backend == "phoenix"
    assert config.evaluator.backend == "ragas"
    assert config.evaluator.judge_llm.provider == "openai"
    assert config.golden_path is None
    assert config.results.phoenix_annotations is False


def test_evaluation_defaults_are_sensible(monkeypatch):
    monkeypatch.setenv("PHOENIX_DB_PATH", "/tmp/test.db")
    config = EvaluationConfig()
    assert config.extractor.backend == "phoenix"
    assert config.evaluator.backend == "ragas"
    assert "faithfulness" in config.metrics
    assert "context_recall" in config.metrics
    assert "single" in config.datasets
    assert config.golden_path is None
    assert config.results.results_dir == "data/eval_results"


def test_evaluation_requires_phoenix_db_path(monkeypatch):
    monkeypatch.delenv("PHOENIX_DB_PATH", raising=False)
    with pytest.raises(ValidationError, match="PHOENIX_DB_PATH"):
        EvaluationConfig(**VALID_EVALUATION)


def test_evaluation_rejects_invalid_extractor_backend(monkeypatch):
    monkeypatch.setenv("PHOENIX_DB_PATH", "/tmp/test.db")
    with pytest.raises(ValidationError):
        EvaluationConfig(**{**VALID_EVALUATION, "extractor": {"backend": "datadog"}})


def test_evaluation_rejects_invalid_evaluator_backend(monkeypatch):
    monkeypatch.setenv("PHOENIX_DB_PATH", "/tmp/test.db")
    with pytest.raises(ValidationError):
        EvaluationConfig(**{**VALID_EVALUATION, "evaluator": {"backend": "deepeval"}})


def test_evaluation_rejects_invalid_judge_llm_provider(monkeypatch):
    monkeypatch.setenv("PHOENIX_DB_PATH", "/tmp/test.db")
    invalid_evaluator = {**VALID_EVALUATION["evaluator"], "judge_llm": {"provider": "ollama", "model": "llama3"}}
    with pytest.raises(ValidationError):
        EvaluationConfig(**{**VALID_EVALUATION, "evaluator": invalid_evaluator})


def test_evaluation_judge_llm_api_key_is_secret_str(monkeypatch):
    monkeypatch.setenv("PHOENIX_DB_PATH", "/tmp/test.db")
    evaluator = {**VALID_EVALUATION["evaluator"], "judge_llm": {**VALID_EVALUATION["evaluator"]["judge_llm"], "api_key": "sk-secret"}}
    config = EvaluationConfig(**{**VALID_EVALUATION, "evaluator": evaluator})
    assert "sk-secret" not in str(config)
    assert "sk-secret" not in repr(config)
    assert config.evaluator.judge_llm.api_key.get_secret_value() == "sk-secret"


def test_evaluation_golden_path_none_is_valid(monkeypatch):
    monkeypatch.setenv("PHOENIX_DB_PATH", "/tmp/test.db")
    config = EvaluationConfig(**{**VALID_EVALUATION, "golden_path": None})
    assert config.golden_path is None


def test_evaluation_golden_path_string_is_accepted(monkeypatch):
    monkeypatch.setenv("PHOENIX_DB_PATH", "/tmp/test.db")
    config = EvaluationConfig(**{**VALID_EVALUATION, "golden_path": "data/golden/questions.yaml"})
    assert config.golden_path == "data/golden/questions.yaml"


@pytest.mark.parametrize("invalid_level", ["VERBOSE", "TRACE", "info", "debug", ""])
def test_evaluation_rejects_invalid_log_level(invalid_level, monkeypatch):
    monkeypatch.setenv("PHOENIX_DB_PATH", "/tmp/test.db")
    with pytest.raises(ValidationError):
        EvaluationConfig(**{**VALID_EVALUATION, "log_level": invalid_level})


def test_evaluation_rejects_anthropic_judge_provider(monkeypatch):
    monkeypatch.setenv("PHOENIX_DB_PATH", "/tmp/test.db")
    invalid_evaluator = {**VALID_EVALUATION["evaluator"], "judge_llm": {"provider": "anthropic", "model": "claude-3"}}
    with pytest.raises(ValidationError):
        EvaluationConfig(**{**VALID_EVALUATION, "evaluator": invalid_evaluator})


# ---------------------------------------------------------------------------
# load_eval_config cache behaviour
# ---------------------------------------------------------------------------

@_requires_real_config
def test_load_eval_config_returns_same_instance(monkeypatch):
    monkeypatch.setenv("PHOENIX_DB_PATH", "/tmp/test.db")
    monkeypatch.setenv("JUDGE_LLM_API_KEY", "sk-test-key")
    load_eval_config.cache_clear()
    try:
        with patch.dict(os.environ, {}, clear=False):
            config_a = load_eval_config()
            config_b = load_eval_config()
            assert config_a is config_b
            assert config_a.evaluator.judge_llm.model == "gpt-4o-mini"
    finally:
        load_eval_config.cache_clear()
