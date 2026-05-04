"""
Config loader for the SEC EDGAR RAG system.

Reads config/config.yaml for tunable parameters and .env for environment-specific
values (DB credentials, API keys). Returns validated, typed config instances.

Two entry points for two separate pipelines:
    load_config()      — query/execution phase (ingestion, retrieval, generation)
    load_eval_config() — evaluation phase (trace extraction, RAGAS scoring, result output)

Usage:
    from config.loader import load_config, load_eval_config, ensure_directories

    config = load_config()
    ensure_directories(config)   # call once at startup
    print(config.embedding.model)

    eval_cfg = load_eval_config()
    print(eval_cfg.evaluator.judge_llm.model)
"""

import os
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator


# ---------------------------------------------------------------------------
# Models for config.yaml sections
# ---------------------------------------------------------------------------

class EdgarConfig(BaseModel):
    tickers: list[str] = Field(min_length=1)
    form_types: list[str] = Field(min_length=1)
    years: list[int] = Field(min_length=1)
    raw_data_dir: Path              # resolved to absolute path at load time
    user_agent: str                 # e.g. "Project Name contact@example.com" — required by SEC fair-use policy

    @field_validator("tickers")
    @classmethod
    def normalize_tickers(cls, v: list[str]) -> list[str]:
        return [t.strip().upper() for t in v]

    @field_validator("years")
    @classmethod
    def years_must_be_valid(cls, v: list[int]) -> list[int]:
        current_year = date.today().year
        for year in v:
            if year < 1993 or year > current_year:
                raise ValueError(
                    f"Year {year} is out of range. EDGAR electronic filings start in 1993; "
                    f"future years (>{current_year}) are not valid."
                )
        return v


class EmbeddingConfig(BaseModel):
    model: str
    dimension: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    device: Literal["cpu", "cuda", "mps"]  # mps = Apple Silicon GPU


class ChunkingConfig(BaseModel):
    child_chunk_size: int = Field(gt=0)
    child_chunk_overlap: int = Field(ge=0)
    parent_chunk_size: int = Field(gt=0)
    parent_chunk_overlap: int = Field(ge=0)

    @model_validator(mode="after")
    def overlaps_must_be_less_than_chunk_sizes(self) -> "ChunkingConfig":
        if self.child_chunk_overlap >= self.child_chunk_size:
            raise ValueError(
                f"child_chunk_overlap ({self.child_chunk_overlap}) must be "
                f"less than child_chunk_size ({self.child_chunk_size})"
            )
        if self.parent_chunk_overlap >= self.parent_chunk_size:
            raise ValueError(
                f"parent_chunk_overlap ({self.parent_chunk_overlap}) must be "
                f"less than parent_chunk_size ({self.parent_chunk_size})"
            )
        return self


class VectorIndexConfig(BaseModel):
    type: Literal["hnsw"]
    distance_function: Literal["cosine", "l2", "dot"]
    hnsw_m: int = Field(default=16, gt=0)
    hnsw_ef_construction: int = Field(default=64, gt=0)


class MetadataFilteringConfig(BaseModel):
    enabled: bool = True


class VectorSearchConfig(BaseModel):
    enabled: bool = True
    quantization: Literal["none", "halfvec", "scalar"] = "halfvec"
    oversample_k: int = Field(gt=0, default=40)
    similarity_threshold: float = Field(ge=0.0, le=1.0, default=0.7)


class FtsConfig(BaseModel):
    query_mode: Literal["standard", "phrase", "web"] = "web"


class Bm25Config(BaseModel):
    k1: float = Field(gt=0, default=1.5)
    b: float = Field(ge=0.0, le=1.0, default=0.75)


class KeywordSearchConfig(BaseModel):
    enabled: bool = True
    implementation: Literal["fts", "bm25"] = "fts"
    top_k: int = Field(gt=0, default=20)
    fts: FtsConfig = Field(default_factory=FtsConfig)
    bm25: Bm25Config = Field(default_factory=Bm25Config)


class FusionConfig(BaseModel):
    implementation: Literal["rrf"] = "rrf"
    rrf_k: int = Field(gt=0, default=60)
    top_k: int = Field(gt=0, default=20)


class RerankingConfig(BaseModel):
    enabled: bool = False
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = Field(gt=0, default=5)


class RetrievalConfig(BaseModel):
    metadata_filtering: MetadataFilteringConfig = Field(default_factory=MetadataFilteringConfig)
    vector_search: VectorSearchConfig = Field(default_factory=VectorSearchConfig)
    keyword_search: KeywordSearchConfig = Field(default_factory=KeywordSearchConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    final_top_k: int = Field(gt=0, default=5)

    @model_validator(mode="after")
    def at_least_one_search_enabled(self) -> "RetrievalConfig":
        if not self.vector_search.enabled and not self.keyword_search.enabled:
            raise ValueError(
                "At least one of vector_search or keyword_search must be enabled."
            )
        return self

    @model_validator(mode="after")
    def pipeline_stages_are_consistent(self) -> "RetrievalConfig":
        if self.vector_search.enabled and self.vector_search.oversample_k < self.fusion.top_k:
            raise ValueError(
                f"vector_search.oversample_k ({self.vector_search.oversample_k}) must be "
                f">= fusion.top_k ({self.fusion.top_k}): vector search cannot provide more "
                "candidates than it fetches."
            )
        if self.reranking.enabled and self.reranking.top_k > self.fusion.top_k:
            raise ValueError(
                f"reranking.top_k ({self.reranking.top_k}) must be "
                f"<= fusion.top_k ({self.fusion.top_k}): reranker cannot receive more "
                "candidates than fusion produces."
            )
        if self.fusion.top_k < self.final_top_k:
            raise ValueError(
                f"fusion.top_k ({self.fusion.top_k}) must be >= final_top_k ({self.final_top_k}): "
                "the pipeline cannot widen results after fusion."
            )
        return self


class LLMConfig(BaseModel):
    provider: Literal["ollama", "openai"]  # extend as new providers are added to llm/factory.py
    model: str
    temperature: float = Field(ge=0.0, le=2.0, default=0.0)
    max_tokens: int = Field(gt=0, default=2048)
    timeout: int = Field(gt=0, default=120)
    base_url: str | None = None         # from LLM_BASE_URL; None for cloud APIs that use the SDK default
    api_key: SecretStr | None = None    # from LLM_API_KEY; None for local models


class HydeConfig(BaseModel):
    enabled: bool = True


class ReflectionConfig(BaseModel):
    enabled: bool = True
    max_iterations: int = Field(gt=0, default=2)


class MultiHopConfig(BaseModel):
    max_hops: int = Field(gt=0, default=3)


class GenerationConfig(BaseModel):
    hyde: HydeConfig = Field(default_factory=HydeConfig)
    reflection: ReflectionConfig = Field(default_factory=ReflectionConfig)
    multi_hop: MultiHopConfig = Field(default_factory=MultiHopConfig)


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class TracingConfig(BaseModel):
    enabled: bool = False

    @model_validator(mode="after")
    def endpoint_required_when_enabled(self) -> "TracingConfig":
        if self.enabled and not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
            raise ValueError(
                "OTEL_EXPORTER_OTLP_ENDPOINT must be set in .env when tracing.enabled is true. "
                "See .env.example for the expected format."
            )
        return self


# ---------------------------------------------------------------------------
# Models for evaluation config (evaluation phase only — not part of AppConfig)
# ---------------------------------------------------------------------------

class JudgeLLMConfig(BaseModel):
    provider: Literal["openai", "anthropic"] = "openai"
    model: str = "gpt-4o-mini"
    api_key: SecretStr | None = None  # from JUDGE_LLM_API_KEY in .env


class ExtractorConfig(BaseModel):
    backend: Literal["phoenix"] = "phoenix"


class EvaluatorConfig(BaseModel):
    backend: Literal["ragas"] = "ragas"
    judge_llm: JudgeLLMConfig = Field(default_factory=JudgeLLMConfig)


class ResultsConfig(BaseModel):
    phoenix_annotations: bool = False
    results_dir: str = "data/eval_results"


class EvaluationConfig(BaseModel):
    extractor: ExtractorConfig = Field(default_factory=ExtractorConfig)
    evaluator: EvaluatorConfig = Field(default_factory=EvaluatorConfig)
    metrics: list[str] = Field(
        default_factory=lambda: [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]
    )
    datasets: list[str] = Field(default_factory=lambda: ["single", "multi_hop", "comparison"])
    # golden_path=None → reference-required metrics (context_recall, answer_correctness)
    # are skipped automatically; set to a YAML file path to enable them
    golden_path: str | None = None
    results: ResultsConfig = Field(default_factory=ResultsConfig)
    log_level: str = "INFO"

    @model_validator(mode="after")
    def phoenix_db_path_required(self) -> "EvaluationConfig":
        if self.extractor.backend == "phoenix" and not os.environ.get("PHOENIX_DB_PATH"):
            raise ValueError(
                "PHOENIX_DB_PATH must be set in .env when extractor.backend is 'phoenix'. "
                "See .env.example for the expected format."
            )
        return self


# ---------------------------------------------------------------------------
# Model for .env values
# ---------------------------------------------------------------------------

class DatabaseConfig(BaseModel):
    engine: Literal["postgres"]
    host: str
    port: int = Field(gt=0, le=65535)
    name: str
    user: str
    password: SecretStr = Field(min_length=1)  # use .get_secret_value() when connecting
    pool_size: int = Field(gt=0)


class VectorStoreConfig(BaseModel):
    engine: Literal["pgvector"]


# ---------------------------------------------------------------------------
# Root config model
# ---------------------------------------------------------------------------

class AppConfig(BaseModel):
    environment: Literal["dev", "prod", "test"]
    edgar: EdgarConfig
    database: DatabaseConfig
    vector_store: VectorStoreConfig
    embedding: EmbeddingConfig
    chunking: ChunkingConfig
    vector_index: VectorIndexConfig
    retrieval: RetrievalConfig
    llm: LLMConfig
    generation: GenerationConfig
    logging: LoggingConfig
    tracing: TracingConfig = Field(default_factory=TracingConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

# Project root is two levels up from this file (sec_edgar/config/loader.py).
# Validated against pyproject.toml so a miscalculation (e.g. from an unexpected
# install path) fails loudly rather than silently reading the wrong files.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
assert (_PROJECT_ROOT / "pyproject.toml").exists(), (
    f"_PROJECT_ROOT appears miscalculated: {_PROJECT_ROOT} — "
    "expected to find pyproject.toml there."
)


def _require_env(key: str) -> str:
    """Read a required env var, raising a clear error if absent or empty."""
    value = os.environ.get(key)
    if not value or not value.strip():
        raise RuntimeError(
            f"Required environment variable '{key}' is not set or is empty. "
            "Check your .env file against .env.example."
        )
    return value


def _require_env_int(key: str) -> int:
    """Read a required env var as an integer, raising a clear error if absent or non-numeric."""
    value = _require_env(key)
    try:
        return int(value)
    except ValueError:
        raise RuntimeError(
            f"Environment variable '{key}' must be an integer, got: {value!r}"
        )


def ensure_directories(config: AppConfig) -> None:
    """Create application directories that must exist before the app can write files.

    Call once at startup after load_config(). Not called inside the loader itself
    so that config loading stays side-effect-free and test-friendly.
    """
    try:
        config.edgar.raw_data_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise RuntimeError(
            f"Cannot create raw_data_dir at {config.edgar.raw_data_dir}: {e}. "
            "Check the path and write permissions."
        ) from e


def _load() -> AppConfig:
    """Read config.yaml and .env, merge, validate, and return AppConfig."""

    # Load .env into os.environ if the file exists. When env vars are already
    # set (e.g. Docker, CI), .env is skipped and the existing values are used.
    # load_dotenv does NOT override variables already present in the environment.
    # _require_env() below provides the clear error if any required var is missing.
    env_path = _PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    # Load YAML
    yaml_path = _PROJECT_ROOT / "config" / "config.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"config.yaml not found at {yaml_path}.")
    with open(yaml_path) as f:
        yaml_data: dict = yaml.safe_load(f)

    if not isinstance(yaml_data, dict):
        raise ValueError(f"config.yaml at {yaml_path} is empty or not valid YAML.")

    # Validate all required sections are present before accessing them
    required_sections = ["database", "vector_store", "edgar", "embedding", "chunking", "vector_index", "retrieval", "llm", "generation", "logging"]
    missing = [s for s in required_sections if s not in yaml_data]
    if missing:
        raise ValueError(
            f"Missing required section(s) in config.yaml: {', '.join(missing)}. "
            "Check your config against the template."
        )

    # Resolve raw_data_dir to an absolute path.
    # Directory creation is deferred to ensure_directories() — not done here.
    raw_data_dir = (_PROJECT_ROOT / yaml_data["edgar"]["raw_data_dir"]).resolve()

    # Build each section's data dict before constructing the models
    edgar_data = {
        **yaml_data["edgar"],
        "raw_data_dir": raw_data_dir,
        "user_agent": _require_env("EDGAR_USER_AGENT"),
    }

    db_data = {
        "engine":    yaml_data["database"]["engine"],
        "host":      _require_env("DB_HOST"),
        "port":      _require_env_int("DB_PORT"),
        "name":      _require_env("DB_NAME"),
        "user":      _require_env("DB_USER"),
        "password":  _require_env("DB_PASSWORD"),
        "pool_size": _require_env_int("DB_POOL_SIZE"),
    }

    llm_base_url = os.environ.get("LLM_BASE_URL") or None
    llm_api_key = os.environ.get("LLM_API_KEY") or None

    llm_data = {
        **yaml_data["llm"],
        "base_url": llm_base_url,
        "api_key": llm_api_key,
    }

    return AppConfig(
        environment=_require_env("ENVIRONMENT"),
        edgar=EdgarConfig(**edgar_data),
        database=DatabaseConfig(**db_data),
        vector_store=VectorStoreConfig(**yaml_data["vector_store"]),
        embedding=EmbeddingConfig(**yaml_data["embedding"]),
        chunking=ChunkingConfig(**yaml_data["chunking"]),
        vector_index=VectorIndexConfig(**yaml_data["vector_index"]),
        retrieval=RetrievalConfig(**yaml_data["retrieval"]),
        llm=LLMConfig(**llm_data),
        generation=GenerationConfig(**yaml_data["generation"]),
        logging=LoggingConfig(**yaml_data["logging"]),
        tracing=TracingConfig(**yaml_data.get("tracing", {})),
    )


@lru_cache(maxsize=1)
def load_config() -> AppConfig:
    """
    Load and return the application config. Results are cached — files are
    read exactly once per process. Call load_config.cache_clear() in tests
    to force a fresh load.
    """
    return _load()


def _load_eval() -> EvaluationConfig:
    """Read config.yaml and .env, extract the evaluation block, and return EvaluationConfig."""
    env_path = _PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    yaml_path = _PROJECT_ROOT / "config" / "config.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"config.yaml not found at {yaml_path}.")
    with open(yaml_path) as f:
        yaml_data: dict = yaml.safe_load(f)

    if not isinstance(yaml_data, dict):
        raise ValueError(f"config.yaml at {yaml_path} is empty or not valid YAML.")

    eval_yaml = yaml_data.get("evaluation", {})

    # Inject JUDGE_LLM_API_KEY from env into the nested judge_llm config,
    # parallel to how LLM_API_KEY is injected into LLMConfig in _load().
    judge_llm_data = {
        **eval_yaml.get("evaluator", {}).get("judge_llm", {}),
        "api_key": os.environ.get("JUDGE_LLM_API_KEY") or None,
    }
    evaluator_data = {**eval_yaml.get("evaluator", {}), "judge_llm": judge_llm_data}
    eval_final = {**eval_yaml, "evaluator": evaluator_data}

    return EvaluationConfig(**eval_final)


@lru_cache(maxsize=1)
def load_eval_config() -> EvaluationConfig:
    """
    Load and return the evaluation config. Results are cached — files are
    read exactly once per process. Call load_eval_config.cache_clear() in tests
    to force a fresh load.
    """
    return _load_eval()
