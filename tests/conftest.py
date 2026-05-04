"""
Shared test fixtures and constants for the SEC EDGAR RAG test suite.
"""

import os
from pathlib import Path

import psycopg2
import pytest
from dotenv import load_dotenv

from config.loader import DatabaseConfig
from db.client.postgres import PostgresClient
from db.setup import _substitute, _SCHEMA_TEMPLATE

# ---------------------------------------------------------------------------
# Valid field values for constructing Pydantic config models in unit tests
# ---------------------------------------------------------------------------

VALID_EDGAR = {
    "tickers": ["NVDA"],
    "form_types": ["10-K"],
    "years": [2024],
    "raw_data_dir": "/tmp/data/raw",
    "user_agent": "Test Suite test@example.com",
}

VALID_EMBEDDING = {
    "model": "BAAI/bge-large-en-v1.5",
    "dimension": 1024,
    "batch_size": 64,
    "device": "cpu",
}

VALID_CHUNKING = {
    "child_chunk_size": 256,
    "child_chunk_overlap": 32,
    "parent_chunk_size": 1024,
    "parent_chunk_overlap": 64,
}

VALID_VECTOR_INDEX = {
    "type": "hnsw",
    "distance_function": "cosine",
    "hnsw_m": 16,
    "hnsw_ef_construction": 64,
}

VALID_RETRIEVAL = {
    "metadata_filtering": {"enabled": True},
    "vector_search": {
        "enabled": True,
        "quantization": "none",           # "none" avoids halfvec cast in test DB queries
        "oversample_k": 5,
        "similarity_threshold": 0.0,      # low threshold so test embeddings always match
    },
    "keyword_search": {
        "enabled": True,
        "implementation": "fts",
        "top_k": 5,
        "fts": {"query_mode": "web"},
        "bm25": {"k1": 1.5, "b": 0.75},
    },
    "fusion": {"implementation": "rrf", "rrf_k": 60, "top_k": 5},
    "reranking": {
        "enabled": False,                 # disabled in tests — avoids loading cross-encoder model
        "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "top_k": 5,
    },
    "final_top_k": 5,
}

VALID_LLM = {
    "provider": "ollama",
    "model": "llama3.1:8b",
    "temperature": 0.0,
    "max_tokens": 2048,
    "timeout": 120,
    "base_url": "http://localhost:11434",
    "api_key": None,
}

VALID_GENERATION = {
    "hyde": {"enabled": True},
    "reflection": {"enabled": True, "max_iterations": 2},
    "multi_hop": {"max_hops": 3},
}

VALID_LOGGING = {
    "level": "INFO",
}

VALID_DATABASE = {
    "engine": "postgres",
    "host": "localhost",
    "port": 5432,
    "name": "sec_edgar",
    "user": "postgres",
    "password": "test-password-not-real",
    "pool_size": 5,
}

VALID_VECTOR_STORE = {
    "engine": "pgvector",
}

VALID_EVALUATION = {
    "extractor": {"backend": "phoenix"},
    "evaluator": {
        "backend": "ragas",
        "judge_llm": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": None,
        },
    },
    "metrics": ["faithfulness", "answer_relevancy"],
    "datasets": ["single"],
    "golden_path": None,
    "results": {
        "phoenix_annotations": False,
        "results_dir": "/tmp/eval_results",
    },
}

# ---------------------------------------------------------------------------
# Integration test fixtures
# ---------------------------------------------------------------------------

def _load_test_db_config() -> DatabaseConfig:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    test_db_name = os.environ.get("TEST_DB_NAME", "").strip()
    if not test_db_name:
        pytest.skip("TEST_DB_NAME not set in .env — skipping integration test")

    return DatabaseConfig(
        engine="postgres",
        host=os.environ.get("DB_HOST", "localhost"),
        port=int(os.environ.get("DB_PORT", 5432)),
        name=test_db_name,
        user=os.environ.get("DB_USER", "postgres"),
        password=os.environ.get("DB_PASSWORD", ""),
        pool_size=2,
    )


@pytest.fixture(scope="session")
def db_client():
    config = _load_test_db_config()

    # create test DB if it doesn't exist
    conn = psycopg2.connect(
        host=config.host,
        port=config.port,
        dbname="postgres",
        user=config.user,
        password=config.password.get_secret_value(),
    )
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f'DROP DATABASE IF EXISTS "{config.name}"')
        cur.execute(f'CREATE DATABASE "{config.name}"')
    conn.close()

    # apply schema to test DB
    from config.loader import AppConfig, ChunkingConfig, EdgarConfig, EmbeddingConfig, GenerationConfig, LLMConfig, LoggingConfig, RetrievalConfig, VectorIndexConfig, VectorStoreConfig
    app_config = AppConfig(
        environment="test",
        edgar=EdgarConfig(**VALID_EDGAR),
        database=config,
        vector_store=VectorStoreConfig(**VALID_VECTOR_STORE),
        embedding=EmbeddingConfig(**VALID_EMBEDDING),
        chunking=ChunkingConfig(**VALID_CHUNKING),
        vector_index=VectorIndexConfig(**VALID_VECTOR_INDEX),
        retrieval=RetrievalConfig(**VALID_RETRIEVAL),
        llm=LLMConfig(**VALID_LLM),
        generation=GenerationConfig(**VALID_GENERATION),
        logging=LoggingConfig(**VALID_LOGGING),
    )
    sql = _substitute(_SCHEMA_TEMPLATE.read_text(), app_config)
    schema_conn = psycopg2.connect(
        host=config.host, port=config.port, dbname=config.name,
        user=config.user, password=config.password.get_secret_value(),
    )
    with schema_conn.cursor() as cur:
        cur.execute(sql)
    schema_conn.commit()
    schema_conn.close()

    client = PostgresClient(config)
    yield client
    client.close()


@pytest.fixture(autouse=False)
def truncate_tables(db_client):
    with db_client.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE filings, parent_chunks, chunks CASCADE")
        conn.commit()
