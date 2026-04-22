"""
Shared test fixtures and constants for the SEC EDGAR RAG test suite.
"""

# Valid field values for constructing Pydantic config models directly in tests.
# Each dict matches the fields of the corresponding model class.

VALID_EDGAR = {
    "tickers": ["NVDA"],
    "form_types": ["10-K"],
    "years": [2024],
    "rate_limit_per_second": 8,
    "raw_data_dir": "/tmp/data/raw",
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
    "top_k": 5,
    "similarity_threshold": 0.7,
}

VALID_LOGGING = {
    "level": "INFO",
}

VALID_DATABASE = {
    "host": "localhost",
    "port": 5432,
    "name": "sec_edgar",
    "user": "postgres",
    "password": "test-password-not-real",
    "pool_size": 5,
}
