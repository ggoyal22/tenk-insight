-- =============================================================================
-- SEC EDGAR RAG System — Database Schema Template
-- =============================================================================
-- TEMPLATE FILE: do not run this file directly against psql.
-- Run via:  python db/setup.py
--
-- The setup script reads config/config.yaml and substitutes the placeholders
-- below before executing the DDL:
--
--   {embedding_dimension}    ← config.embedding.dimension
--   {hnsw_ops_class}         ← derived from config.vector_index.distance_function
--                                cosine → vector_cosine_ops
--                                l2     → vector_l2_ops
--                                dot    → vector_ip_ops
--   {hnsw_m}                 ← config.vector_index.hnsw_m
--   {hnsw_ef_construction}   ← config.vector_index.hnsw_ef_construction
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS vector;


-- ---------------------------------------------------------------------------
-- Shared trigger function — auto-sets updated_at on any UPDATE
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;


-- ---------------------------------------------------------------------------
-- filings
-- One row per SEC filing document (10-K, 10-Q, 8-K, …).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS filings (
    id               UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    ticker           VARCHAR     NOT NULL,                                                          -- e.g. "NVDA"
    company_name     VARCHAR     NOT NULL,                                                          -- e.g. "NVIDIA Corporation"
    cik              VARCHAR     NOT NULL CHECK (cik ~ '^\d{1,10}$'),                              -- EDGAR Central Index Key (1–10 digit numeric string)
    accession_number VARCHAR     NOT NULL UNIQUE CHECK (accession_number ~ '^\d{10}-\d{2}-\d{6}$'), -- EDGAR canonical ID, e.g. "0001045810-23-000017"
    form_type        VARCHAR     NOT NULL,                                                          -- e.g. "10-K", "10-Q", "8-K"
    filing_date      DATE        NOT NULL,                                                          -- date the filing was submitted to SEC
    fiscal_year_end  DATE,                                                                          -- end of the fiscal year covered by this filing
    sic_code         VARCHAR,                                                                       -- Standard Industrial Classification code
    source_url       TEXT        NOT NULL,                                                          -- full EDGAR URL to the filing document
    downloaded_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),                                            -- when raw filing was fetched from EDGAR
    updated_at       TIMESTAMPTZ                                                                    -- auto-set by trigger on every UPDATE
);

CREATE INDEX IF NOT EXISTS idx_filings_ticker          ON filings (ticker);
CREATE INDEX IF NOT EXISTS idx_filings_cik             ON filings (cik);
CREATE INDEX IF NOT EXISTS idx_filings_form_type       ON filings (form_type);
CREATE INDEX IF NOT EXISTS idx_filings_filing_date     ON filings (filing_date);
CREATE INDEX IF NOT EXISTS idx_filings_fiscal_year_end ON filings (fiscal_year_end);

CREATE OR REPLACE TRIGGER filings_set_updated_at
    BEFORE UPDATE ON filings
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();


-- ---------------------------------------------------------------------------
-- parent_chunks
-- One row per large context chunk (~1024 tokens).
-- Parent chunks are NOT embedded — they exist solely to provide richer context
-- to the LLM after a child chunk is retrieved via vector search.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS parent_chunks (
    id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    filing_id    UUID        NOT NULL REFERENCES filings (id) ON DELETE CASCADE,
    chunk_index  INT         NOT NULL CHECK (chunk_index >= 0),            -- sequential position within the filing (0-based)
    section      VARCHAR     NOT NULL,                                      -- filing section, e.g. "Item 1A Risk Factors"
    text         TEXT        NOT NULL,                                      -- full text of the parent chunk
    token_count  INT         NOT NULL CHECK (token_count > 0),             -- token count (for LLM context window management)
    content_hash VARCHAR(64) NOT NULL CHECK (content_hash ~ '^[0-9a-f]{64}$'), -- SHA-256 hex digest of text; detects changes on re-ingestion
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ,                                              -- auto-set by trigger on every UPDATE

    UNIQUE (filing_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_parent_chunks_filing_id ON parent_chunks (filing_id);
CREATE INDEX IF NOT EXISTS idx_parent_chunks_section   ON parent_chunks (filing_id, section);

CREATE OR REPLACE TRIGGER parent_chunks_set_updated_at
    BEFORE UPDATE ON parent_chunks
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();


-- ---------------------------------------------------------------------------
-- chunks
-- One row per small retrieval chunk (~256 tokens).
-- These are the units that get embedded and searched via vector similarity.
-- Each chunk optionally belongs to a parent_chunk for context expansion.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
    id               UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    filing_id        UUID        NOT NULL REFERENCES filings (id) ON DELETE CASCADE,
    parent_chunk_id  UUID        REFERENCES parent_chunks (id) ON DELETE SET NULL,
    -- chunk_index is filing-scoped (not parent-scoped): sequential across all chunks in the filing.
    -- Adjacent-chunk retrieval must filter by parent_chunk_id, not just by chunk_index ± 1.
    chunk_index      INT         NOT NULL CHECK (chunk_index >= 0),        -- sequential position within the filing (0-based)
    section          VARCHAR     NOT NULL,                                  -- filing section, e.g. "Item 7 MD&A"
    chunk_type       VARCHAR     NOT NULL CHECK (chunk_type IN ('narrative', 'table', 'list')), -- content type
    text             TEXT        NOT NULL,                                  -- text of this chunk
    token_count      INT         NOT NULL CHECK (token_count > 0),         -- token count of this chunk's text
    -- page_number is nullable; CHECK (page_number > 0) only fires on non-NULL values (correct behaviour)
    page_number      INT         CHECK (page_number > 0),                  -- source page in the original filing (nullable if unavailable)
    embedding        VECTOR({embedding_dimension}),                        -- dense vector; dimension from config.embedding.dimension
    embedding_model  VARCHAR,                                              -- model used to produce the embedding, e.g. "BAAI/bge-large-en-v1.5"
    content_hash     VARCHAR(64) NOT NULL CHECK (content_hash ~ '^[0-9a-f]{64}$'), -- SHA-256 hex digest of text; prevents redundant re-embedding
    embedded_at      TIMESTAMPTZ,                                          -- null until embedded
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ,                                          -- auto-set by trigger on every UPDATE

    -- embedding and embedding_model must both be set or both be null
    CHECK ((embedding IS NULL) = (embedding_model IS NULL)),

    UNIQUE (filing_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_filing_id        ON chunks (filing_id);
CREATE INDEX IF NOT EXISTS idx_chunks_parent_chunk_id  ON chunks (parent_chunk_id);
CREATE INDEX IF NOT EXISTS idx_chunks_section          ON chunks (filing_id, section);
CREATE INDEX IF NOT EXISTS idx_chunks_chunk_type       ON chunks (chunk_type);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_model  ON chunks (embedding_model);

CREATE OR REPLACE TRIGGER chunks_set_updated_at
    BEFORE UPDATE ON chunks
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- Vector similarity index (HNSW).
-- Ops class and parameters are substituted from config by the setup script.
-- Build this index after bulk data load for best performance.
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON chunks USING hnsw (embedding {hnsw_ops_class})
    WITH (m = {hnsw_m}, ef_construction = {hnsw_ef_construction});
