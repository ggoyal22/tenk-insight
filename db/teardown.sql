-- =============================================================================
-- SEC EDGAR RAG System — Teardown Script
-- =============================================================================
-- Drops all objects created by schema.template.sql, in reverse dependency
-- order (children before parents to respect FK constraints).
--
-- WARNING: This is destructive. All data will be lost.
-- Intended for local development resets and CI teardown only.
--
-- Wrapped in a transaction — if any statement fails, the entire teardown
-- is rolled back, leaving the database in a consistent state.
-- =============================================================================

BEGIN;

-- Triggers
DROP TRIGGER  IF EXISTS chunks_set_updated_at ON chunks;
DROP TRIGGER  IF EXISTS parent_chunks_set_updated_at ON parent_chunks;
DROP TRIGGER  IF EXISTS filings_set_updated_at ON filings;
DROP FUNCTION IF EXISTS set_updated_at();

-- chunks indexes and table
DROP INDEX  IF EXISTS idx_chunks_embedding;
DROP INDEX  IF EXISTS idx_chunks_embedding_model;
DROP INDEX  IF EXISTS idx_chunks_chunk_type;
DROP INDEX  IF EXISTS idx_chunks_section;
DROP INDEX  IF EXISTS idx_chunks_parent_chunk_id;
DROP INDEX  IF EXISTS idx_chunks_filing_id;
DROP TABLE  IF EXISTS chunks;

-- parent_chunks indexes and table
DROP INDEX  IF EXISTS idx_parent_chunks_section;
DROP INDEX  IF EXISTS idx_parent_chunks_filing_id;
DROP TABLE  IF EXISTS parent_chunks;

-- filings indexes and table
DROP INDEX  IF EXISTS idx_filings_fiscal_year_end;
DROP INDEX  IF EXISTS idx_filings_filing_date;
DROP INDEX  IF EXISTS idx_filings_form_type;
DROP INDEX  IF EXISTS idx_filings_cik;
DROP INDEX  IF EXISTS idx_filings_ticker;
DROP TABLE  IF EXISTS filings;

-- Extension — CASCADE handles any remaining dependent objects.
-- WARNING: drops the vector extension instance-wide. Do not run on a shared
-- PostgreSQL server where other databases also use pgvector.
DROP EXTENSION IF EXISTS vector CASCADE;

COMMIT;
