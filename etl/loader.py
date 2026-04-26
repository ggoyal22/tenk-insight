import logging
from datetime import datetime, timezone
from uuid import UUID, uuid4

import psycopg2.errors

from db.client.base import DatabaseClient
from db.models import ChunkRecord, FilingRecord, ParentChunkRecord
from db.repositories.chunks import ChunksRepo
from db.repositories.filings import FilingsRepo
from db.repositories.parent_chunks import ParentChunksRepo
from db.vector.base import VectorStore
from etl.types import ChildChunk, ParentChunk

logger = logging.getLogger(__name__)


class Loader:
    def __init__(
        self,
        db_client: DatabaseClient,
        filings_repo: FilingsRepo,
        parent_chunks_repo: ParentChunksRepo,
        chunks_repo: ChunksRepo,
        vector_store: VectorStore,
    ) -> None:
        self._db_client = db_client
        self._filings_repo = filings_repo
        self._parent_chunks_repo = parent_chunks_repo
        self._chunks_repo = chunks_repo
        self._vector_store = vector_store

    def load(
        self,
        filing: FilingRecord,
        parents: list[ParentChunk],
        children: list[ChildChunk],
    ) -> UUID:
        # Dedup: check for existing filing and whether embeddings are complete
        existing = self._filings_repo.get_by_accession_number(filing.accession_number)
        if existing:
            existing_chunks = self._chunks_repo.get_by_filing_id(existing.id)
            if existing_chunks:
                missing = [c for c in existing_chunks if c.embedded_at is None]
                if not missing:
                    logger.info(
                        "Filing %s already fully ingested (%d chunks) — skipping",
                        filing.accession_number, len(existing_chunks),
                    )
                    return existing.id
                logger.warning(
                    "Filing %s has %d/%d chunks missing embeddings — re-embedding",
                    filing.accession_number, len(missing), len(existing_chunks),
                )
                return self._reembed_missing(
                    existing.id, filing.accession_number, missing, children
                )
            # Filing exists with 0 chunks (orphaned from a prior crash before step 3
            # committed) — fall through to full ingestion; the INSERT will fail the
            # UNIQUE constraint, so delete the orphan row first.
            logger.warning(
                "Filing %s exists with 0 chunks — deleting orphan and re-ingesting",
                filing.accession_number,
            )
            self._filings_repo.delete(existing.id)

        # Steps 1–3: insert filing + parent chunks + child chunk metadata atomically.
        # A crash at any point rolls back the entire transaction — the DB is left clean.
        chunk_uuid_map: dict[int, UUID] = {}
        try:
            with self._db_client.connection() as conn:
                # Step 1 — filing row
                filing_uuid = self._filings_repo.insert(filing, conn=conn)
                logger.info("Inserted filing %s → %s", filing.accession_number, filing_uuid)

                # Step 2 — parent chunks; collect UUIDs to wire child FK in step 3
                now = datetime.now(timezone.utc)
                parent_uuid_map: dict[int, UUID] = {}
                for parent in parents:
                    record = ParentChunkRecord(
                        id=uuid4(),
                        filing_id=filing_uuid,
                        chunk_index=parent.filing_chunk_index,
                        section=parent.section_name,
                        text=parent.text,
                        token_count=parent.token_count,
                        content_hash=parent.content_hash,
                        created_at=now,
                    )
                    parent_uuid_map[parent.filing_chunk_index] = (
                        self._parent_chunks_repo.insert(record, conn=conn)
                    )
                logger.info("Inserted %d parent chunks", len(parents))

                # Step 3 — child chunk metadata (embedding written separately in step 5)
                child_records = [
                    ChunkRecord(
                        id=uuid4(),
                        filing_id=filing_uuid,
                        chunk_index=child.filing_chunk_index,
                        section=child.section_name,
                        chunk_type=child.content_type,
                        text=child.text,
                        token_count=child.token_count,
                        content_hash=child.content_hash,
                        created_at=now,
                        parent_chunk_id=parent_uuid_map[child.parent_chunk_index],
                    )
                    for child in children
                ]
                self._chunks_repo.insert_many(child_records, conn=conn)
                logger.info("Inserted %d child chunks", len(children))

                # Step 4 — fetch back DB-assigned UUIDs needed for vector upsert
                inserted = self._chunks_repo.get_by_filing_id(filing_uuid, conn=conn)
                chunk_uuid_map = {c.chunk_index: c.id for c in inserted}

                conn.commit()

        except psycopg2.errors.UniqueViolation:
            # A concurrent worker inserted the same filing between our dedup check
            # and our INSERT — treat it as already handled.
            logger.warning(
                "Filing %s was concurrently inserted by another worker — skipping",
                filing.accession_number,
            )
            existing = self._filings_repo.get_by_accession_number(filing.accession_number)
            assert existing is not None
            return existing.id

        # Step 5 — upsert embeddings (outside transaction: VectorStore is DB-agnostic
        # and embedding NULL chunks are detectable and re-embeddable on the next run)
        upserted = 0
        for child in children:
            if child.embedding is None:
                continue
            self._vector_store.upsert(
                chunk_uuid_map[child.filing_chunk_index],
                child.embedding,
                {"embedding_model": child.embedding_model},
            )
            upserted += 1
        logger.info("Upserted %d embeddings", upserted)

        return filing_uuid

    def _reembed_missing(
        self,
        filing_id: UUID,
        accession_number: str,
        missing_chunks: list[ChunkRecord],
        children: list[ChildChunk],
    ) -> UUID:
        missing_index = {c.chunk_index: c.id for c in missing_chunks}
        upserted = 0
        for child in children:
            chunk_id = missing_index.get(child.filing_chunk_index)
            if chunk_id is None or child.embedding is None:
                continue
            self._vector_store.upsert(
                chunk_id,
                child.embedding,
                {"embedding_model": child.embedding_model},
            )
            upserted += 1
        logger.info(
            "Re-embedded %d/%d missing chunks for %s",
            upserted, len(missing_chunks), accession_number,
        )
        return filing_id
