from config.loader import AppConfig
from db.client.base import DatabaseClient
from db.client.postgres import PostgresClient
from db.repositories.chunks import ChunksRepo
from db.repositories.filings import FilingsRepo
from db.repositories.parent_chunks import ParentChunksRepo
from db.repositories.postgres.chunks import PostgresChunksRepository
from db.repositories.postgres.filings import PostgresFilingsRepository
from db.repositories.postgres.parent_chunks import PostgresParentChunksRepository
from db.vector.base import VectorStore
from db.vector.pgvector import PgvectorStore


def create_db_client(config: AppConfig) -> DatabaseClient:
    engine = config.database.engine
    if engine == "postgres":
        return PostgresClient(config.database)
    raise ValueError(f"Unsupported database engine: {engine!r}")


def create_vector_store(config: AppConfig, client: DatabaseClient) -> VectorStore:
    engine = config.vector_store.engine
    if engine == "pgvector":
        return PgvectorStore(
            client=client,
            similarity_threshold=config.retrieval.similarity_threshold,
            distance_function=config.vector_index.distance_function,
        )
    raise ValueError(f"Unsupported vector store engine: {engine!r}")


def create_filings_repo(client: DatabaseClient) -> FilingsRepo:
    return PostgresFilingsRepository(client)


def create_parent_chunks_repo(client: DatabaseClient) -> ParentChunksRepo:
    return PostgresParentChunksRepository(client)


def create_chunks_repo(client: DatabaseClient) -> ChunksRepo:
    return PostgresChunksRepository(client)
