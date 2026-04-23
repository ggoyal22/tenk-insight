from config.loader import AppConfig
from db.client.base import DatabaseClient
from db.client.postgres import PostgresClient
from db.repositories.postgres.chunks import ChunksRepository
from db.repositories.postgres.filings import FilingsRepository
from db.repositories.postgres.parent_chunks import ParentChunksRepository
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


def create_filings_repo(client: DatabaseClient) -> FilingsRepository:
    return FilingsRepository(client)


def create_parent_chunks_repo(client: DatabaseClient) -> ParentChunksRepository:
    return ParentChunksRepository(client)


def create_chunks_repo(client: DatabaseClient) -> ChunksRepository:
    return ChunksRepository(client)
