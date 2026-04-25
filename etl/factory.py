from config.loader import AppConfig
from db.client.base import DatabaseClient
from db.repositories.postgres.chunks import ChunksRepository
from db.repositories.postgres.filings import FilingsRepository
from db.repositories.postgres.parent_chunks import ParentChunksRepository
from db.vector.base import VectorStore
from etl.chunker.base import Chunker
from etl.chunker.recursive import RecursiveChunker
from etl.downloader.base import Downloader
from etl.downloader.edgartools import EdgarToolsDownloader
from etl.embedder.base import Embedder
from etl.embedder.sentence_transformer import SentenceTransformerEmbedder
from etl.loader import Loader
from etl.parser.base import Parser
from etl.parser.edgartools import EdgarToolsParser


def create_downloader(config: AppConfig) -> Downloader:
    return EdgarToolsDownloader(config)


def create_parser() -> Parser:
    return EdgarToolsParser()


def create_chunker(config: AppConfig) -> Chunker:
    return RecursiveChunker(config.chunking)


def create_embedder(config: AppConfig) -> Embedder:
    return SentenceTransformerEmbedder(config.embedding, config.chunking.child_chunk_size)


def create_loader(
    db_client: DatabaseClient,
    filings_repo: FilingsRepository,
    parent_chunks_repo: ParentChunksRepository,
    chunks_repo: ChunksRepository,
    vector_store: VectorStore,
) -> Loader:
    return Loader(db_client, filings_repo, parent_chunks_repo, chunks_repo, vector_store)
