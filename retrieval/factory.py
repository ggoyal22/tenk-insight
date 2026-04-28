import logging

from config.loader import AppConfig
from db.client.base import DatabaseClient
from db.repositories.chunks import ChunksRepo
from db.repositories.filings import FilingsRepo
from db.repositories.parent_chunks import ParentChunksRepo
from db.vector.base import VectorStore
from retrieval.fusion.base import BaseFusion
from retrieval.fusion.rrf import RRFFusion
from retrieval.keyword.base import BaseKeywordRetriever
from retrieval.keyword.postgres_fts import PostgresFTSRetriever
from retrieval.reranker.base import BaseReranker
from retrieval.reranker.cross_encoder import CrossEncoderReranker
from retrieval.retriever import Retriever
from retrieval.vector.base import BaseVectorRetriever
from retrieval.vector.pgvector import PgvectorRetriever

logger = logging.getLogger(__name__)

# Registry: (database.engine, keyword_search.implementation) → retriever class
_KEYWORD_RETRIEVERS: dict[tuple[str, str], type[BaseKeywordRetriever]] = {
    ("postgres", "fts"): PostgresFTSRetriever,
}

_FUSION_STRATEGIES: dict[str, type[BaseFusion]] = {
    "rrf": RRFFusion,
}

_RERANKERS: dict[str, type[BaseReranker]] = {
    "cross_encoder": CrossEncoderReranker,
}

# Registry: vector_store.engine → retriever class
_VECTOR_RETRIEVERS: dict[str, type[BaseVectorRetriever]] = {
    "pgvector": PgvectorRetriever,
}


def _check_search_vector_column(db_client: DatabaseClient) -> None:
    """Raise if the search_vector column is absent — indicates migration 001 not applied."""
    with db_client.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'chunks' AND column_name = 'search_vector'
                )
            """)
            exists = cur.fetchone()[0]
    if not exists:
        raise RuntimeError(
            "Keyword search is enabled but the 'search_vector' column does not exist on "
            "the 'chunks' table. Run the retrieval migration first:\n"
            "    python db/migrations/001_retrieval.py"
        )


def build_retriever_from_config(config: AppConfig, db_client: DatabaseClient) -> Retriever:
    """Convenience entry point — assembles the full retrieval stack from config and a DB client."""
    from db.factory import create_chunks_repo, create_filings_repo, create_parent_chunks_repo, create_vector_store
    return build_retriever(
        config=config,
        db_client=db_client,
        vector_store=create_vector_store(config, db_client),
        chunks_repo=create_chunks_repo(db_client),
        parent_chunks_repo=create_parent_chunks_repo(db_client),
        filings_repo=create_filings_repo(db_client),
    )


def build_retriever(
    config: AppConfig,
    db_client: DatabaseClient,
    vector_store: VectorStore,
    chunks_repo: ChunksRepo,
    parent_chunks_repo: ParentChunksRepo,
    filings_repo: FilingsRepo,
) -> Retriever:
    rc = config.retrieval

    vector_retriever: BaseVectorRetriever | None = None
    if rc.vector_search.enabled:
        engine = config.vector_store.engine
        cls = _VECTOR_RETRIEVERS.get(engine)
        if cls is None:
            raise ValueError(f"No vector retriever for vector_store.engine={engine!r}")
        vector_retriever = cls(vector_store, rc.vector_search.oversample_k)

    keyword_retriever: BaseKeywordRetriever | None = None
    if rc.keyword_search.enabled:
        key = (config.database.engine, rc.keyword_search.implementation)
        cls = _KEYWORD_RETRIEVERS.get(key)
        if cls is None:
            raise ValueError(
                f"No keyword retriever for database.engine={config.database.engine!r}, "
                f"implementation={rc.keyword_search.implementation!r}"
            )
        _check_search_vector_column(db_client)

        if rc.keyword_search.implementation == "fts":
            keyword_retriever = cls(chunks_repo, rc.keyword_search.fts.query_mode)  # type: ignore[call-arg]
        else:
            keyword_retriever = cls(chunks_repo)

    fusion_cls = _FUSION_STRATEGIES.get(rc.fusion.implementation)
    if fusion_cls is None:
        raise ValueError(f"Unknown fusion implementation: {rc.fusion.implementation!r}")
    fusion = fusion_cls(k=rc.fusion.rrf_k)

    reranker: BaseReranker | None = None
    if rc.reranking.enabled:
        reranker = CrossEncoderReranker(rc.reranking.model)

    return Retriever(
        config=rc,
        fusion=fusion,
        chunks_repo=chunks_repo,
        parent_chunks_repo=parent_chunks_repo,
        filings_repo=filings_repo,
        vector_retriever=vector_retriever,
        keyword_retriever=keyword_retriever,
        reranker=reranker,
    )
