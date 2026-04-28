"""Vector (dense) retrieval backends backed by pgvector."""
from retrieval.vector.base import BaseVectorRetriever
from retrieval.vector.pgvector import PgvectorRetriever

__all__ = ["BaseVectorRetriever", "PgvectorRetriever"]
