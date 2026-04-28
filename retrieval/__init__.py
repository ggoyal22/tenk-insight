"""Hybrid retrieval pipeline for SEC EDGAR RAG — combines vector and keyword search."""
from retrieval.factory import build_retriever_from_config
from retrieval.retriever import Retriever
from retrieval.types import MetadataFilter, RetrievalResult

__all__ = ["build_retriever_from_config", "Retriever", "MetadataFilter", "RetrievalResult"]
