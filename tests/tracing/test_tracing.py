"""Tests for the tracing instrumentation: parent-context resolution, the query
root span, and the embedder/retriever spans."""

import json
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from openinference.semconv.trace import (
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

from config.loader import (
    FusionConfig, KeywordSearchConfig, RerankingConfig, RetrievalConfig,
    SectionRetryConfig, VectorSearchConfig,
)
from etl.embedder.base import Embedder
from retrieval.fusion.rrf import RRFFusion
from retrieval.retriever import Retriever
from retrieval.types import MetadataFilter
from tests.test_retrieval import _make_repos
from tracing.context import query_span, resolve_parent_context

SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND


@pytest.fixture(scope="module")
def span_exporter():
    """Capture spans through an in-memory exporter on the global tracer provider.

    Production code resolves its tracer from the global provider, and OTel only
    honours set_tracer_provider once per process, so the exporter is attached to
    whatever SDK provider is (or becomes) global and shared across this module.
    """
    exporter = InMemorySpanExporter()
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter


@pytest.fixture(autouse=True)
def _clear_spans(span_exporter):
    span_exporter.clear()
    yield


def _spans_named(exporter, name):
    return [s for s in exporter.get_finished_spans() if s.name == name]


# ── resolve_parent_context ────────────────────────────────────────────────────

def test_resolve_parent_context_returns_none_outside_langchain_run():
    assert resolve_parent_context() is None


def test_resolve_parent_context_swallows_get_current_span_error():
    # openinference's get_current_span raises when no LangChain runtime is loaded;
    # the helper must fall back to the ambient context instead of propagating.
    with patch("tracing.context.get_current_span", side_effect=AttributeError("no runtime")):
        assert resolve_parent_context() is None


# ── query root span ───────────────────────────────────────────────────────────

def test_query_span_emits_chain_root_with_query_attributes(span_exporter):
    with query_span("What was total revenue?", pipeline_mode="classic"):
        pass

    spans = _spans_named(span_exporter, "query")
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs[SPAN_KIND] == OpenInferenceSpanKindValues.CHAIN.value
    assert attrs[SpanAttributes.INPUT_VALUE] == "What was total revenue?"
    assert attrs["pipeline.mode"] == "classic"


# ── embedder span ─────────────────────────────────────────────────────────────

class _FakeEmbedder(Embedder):
    @property
    def dimension(self) -> int:
        return 3

    @property
    def model_name(self) -> str:
        return "fake-embed-model"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0, 0.0] for _ in texts]


def test_embedder_emits_embedding_span(span_exporter):
    _FakeEmbedder().embed(["alpha", "beta"])

    spans = _spans_named(span_exporter, "embedder.embed")
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs[SPAN_KIND] == OpenInferenceSpanKindValues.EMBEDDING.value
    assert attrs[SpanAttributes.EMBEDDING_MODEL_NAME] == "fake-embed-model"
    assert attrs["embedding.text_count"] == 2


# ── retriever spans ───────────────────────────────────────────────────────────

def _retriever(vector, keyword, reranker, repos, *, min_top_score: float = 3.0) -> Retriever:
    config = RetrievalConfig(
        vector_search=VectorSearchConfig(enabled=True, oversample_k=5, similarity_threshold=0.0),
        keyword_search=KeywordSearchConfig(enabled=True, top_k=5),
        fusion=FusionConfig(top_k=5),
        reranking=RerankingConfig(enabled=True, top_k=5),
        section_retry=SectionRetryConfig(enabled=True, min_top_score=min_top_score),
        final_top_k=5,
    )
    chunks_repo, parents_repo, filings_repo = repos
    return Retriever(
        config=config, fusion=RRFFusion(),
        chunks_repo=chunks_repo, parent_chunks_repo=parents_repo, filings_repo=filings_repo,
        vector_retriever=vector, keyword_retriever=keyword, reranker=reranker,
    )


def test_retrieval_search_nests_vector_keyword_rerank_children(span_exporter):
    chunk_id, parent_id, filing_id = uuid4(), uuid4(), uuid4()
    repos = _make_repos(chunk_id, parent_id, filing_id)
    vector = MagicMock()
    vector.search.return_value = [(chunk_id, 0.9)]
    keyword = MagicMock()
    keyword.search.return_value = [(chunk_id, 0.85)]
    reranker = MagicMock()
    reranker.rerank.return_value = [(chunk_id, 5.0)]

    retriever = _retriever(vector, keyword, reranker, repos)
    retriever.retrieve(keyword_query="gpu", semantic_embedding=[0.1] * 8)

    search = _spans_named(span_exporter, "retrieval.search")
    assert len(search) == 1
    search_span = search[0]
    assert search_span.attributes[SPAN_KIND] == OpenInferenceSpanKindValues.RETRIEVER.value
    assert search_span.attributes["retrieval.section_retry"] is False
    # retrieval.search keeps only summary attributes — the candidate lists (as span
    # output) and the fused count live on the per-stage child spans, never here.
    assert SpanAttributes.OUTPUT_VALUE not in search_span.attributes
    assert "retrieval.fused_count" not in search_span.attributes

    search_id = search_span.context.span_id
    # None of the stage children carry an OpenInference span kind — in particular
    # retrieval.rerank stays generic so Phoenix renders its output.value rather than
    # the empty reranker-document template.
    for child_name in (
        "retrieval.vector_search",
        "retrieval.keyword_search",
        "retrieval.fusion",
        "retrieval.rerank",
    ):
        children = _spans_named(span_exporter, child_name)
        assert len(children) == 1, child_name
        child = children[0]
        assert child.parent is not None and child.parent.span_id == search_id
        assert SPAN_KIND not in child.attributes
        # each stage records its own ordered ranked list as JSON span output
        assert child.attributes[SpanAttributes.OUTPUT_MIME_TYPE] == OpenInferenceMimeTypeValues.JSON.value
        candidates = json.loads(child.attributes[SpanAttributes.OUTPUT_VALUE])
        assert candidates[0][0] == str(chunk_id)

    def _top_score(name: str) -> float:
        span = _spans_named(span_exporter, name)[0]
        return json.loads(span.attributes[SpanAttributes.OUTPUT_VALUE])[0][1]

    assert _top_score("retrieval.vector_search") == 0.9
    assert _top_score("retrieval.keyword_search") == 0.85
    assert _top_score("retrieval.rerank") == 5.0
    assert _spans_named(span_exporter, "retrieval.fusion")[0].attributes["retrieval.fused_count"] == 1


def test_section_retry_flag_distinguishes_primary_and_widened_search(span_exporter):
    chunk_id, parent_id, filing_id = uuid4(), uuid4(), uuid4()
    repos = _make_repos(chunk_id, parent_id, filing_id)
    vector = MagicMock()
    vector.search.return_value = [(chunk_id, 0.9)]
    keyword = MagicMock()
    keyword.search.return_value = [(chunk_id, 0.85)]
    # Primary search reranks below the floor (3.0) so retrieve() widens the
    # section filter; the widened search clears the floor.
    reranker = MagicMock()
    reranker.rerank.side_effect = [[(chunk_id, 1.0)], [(chunk_id, 5.0)]]

    retriever = _retriever(vector, keyword, reranker, repos, min_top_score=3.0)
    retriever.retrieve(
        keyword_query="gpu", semantic_embedding=[0.1] * 8,
        filters=MetadataFilter(section="Item 1"),
    )

    search = _spans_named(span_exporter, "retrieval.search")
    assert len(search) == 2
    flags = sorted(s.attributes["retrieval.section_retry"] for s in search)
    assert flags == [False, True]
