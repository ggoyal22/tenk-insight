from abc import ABC, abstractmethod
from functools import wraps

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes

from tracing.context import resolve_parent_context


def _trace_embed(fn):
    @wraps(fn)
    def wrapper(self, texts: list[str], *args, **kwargs):
        tracer = trace.get_tracer(__name__)
        # embed runs inside the embed_queries node during a graph run, whose
        # instrumentor span isn't in the OTel ambient context; resolve it so this
        # span nests under the node. Returns None during ingestion (no graph run),
        # leaving a clean no-op under the no-op tracer.
        parent = resolve_parent_context()
        with tracer.start_as_current_span("embedder.embed", context=parent) as span:
            span.set_attribute(
                SpanAttributes.OPENINFERENCE_SPAN_KIND,
                OpenInferenceSpanKindValues.EMBEDDING.value,
            )
            span.set_attribute(SpanAttributes.EMBEDDING_MODEL_NAME, self.model_name)
            span.set_attribute("embedding.text_count", len(texts))
            try:
                return fn(self, texts, *args, **kwargs)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

    return wrapper


class Embedder(ABC):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "embed" in cls.__dict__:
            cls.embed = _trace_embed(cls.embed)

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimension — must match config.embedding.dimension and the DB schema."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Embedding model identifier — written to child chunks for traceability."""
        ...

    def _validate_texts(self, texts: list[str]) -> None:
        if not texts:
            raise ValueError("Cannot embed an empty list of texts.")

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into dense vectors.

        Args:
            texts: list of text strings to embed

        Returns:
            list of embedding vectors, same order as input
        """
        ...
