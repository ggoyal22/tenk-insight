import logging
from contextlib import contextmanager
from typing import Iterator

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import Span
from openinference.instrumentation.langchain import get_current_span
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes

logger = logging.getLogger(__name__)


def resolve_parent_context() -> Context | None:
    """Resolve the parent OTel context for a manual span created inside the pipeline.

    The LangChain instrumentor records LangGraph/node spans through callbacks that
    never enter the OTel ambient context, so a manual span opened inside a node
    (e.g. an LLM call) has no ambient parent and would start its own trace. This
    reads the instrumentor's own notion of the current LangChain/LangGraph span and
    returns a context rooted at it, so the manual span nests under the issuing node.

    Returns None when no instrumentor span is active — the caller then opens its
    span against the ambient context, i.e. standard behaviour. Also returns None
    when the LangChain runtime isn't loaded (an LLM call outside any graph run),
    so the LLM path never breaks on the lookup.
    """
    try:
        span = get_current_span()
    except Exception:
        return None
    if span is None or not span.get_span_context().is_valid:
        return None
    return trace.set_span_in_context(span)


@contextmanager
def query_span(query: str, pipeline_mode: str = "classic") -> Iterator[Span]:
    """Open the root span for one pipeline invocation.

    Every span produced while answering a query nests under this one, giving
    Phoenix a single trace per question. A no-op when tracing is disabled, since
    the no-op tracer yields a non-recording span whose attribute setters do nothing.
    """
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("query") as span:
        span.set_attribute(
            SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value
        )
        span.set_attribute(SpanAttributes.INPUT_VALUE, query)
        span.set_attribute("pipeline.mode", pipeline_mode)
        yield span
