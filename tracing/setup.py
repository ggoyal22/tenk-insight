import logging
import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from openinference.instrumentation.langchain import LangChainInstrumentor

from config.loader import TracingConfig

logger = logging.getLogger(__name__)


def setup_tracing(config: TracingConfig) -> None:
    """Configure the OTel tracer provider and instrument LangGraph.

    No-op when tracing.enabled is false — safe to call unconditionally at startup.
    When enabled, reads OTEL_EXPORTER_OTLP_ENDPOINT and OTEL_SERVICE_NAME from
    the environment automatically (standard OTel env vars, no code changes needed
    to switch backends).
    """
    if not config.enabled:
        return

    try:
        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)
        LangChainInstrumentor().instrument()
        logger.info(
            "Tracing enabled — exporting spans to %s",
            os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "(endpoint not set)"),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialise tracing: {exc}. "
            "Check that OTEL_EXPORTER_OTLP_ENDPOINT is reachable and that "
            "openinference-instrumentation-langchain is installed."
        ) from exc
