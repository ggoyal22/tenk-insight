from abc import ABC, abstractmethod
from functools import wraps
from typing import TypeVar

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from openinference.semconv.trace import MessageAttributes, OpenInferenceSpanKindValues, SpanAttributes
from pydantic import BaseModel

from llm.types import LLMResponse, Message, StructuredResponse

T = TypeVar("T", bound=BaseModel)


def _trace_llm(fn):
    @wraps(fn)
    def wrapper(self, messages: list[Message], *args, **kwargs):
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(f"llm.{fn.__name__}") as span:
            span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.LLM.value)
            span.set_attribute(SpanAttributes.LLM_PROVIDER, getattr(self, "_provider", "unknown"))
            span.set_attribute(SpanAttributes.LLM_MODEL_NAME, self._model)

            for i, m in enumerate(messages):
                span.set_attribute(f"{SpanAttributes.LLM_INPUT_MESSAGES}.{i}.{MessageAttributes.MESSAGE_ROLE}", m.role)
                span.set_attribute(f"{SpanAttributes.LLM_INPUT_MESSAGES}.{i}.{MessageAttributes.MESSAGE_CONTENT}", m.content)

            try:
                result = fn(self, messages, *args, **kwargs)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

            span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, result.usage.input_tokens)
            span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, result.usage.output_tokens)

            output = result.content if isinstance(result, LLMResponse) else result.parsed.model_dump_json()
            span.set_attribute(f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}", "assistant")
            span.set_attribute(f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}", output)

            return result

    return wrapper


class LLMError(Exception):
    """Raised when an LLM provider call fails."""

    def __init__(self, message: str, provider: str, model: str) -> None:
        super().__init__(f"[{provider}/{model}] {message}")
        self.provider = provider
        self.model = model


class BaseLLM(ABC):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "chat" in cls.__dict__:
            cls.chat = _trace_llm(cls.chat)
        if "chat_structured" in cls.__dict__:
            cls.chat_structured = _trace_llm(cls.chat_structured)

    @abstractmethod
    def chat(self, messages: list[Message]) -> LLMResponse:
        """Send a conversation and return the text response with token usage."""
        ...

    @abstractmethod
    def chat_structured(self, messages: list[Message], schema: type[T]) -> StructuredResponse[T]:
        """Send a conversation and return a parsed Pydantic model with token usage.

        The provider is instructed to emit JSON conforming to schema's JSON Schema.
        Raises LLMError if the call fails or the response cannot be parsed.
        """
        ...
