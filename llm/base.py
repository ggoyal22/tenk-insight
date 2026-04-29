from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

from llm.types import LLMResponse, Message, StructuredResponse

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Raised when an LLM provider call fails."""

    def __init__(self, message: str, provider: str, model: str) -> None:
        super().__init__(f"[{provider}/{model}] {message}")
        self.provider = provider
        self.model = model


class BaseLLM(ABC):
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
