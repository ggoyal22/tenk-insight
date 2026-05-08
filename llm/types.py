from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@dataclass
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class LLMUsage:
    input_tokens: int
    output_tokens: int

    def __add__(self, other: "LLMUsage") -> "LLMUsage":
        return LLMUsage(self.input_tokens + other.input_tokens, self.output_tokens + other.output_tokens)


@dataclass
class LLMResponse:
    content: str
    usage: LLMUsage


@dataclass
class StructuredResponse(Generic[T]):
    parsed: T
    usage: LLMUsage
