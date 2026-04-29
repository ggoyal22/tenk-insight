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


@dataclass
class LLMResponse:
    content: str
    usage: LLMUsage


@dataclass
class StructuredResponse(Generic[T]):
    parsed: T
    usage: LLMUsage
