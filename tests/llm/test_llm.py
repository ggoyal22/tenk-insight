"""
Tests for the provider-agnostic parts of the llm/ module: types, LLMError,
and the build_llm factory.
"""

import pytest
from pydantic import BaseModel

from config.loader import LLMConfig
from llm.base import BaseLLM, LLMError
from llm.factory import build_llm
from llm.ollama import OllamaLLM
from llm.types import LLMResponse, LLMUsage, Message, StructuredResponse
from tests.conftest import VALID_LLM


# ---------------------------------------------------------------------------
# LLMError
# ---------------------------------------------------------------------------

def test_llm_error_message_includes_provider_and_model():
    err = LLMError("request timed out", provider="ollama", model="llama3.1:8b")
    assert "ollama" in str(err)
    assert "llama3.1:8b" in str(err)
    assert "request timed out" in str(err)


def test_llm_error_exposes_provider_and_model():
    err = LLMError("failed", provider="ollama", model="llama3.1:8b")
    assert err.provider == "ollama"
    assert err.model == "llama3.1:8b"


# ---------------------------------------------------------------------------
# build_llm factory
# ---------------------------------------------------------------------------

def test_build_llm_returns_ollama_instance():
    config = LLMConfig(**VALID_LLM)
    llm = build_llm(config)
    assert isinstance(llm, OllamaLLM)
    assert isinstance(llm, BaseLLM)


def test_build_llm_rejects_unknown_provider():
    # model_construct bypasses Pydantic validation to simulate a provider added to the
    # Literal but not yet registered in the factory — the factory must still raise clearly.
    config = LLMConfig.model_construct(**{**VALID_LLM, "provider": "vllm"})
    with pytest.raises(ValueError, match="vllm"):
        build_llm(config)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

def test_message_fields():
    msg = Message(role="user", content="hello")
    assert msg.role == "user"
    assert msg.content == "hello"


def test_llm_usage_fields():
    usage = LLMUsage(input_tokens=10, output_tokens=20)
    assert usage.input_tokens == 10
    assert usage.output_tokens == 20


def test_llm_response_fields():
    usage = LLMUsage(input_tokens=5, output_tokens=8)
    response = LLMResponse(content="Paris", usage=usage)
    assert response.content == "Paris"
    assert response.usage is usage


def test_structured_response_fields():
    class _Schema(BaseModel):
        value: str

    usage = LLMUsage(input_tokens=3, output_tokens=6)
    parsed = _Schema(value="test")
    result = StructuredResponse(parsed=parsed, usage=usage)
    assert result.parsed is parsed
    assert result.usage is usage
