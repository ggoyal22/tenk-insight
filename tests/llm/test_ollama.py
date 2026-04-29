"""
Tests for OllamaLLM. All tests mock ollama.Client — no live Ollama server needed.
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from config.loader import LLMConfig
from llm.base import LLMError
from llm.ollama import OllamaLLM
from llm.types import LLMResponse, StructuredResponse
from llm.types import Message
from tests.conftest import VALID_LLM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ollama_response(
    content: str, input_tokens: int = 10, output_tokens: int = 20
) -> MagicMock:
    """Build a mock that looks like an ollama.ChatResponse."""
    response = MagicMock()
    response.message.content = content
    response.prompt_eval_count = input_tokens
    response.eval_count = output_tokens
    return response


class _SampleSchema(BaseModel):
    answer: str
    confidence: float


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_ollama_llm_rejects_missing_base_url():
    config = LLMConfig(**{**VALID_LLM, "base_url": None})
    with pytest.raises(ValueError, match="LLM_BASE_URL"):
        OllamaLLM(config)


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------

def test_chat_returns_llm_response():
    mock_response = _make_ollama_response("Paris", input_tokens=5, output_tokens=3)
    with patch("llm.ollama.Client") as MockClient:
        MockClient.return_value.chat.return_value = mock_response
        llm = OllamaLLM(LLMConfig(**VALID_LLM))
        result = llm.chat([Message(role="user", content="Capital of France?")])

    assert isinstance(result, LLMResponse)
    assert result.content == "Paris"
    assert result.usage.input_tokens == 5
    assert result.usage.output_tokens == 3


def test_chat_wraps_api_error_as_llm_error():
    from ollama import ResponseError
    with patch("llm.ollama.Client") as MockClient:
        MockClient.return_value.chat.side_effect = ResponseError("model not found")
        llm = OllamaLLM(LLMConfig(**VALID_LLM))
        with pytest.raises(LLMError, match="ollama"):
            llm.chat([Message(role="user", content="hello")])


def test_chat_handles_null_token_counts():
    mock_response = _make_ollama_response("ok")
    mock_response.prompt_eval_count = None
    mock_response.eval_count = None
    with patch("llm.ollama.Client") as MockClient:
        MockClient.return_value.chat.return_value = mock_response
        llm = OllamaLLM(LLMConfig(**VALID_LLM))
        result = llm.chat([Message(role="user", content="hi")])

    assert result.usage.input_tokens == 0
    assert result.usage.output_tokens == 0


# ---------------------------------------------------------------------------
# chat_structured
# ---------------------------------------------------------------------------

def test_chat_structured_returns_parsed_model():
    payload = '{"answer": "Paris", "confidence": 0.95}'
    mock_response = _make_ollama_response(payload, input_tokens=8, output_tokens=12)
    with patch("llm.ollama.Client") as MockClient:
        MockClient.return_value.chat.return_value = mock_response
        llm = OllamaLLM(LLMConfig(**VALID_LLM))
        result = llm.chat_structured([Message(role="user", content="Capital?")], _SampleSchema)

    assert isinstance(result, StructuredResponse)
    assert isinstance(result.parsed, _SampleSchema)
    assert result.parsed.answer == "Paris"
    assert result.parsed.confidence == pytest.approx(0.95)
    assert result.usage.input_tokens == 8
    assert result.usage.output_tokens == 12


def test_chat_structured_passes_json_schema_to_client():
    payload = '{"answer": "Paris", "confidence": 0.9}'
    mock_response = _make_ollama_response(payload)
    with patch("llm.ollama.Client") as MockClient:
        mock_client = MockClient.return_value
        mock_client.chat.return_value = mock_response
        llm = OllamaLLM(LLMConfig(**VALID_LLM))
        llm.chat_structured([Message(role="user", content="hi")], _SampleSchema)

        _, kwargs = mock_client.chat.call_args
        assert kwargs.get("format") == _SampleSchema.model_json_schema()


def test_chat_structured_raises_llm_error_on_invalid_json():
    mock_response = _make_ollama_response("not valid json at all")
    with patch("llm.ollama.Client") as MockClient:
        MockClient.return_value.chat.return_value = mock_response
        llm = OllamaLLM(LLMConfig(**VALID_LLM))
        with pytest.raises(LLMError, match="_SampleSchema"):
            llm.chat_structured([Message(role="user", content="hi")], _SampleSchema)


def test_chat_structured_wraps_api_error_as_llm_error():
    from ollama import ResponseError
    with patch("llm.ollama.Client") as MockClient:
        MockClient.return_value.chat.side_effect = ResponseError("context length exceeded")
        llm = OllamaLLM(LLMConfig(**VALID_LLM))
        with pytest.raises(LLMError):
            llm.chat_structured([Message(role="user", content="hi")], _SampleSchema)
