import logging
from typing import TypeVar

from ollama import Client
from ollama import ResponseError as OllamaResponseError
from pydantic import BaseModel, ValidationError

from config.loader import LLMConfig
from llm.base import BaseLLM, LLMError
from llm.types import LLMResponse, LLMUsage, Message, StructuredResponse

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_PROVIDER = "ollama"


class OllamaLLM(BaseLLM):
    def __init__(self, config: LLMConfig) -> None:
        if not config.base_url:
            raise ValueError(
                "LLM_BASE_URL must be set in .env when using the Ollama provider."
            )
        self._client = Client(host=config.base_url, timeout=config.timeout)
        self._provider = _PROVIDER
        self._model = config.model
        self._options = {
            "temperature": config.temperature,
            "num_predict": config.max_tokens,
        }

    def chat(self, messages: list[Message], max_tokens: int | None = None) -> LLMResponse:
        options = {**self._options, **({"num_predict": max_tokens} if max_tokens is not None else {})}
        try:
            response = self._client.chat(
                model=self._model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                options=options,
            )
        except OllamaResponseError as e:
            raise LLMError(str(e), _PROVIDER, self._model) from e

        return LLMResponse(
            content=response.message.content,
            usage=LLMUsage(
                input_tokens=response.prompt_eval_count or 0,
                output_tokens=response.eval_count or 0,
            ),
        )

    def chat_structured(self, messages: list[Message], schema: type[T], max_tokens: int | None = None) -> StructuredResponse[T]:
        options = {**self._options, **({"num_predict": max_tokens} if max_tokens is not None else {})}
        try:
            response = self._client.chat(
                model=self._model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                format=schema.model_json_schema(),
                options=options,
            )
        except OllamaResponseError as e:
            raise LLMError(str(e), _PROVIDER, self._model) from e

        try:
            parsed = schema.model_validate_json(response.message.content)
        except ValidationError as e:
            raise LLMError(
                f"Response could not be parsed into {schema.__name__}: {e}",
                _PROVIDER,
                self._model,
            ) from e

        return StructuredResponse(
            parsed=parsed,
            usage=LLMUsage(
                input_tokens=response.prompt_eval_count or 0,
                output_tokens=response.eval_count or 0,
            ),
        )
