import logging
from typing import TypeVar

from openai import OpenAI, OpenAIError
from pydantic import BaseModel

from config.loader import LLMConfig
from llm.base import BaseLLM, LLMError
from llm.types import LLMResponse, LLMUsage, Message, StructuredResponse

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_PROVIDER = "openai"


class OpenAILLM(BaseLLM):
    def __init__(self, config: LLMConfig) -> None:
        if not config.api_key:
            raise ValueError(
                "LLM_API_KEY must be set in .env when using the OpenAI provider."
            )
        self._client = OpenAI(
            api_key=config.api_key.get_secret_value(),
            base_url=config.base_url or None,
            timeout=config.timeout,
        )
        self._model = config.model
        self._params = {
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }

    def chat(self, messages: list[Message]) -> LLMResponse:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                **self._params,
            )
        except OpenAIError as e:
            raise LLMError(str(e), _PROVIDER, self._model) from e

        return LLMResponse(
            content=response.choices[0].message.content,
            usage=LLMUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            ),
        )

    def chat_structured(self, messages: list[Message], schema: type[T]) -> StructuredResponse[T]:
        try:
            response = self._client.beta.chat.completions.parse(
                model=self._model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                response_format=schema,
                **self._params,
            )
        except OpenAIError as e:
            raise LLMError(str(e), _PROVIDER, self._model) from e

        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise LLMError(
                f"Response could not be parsed into {schema.__name__}",
                _PROVIDER,
                self._model,
            )

        return StructuredResponse(
            parsed=parsed,
            usage=LLMUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            ),
        )
