from config.loader import LLMConfig
from llm.base import BaseLLM
from llm.ollama import OllamaLLM
from llm.openai import OpenAILLM

_PROVIDERS: dict[str, type[BaseLLM]] = {
    "ollama": OllamaLLM,
    "openai": OpenAILLM,
}


def build_llm(config: LLMConfig) -> BaseLLM:
    cls = _PROVIDERS.get(config.provider)
    if cls is None:
        raise ValueError(
            f"Unknown LLM provider: {config.provider!r}. "
            f"Supported providers: {list(_PROVIDERS)}"
        )
    return cls(config)
