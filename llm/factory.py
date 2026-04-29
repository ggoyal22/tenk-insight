from config.loader import LLMConfig
from llm.base import BaseLLM
from llm.ollama import OllamaLLM

_PROVIDERS: dict[str, type[BaseLLM]] = {
    "ollama": OllamaLLM,
}


def build_llm(config: LLMConfig) -> BaseLLM:
    cls = _PROVIDERS.get(config.provider)
    if cls is None:
        raise ValueError(
            f"Unknown LLM provider: {config.provider!r}. "
            f"Supported providers: {list(_PROVIDERS)}"
        )
    return cls(config)
