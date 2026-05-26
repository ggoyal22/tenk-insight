import logging

from llm.types import LLMUsage

logger = logging.getLogger(__name__)

# Prices in USD per 1M tokens (input, output).
# Source: OpenAI pricing page — update here when rates change.
_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini":  (0.15,  0.60),
    "gpt-4o":       (2.50, 10.00),
    "gpt-4.1":      (2.00,  8.00),
    "gpt-4.1-mini": (0.40,  1.60),
    "gpt-4.1-nano": (0.10,  0.40),
}


def compute_cost(usage: LLMUsage, model: str, provider: str) -> float | str:
    """Return estimated USD cost for the given token usage, model, and provider.

    Returns 0.0 for local providers (Ollama) where there is no API charge.
    Returns "Unknown" for cloud models not present in the price table.
    """
    if provider == "ollama":
        return 0.0
    if model not in _PRICES:
        logger.warning("No pricing data for model %r — cost reported as Unknown", model)
        return "Unknown"
    input_price, output_price = _PRICES[model]
    return (usage.input_tokens * input_price + usage.output_tokens * output_price) / 1_000_000
