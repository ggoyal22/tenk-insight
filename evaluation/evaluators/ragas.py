"""RAGAS-backed evaluator.

Metric name → RAGAS class mapping:
  faithfulness          → Faithfulness
  answer_relevancy      → AnswerRelevancy
  context_precision     → ContextPrecisionWithoutReference  (query-relative; no golden needed)
  context_recall        → ContextRecall                     (reference-required)

Reference-required metrics are silently dropped when no sample in the batch
has a non-None reference, rather than raising an error.
"""

import logging
import math
from typing import Any

from openai import AsyncOpenAI
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithoutReference,
    ContextRecall,
    Faithfulness,
)

from config.loader import JudgeLLMConfig
from evaluation.evaluators.base import BaseEvaluator
from evaluation.types import EvalSample, EvaluationResult

logger = logging.getLogger(__name__)

_METRIC_REGISTRY: dict[str, type] = {
    "faithfulness": Faithfulness,
    "answer_relevancy": AnswerRelevancy,
    "context_precision": ContextPrecisionWithoutReference,
    "context_recall": ContextRecall,
}

# Keys each metric's ascore() expects — used to build batch_score inputs.
_METRIC_KWARGS: dict[str, list[str]] = {
    "faithfulness": ["user_input", "response", "retrieved_contexts"],
    "answer_relevancy": ["user_input", "response"],
    "context_precision": ["user_input", "response", "retrieved_contexts"],
    "context_recall": ["user_input", "retrieved_contexts", "reference"],
}

_REFERENCE_REQUIRED: frozenset[str] = frozenset({"context_recall"})


class RagasEvaluator(BaseEvaluator):
    def __init__(self, judge_llm_config: JudgeLLMConfig) -> None:
        self._judge_llm_config = judge_llm_config

    def evaluate(self, samples: list[EvalSample], metrics: list[str]) -> EvaluationResult:
        unknown = [m for m in metrics if m not in _METRIC_REGISTRY]
        if unknown:
            raise ValueError(
                f"Unknown metric(s): {unknown}. "
                f"Supported: {sorted(_METRIC_REGISTRY.keys())}"
            )

        if not samples:
            logger.warning("No samples provided — skipping evaluation")
            return EvaluationResult(scores=[], aggregate={})

        has_reference = any(s.reference is not None for s in samples)
        active_metrics = []
        for name in metrics:
            if name in _REFERENCE_REQUIRED and not has_reference:
                logger.warning(
                    "Metric '%s' requires reference answers but none are set — skipping", name
                )
                continue
            active_metrics.append(name)

        if not active_metrics:
            logger.warning("No active metrics after filtering — returning empty scores")
            return EvaluationResult(scores=[{} for _ in samples], aggregate={})

        llm = self._build_llm()
        embeddings = self._build_embeddings()

        per_sample: list[dict[str, float]] = [{} for _ in samples]

        for name in active_metrics:
            metric = (
                _METRIC_REGISTRY[name](llm=llm, embeddings=embeddings)
                if name == "answer_relevancy"
                else _METRIC_REGISTRY[name](llm=llm)
            )
            inputs = [_sample_to_kwargs(s, _METRIC_KWARGS[name]) for s in samples]
            results = metric.batch_score(inputs)
            for i, res in enumerate(results):
                try:
                    v = float(res.value)
                    if not math.isnan(v):
                        per_sample[i][name] = v
                except (TypeError, ValueError):
                    pass

        aggregate = {
            name: _mean([s[name] for s in per_sample if name in s])
            for name in active_metrics
            if any(name in s for s in per_sample)
        }

        return EvaluationResult(scores=per_sample, aggregate=aggregate)

    def _build_llm(self) -> Any:
        cfg = self._judge_llm_config
        if cfg.provider == "openai":
            api_key = cfg.api_key.get_secret_value() if cfg.api_key else None
            self._openai_client = AsyncOpenAI(api_key=api_key)
            return llm_factory(cfg.model, provider="openai", client=self._openai_client)
        raise ValueError(f"Unsupported judge LLM provider: '{cfg.provider}'")

    def _build_embeddings(self) -> Any:
        cfg = self._judge_llm_config
        if cfg.provider == "openai":
            return embedding_factory(provider="openai", client=self._openai_client)
        raise ValueError(f"Unsupported judge LLM provider: '{cfg.provider}'")


def _sample_to_kwargs(sample: EvalSample, keys: list[str]) -> dict:
    all_fields = {
        "user_input": sample.user_input,
        "response": sample.response,
        "retrieved_contexts": sample.retrieved_contexts,
        "reference": sample.reference,
    }
    return {k: all_fields[k] for k in keys}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")
