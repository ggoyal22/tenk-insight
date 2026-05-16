"""RAGAS-backed evaluator.

Metric name → RAGAS class mapping:
  faithfulness            → Faithfulness
  answer_relevancy        → AnswerRelevancy
  context_precision       → ContextPrecisionWithoutReference  (query-relative; no golden needed)
  context_recall          → ContextRecall                     (reference-required)
  answer_correctness      → AnswerCorrectness                 (reference-required)
  factual_correctness     → FactualCorrectness                (reference-required)
  semantic_similarity     → SemanticSimilarity                (reference-required; embeddings only)
  context_relevance       → ContextRelevance
  context_entity_recall   → ContextEntityRecall               (reference-required)
  noise_sensitivity       → NoiseSensitivity                  (reference-required)
  context_utilization     → ContextUtilization
  response_groundedness   → ResponseGroundedness

Reference-required metrics are silently dropped when no sample in the batch
has a non-None reference, rather than raising an error.
"""

import asyncio
import logging
import math
from typing import Any

import httpx
from openai import AsyncOpenAI
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerCorrectness,
    AnswerRelevancy,
    ContextEntityRecall,
    ContextPrecisionWithoutReference,
    ContextRecall,
    ContextRelevance,
    ContextUtilization,
    FactualCorrectness,
    Faithfulness,
    NoiseSensitivity,
    ResponseGroundedness,
    SemanticSimilarity,
)

from config.loader import JudgeLLMConfig
from evaluation.evaluators.base import BaseEvaluator
from evaluation.types import EvalSample, EvaluationResult

logger = logging.getLogger(__name__)

_METRIC_REGISTRY: dict[str, type] = {
    "faithfulness":          Faithfulness,
    "answer_relevancy":      AnswerRelevancy,
    "context_precision":     ContextPrecisionWithoutReference,
    "context_recall":        ContextRecall,
    "answer_correctness":    AnswerCorrectness,
    "factual_correctness":   FactualCorrectness,
    "semantic_similarity":   SemanticSimilarity,
    "context_relevance":     ContextRelevance,
    "context_entity_recall": ContextEntityRecall,
    "noise_sensitivity":     NoiseSensitivity,
    "context_utilization":   ContextUtilization,
    "response_groundedness": ResponseGroundedness,
}

# Keys each metric's ascore() expects — used to build batch_score inputs.
_METRIC_KWARGS: dict[str, list[str]] = {
    "faithfulness":          ["user_input", "response", "retrieved_contexts"],
    "answer_relevancy":      ["user_input", "response"],
    "context_precision":     ["user_input", "response", "retrieved_contexts"],
    "context_recall":        ["user_input", "retrieved_contexts", "reference"],
    "answer_correctness":    ["user_input", "response", "reference"],
    "factual_correctness":   ["response", "reference"],
    "semantic_similarity":   ["response", "reference"],
    "context_relevance":     ["user_input", "retrieved_contexts"],
    "context_entity_recall": ["reference", "retrieved_contexts"],
    "noise_sensitivity":     ["user_input", "response", "reference", "retrieved_contexts"],
    "context_utilization":   ["user_input", "response", "retrieved_contexts"],
    "response_groundedness": ["response", "retrieved_contexts"],
}

_REFERENCE_REQUIRED: frozenset[str] = frozenset({
    "context_recall", "answer_correctness", "factual_correctness",
    "semantic_similarity", "context_entity_recall", "noise_sensitivity",
})
_CONTEXTS_REQUIRED: frozenset[str] = frozenset({
    "faithfulness", "context_precision", "context_recall",
    "context_relevance", "context_entity_recall", "noise_sensitivity",
    "context_utilization", "response_groundedness",
})
# Metrics that take both llm + embeddings.
_EMBEDDINGS_REQUIRED: frozenset[str] = frozenset({"answer_relevancy", "answer_correctness"})
# Metrics that take only embeddings (no llm).
_EMBEDDINGS_ONLY: frozenset[str] = frozenset({"semantic_similarity"})


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
        asyncio.run(self._score_all(active_metrics, samples, per_sample, llm, embeddings))

        aggregate = {
            name: _mean([s[name] for s in per_sample if name in s])
            for name in active_metrics
            if any(name in s for s in per_sample)
        }

        return EvaluationResult(scores=per_sample, aggregate=aggregate)

    async def _score_all(
        self,
        active_metrics: list[str],
        samples: list[EvalSample],
        per_sample: list[dict[str, float]],
        llm: Any,
        embeddings: Any,
    ) -> None:
        async def _score_metric(name: str, delay: float = 0.0) -> None:
            if delay:
                await asyncio.sleep(delay)
            if name in _EMBEDDINGS_ONLY:
                metric = _METRIC_REGISTRY[name](embeddings=embeddings)
            elif name in _EMBEDDINGS_REQUIRED:
                metric = _METRIC_REGISTRY[name](llm=llm, embeddings=embeddings)
            else:
                metric = _METRIC_REGISTRY[name](llm=llm)

            if name in _CONTEXTS_REQUIRED:
                # RAGAS raises ValueError on empty lists — score those 0.0 directly
                scorable = [(i, s) for i, s in enumerate(samples) if s.retrieved_contexts]
                empty_count = len(samples) - len(scorable)
                if empty_count:
                    logger.warning(
                        "Metric '%s': %d sample(s) have no retrieved contexts — scored 0.0",
                        name, empty_count,
                    )
                for i, s in enumerate(samples):
                    if not s.retrieved_contexts:
                        per_sample[i][name] = 0.0
            else:
                scorable = list(enumerate(samples))

            if not scorable:
                return

            inputs = [_sample_to_kwargs(s, _METRIC_KWARGS[name]) for _, s in scorable]
            results = await metric.abatch_score(inputs)
            for (i, _), res in zip(scorable, results):
                try:
                    v = float(res.value)
                    if not math.isnan(v):
                        per_sample[i][name] = v
                except (TypeError, ValueError):
                    pass

        await asyncio.gather(*[
            _score_metric(name, delay=i * 0.5)
            for i, name in enumerate(active_metrics)
        ])

    def _build_llm(self) -> Any:
        cfg = self._judge_llm_config
        if cfg.provider == "openai":
            api_key = cfg.api_key.get_secret_value() if cfg.api_key else None
            self._openai_client = AsyncOpenAI(
                api_key=api_key,
                http_client=httpx.AsyncClient(
                    limits=httpx.Limits(max_connections=20, max_keepalive_connections=20),
                ),
            )
            return llm_factory(cfg.model, provider="openai", client=self._openai_client, max_tokens=cfg.max_tokens)
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
