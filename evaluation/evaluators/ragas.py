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

from openai import OpenAI
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithoutReference,
    ContextRecall,
    Faithfulness,
)

from config.loader import JudgeLLMConfig
from evaluation.evaluators.base import BaseEvaluator
from evaluation.types import EvalSample

logger = logging.getLogger(__name__)

_METRIC_REGISTRY: dict[str, type] = {
    "faithfulness": Faithfulness,
    "answer_relevancy": AnswerRelevancy,
    "context_precision": ContextPrecisionWithoutReference,
    "context_recall": ContextRecall,
}

_REFERENCE_REQUIRED: frozenset[str] = frozenset({"context_recall"})


class RagasEvaluator(BaseEvaluator):
    def __init__(self, judge_llm_config: JudgeLLMConfig) -> None:
        self._judge_llm_config = judge_llm_config

    def evaluate(self, samples: list[EvalSample], metrics: list[str]) -> dict[str, float]:
        unknown = [m for m in metrics if m not in _METRIC_REGISTRY]
        if unknown:
            raise ValueError(
                f"Unknown metric(s): {unknown}. "
                f"Supported: {sorted(_METRIC_REGISTRY.keys())}"
            )

        if not samples:
            logger.warning("No samples provided — skipping evaluation")
            return {}

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
            return {}

        llm = self._build_llm()
        ragas_metrics = [_METRIC_REGISTRY[name](llm=llm) for name in active_metrics]

        dataset = EvaluationDataset(
            samples=[
                SingleTurnSample(
                    user_input=s.user_input,
                    retrieved_contexts=s.retrieved_contexts,
                    response=s.response,
                    reference=s.reference,
                )
                for s in samples
            ]
        )

        result = evaluate(dataset=dataset, metrics=ragas_metrics)
        scores = result.to_pandas()

        return {
            name: float(scores[name].mean())
            for name in active_metrics
            if name in scores.columns
        }

    def _build_llm(self):
        cfg = self._judge_llm_config
        if cfg.provider == "openai":
            api_key = cfg.api_key.get_secret_value() if cfg.api_key else None
            client = OpenAI(api_key=api_key)
            return llm_factory(cfg.model, provider="openai", client=client)
        raise ValueError(f"Unsupported judge LLM provider: '{cfg.provider}'")
