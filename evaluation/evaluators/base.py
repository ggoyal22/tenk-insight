"""Abstract base class for evaluators."""

from abc import ABC, abstractmethod

from evaluation.types import EvalSample, EvaluationResult


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, samples: list[EvalSample], metrics: list[str]) -> EvaluationResult:
        """Evaluate samples against the given metrics.

        Returns EvaluationResult with per-sample scores and their aggregate means.
        Metrics that cannot be computed (e.g. context_recall when no sample has a
        reference) are omitted from both scores and aggregate rather than raising.
        """
        ...
