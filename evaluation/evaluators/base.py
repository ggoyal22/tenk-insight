"""Abstract base class for evaluators."""

from abc import ABC, abstractmethod

from evaluation.types import EvalSample


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, samples: list[EvalSample], metrics: list[str]) -> dict[str, float]:
        """Evaluate samples against the given metrics.

        Returns {metric_name: mean_score}. Metrics that cannot be computed
        (e.g. context_recall when no sample has a reference) are omitted
        from the result rather than raising.
        """
        ...
