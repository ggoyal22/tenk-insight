"""Abstract base class for result exporters."""

from abc import ABC, abstractmethod

from evaluation.types import EvalSample, EvaluationResult, RunResult


class BaseResultExporter(ABC):
    @abstractmethod
    def export(
        self,
        result: RunResult,
        samples: list[EvalSample],
        evaluation: EvaluationResult,
    ) -> None:
        """Persist evaluation results to the exporter's destination.

        result     — aggregate run metadata and dataset-level scores
        samples    — the EvalSamples that were evaluated (parallel to evaluation.scores)
        evaluation — per-sample scores and aggregate means from the evaluator
        """
        ...
