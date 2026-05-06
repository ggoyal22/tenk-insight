"""Evaluation runner — orchestrates the full evaluation pipeline.

Sequence for each call to run():
  1. Extract EvalSamples per dataset from the configured backend.
  2. Attach golden reference answers to samples (enables context_recall).
  3. Evaluate each dataset group with the configured evaluator.
  4. Assemble a RunResult from per-dataset aggregate scores.
  5. Export results via all configured exporters.
"""

import logging
import subprocess
from datetime import datetime, timezone

from config.loader import EvaluationConfig
from evaluation.evaluators.base import BaseEvaluator
from evaluation.exporters.base import BaseResultExporter
from evaluation.extractors.base import BaseExtractor
from evaluation.golden_loader import attach_references
from evaluation.types import DatasetScores, EvalSample, EvaluationResult, RunResult

logger = logging.getLogger(__name__)


class EvaluationRunner:
    def __init__(
        self,
        extractor: BaseExtractor,
        evaluator: BaseEvaluator,
        exporters: list[BaseResultExporter],
        golden: dict[str, str],
    ) -> None:
        self._extractor = extractor
        self._evaluator = evaluator
        self._exporters = exporters
        self._golden = golden

    def run(self, config: EvaluationConfig, since: datetime | None = None) -> RunResult:
        run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")
        git_sha = _resolve_git_sha()

        samples_by_dataset = self._extractor.extract(config.datasets, since=since)

        all_samples: list[EvalSample] = []
        all_scores: list[dict[str, float]] = []
        dataset_scores: list[DatasetScores] = []

        for dataset, samples in samples_by_dataset.items():
            if not samples:
                continue

            attach_references(samples, self._golden)
            eval_result = self._evaluator.evaluate(samples, config.metrics)

            all_samples.extend(samples)
            all_scores.extend(eval_result.scores)
            dataset_scores.append(
                DatasetScores(
                    dataset=dataset,
                    scores=eval_result.aggregate,
                    n_samples=len(samples),
                )
            )

        golden_used = any(s.reference is not None for s in all_samples)
        result = RunResult(
            run_id=run_id,
            git_sha=git_sha,
            extractor=config.extractor.backend,
            evaluator=config.evaluator.backend,
            golden_used=golden_used,
            datasets=dataset_scores,
        )

        flat_evaluation = EvaluationResult(scores=all_scores, aggregate={})
        for exporter in self._exporters:
            exporter.export(result, all_samples, flat_evaluation)

        logger.info(
            "Evaluation run '%s' complete — %d dataset(s), %d sample(s) total",
            run_id, len(dataset_scores), len(all_samples),
        )
        return result


def _resolve_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"
