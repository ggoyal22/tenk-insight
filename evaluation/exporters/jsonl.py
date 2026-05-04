"""JSONL exporter — appends one JSON line per evaluation run to runs.jsonl."""

import dataclasses
import json
import logging
from pathlib import Path

from evaluation.exporters.base import BaseResultExporter
from evaluation.types import EvalSample, EvaluationResult, RunResult

logger = logging.getLogger(__name__)


class JSONLResultExporter(BaseResultExporter):
    def __init__(self, results_dir: str) -> None:
        self._path = Path(results_dir) / "runs.jsonl"

    def export(
        self,
        result: RunResult,
        samples: list[EvalSample],
        evaluation: EvaluationResult,
    ) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a") as f:
            f.write(json.dumps(dataclasses.asdict(result)) + "\n")
        logger.info("Appended run '%s' to '%s'", result.run_id, self._path)
