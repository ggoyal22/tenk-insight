"""Factory functions for building evaluation pipeline components from config."""

import os

from config.loader import EvaluationConfig


def _get_phoenix_db_path() -> str:
    path = (os.environ.get("PHOENIX_DB_PATH") or "").strip()
    if not path:
        raise ValueError(
            "PHOENIX_DB_PATH is not set or is empty. "
            "This should have been caught by EvaluationConfig validation — check your .env file."
        )
    return path
from evaluation.evaluators.base import BaseEvaluator
from evaluation.evaluators.ragas import RagasEvaluator
from evaluation.exporters.base import BaseResultExporter
from evaluation.exporters.jsonl import JSONLResultExporter
from evaluation.exporters.phoenix import PhoenixTraceAnnotationExporter
from evaluation.extractors.base import BaseExtractor
from evaluation.extractors.phoenix import PhoenixExtractor


def build_extractor(
    config: EvaluationConfig, project_name: str | None = None
) -> BaseExtractor:
    if config.extractor.backend == "phoenix":
        return PhoenixExtractor(_get_phoenix_db_path(), project_name=project_name)
    raise ValueError(f"Unknown extractor backend: '{config.extractor.backend}'")


def build_evaluator(config: EvaluationConfig) -> BaseEvaluator:
    if config.evaluator.backend == "ragas":
        return RagasEvaluator(config.evaluator.judge_llm)
    raise ValueError(f"Unknown evaluator backend: '{config.evaluator.backend}'")


def build_exporters(config: EvaluationConfig) -> list[BaseResultExporter]:
    exporters: list[BaseResultExporter] = [JSONLResultExporter(config.results.results_dir)]
    if config.results.phoenix_annotations:
        exporters.append(PhoenixTraceAnnotationExporter(_get_phoenix_db_path()))
    return exporters
