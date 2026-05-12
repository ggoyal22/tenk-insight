"""Tests for evaluation/factory.py and evaluation/runner.py."""

import os
from unittest.mock import MagicMock, patch

import pytest

from evaluation.evaluators.ragas import RagasEvaluator
from evaluation.exporters.jsonl import JSONLResultExporter
from evaluation.exporters.phoenix import PhoenixTraceAnnotationExporter
from evaluation.extractors.phoenix import PhoenixExtractor
from evaluation.factory import build_evaluator, build_exporters, build_extractor
from evaluation.runner import EvaluationRunner, _resolve_git_sha
from evaluation.types import DatasetScores, EvalSample, EvaluationResult, RunResult
from tests.conftest import VALID_EVALUATION
from config.loader import EvaluationConfig

_FAKE_DB = "/tmp/test_phoenix.db"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def phoenix_db_env():
    """Ensure PHOENIX_DB_PATH is set for all tests in this module."""
    with patch.dict(os.environ, {"PHOENIX_DB_PATH": _FAKE_DB}):
        yield


def _eval_config(**kwargs) -> EvaluationConfig:
    data = {**VALID_EVALUATION, **kwargs}
    return EvaluationConfig(**data)


def _sample(trace_id: str = "t1", query_type: str = "single") -> EvalSample:
    return EvalSample(
        trace_id=trace_id,
        query_type=query_type,
        user_input="What is NVDA revenue?",
        retrieved_contexts=["Revenue was $60.9B"],
        response="$60.9B",
    )


def _eval_result(n: int = 1, score: float = 0.9) -> EvaluationResult:
    scores = [{"faithfulness": score}] * n
    return EvaluationResult(scores=scores, aggregate={"faithfulness": score})


def _make_runner(samples_by_dataset, eval_result, exporters=None, golden=None):
    extractor = MagicMock()
    extractor.extract.return_value = samples_by_dataset
    evaluator = MagicMock()
    evaluator.evaluate.return_value = eval_result
    return EvaluationRunner(
        extractor=extractor,
        evaluator=evaluator,
        exporters=exporters or [],
        golden=golden or {},
    )


# ---------------------------------------------------------------------------
# Factory — build_extractor
# ---------------------------------------------------------------------------

def test_build_extractor_returns_phoenix_extractor(tmp_path):
    extractor = build_extractor(_eval_config())
    assert isinstance(extractor, PhoenixExtractor)


def test_build_extractor_raises_on_unknown_backend():
    config = _eval_config()
    config.extractor.backend = "unknown"
    with pytest.raises(ValueError, match="Unknown extractor backend"):
        build_extractor(config)


# ---------------------------------------------------------------------------
# Factory — build_evaluator
# ---------------------------------------------------------------------------

def test_build_evaluator_returns_ragas_evaluator():
    evaluator = build_evaluator(_eval_config())
    assert isinstance(evaluator, RagasEvaluator)


def test_build_evaluator_raises_on_unknown_backend():
    config = _eval_config()
    config.evaluator.backend = "unknown"
    with pytest.raises(ValueError, match="Unknown evaluator backend"):
        build_evaluator(config)


# ---------------------------------------------------------------------------
# Factory — build_exporters
# ---------------------------------------------------------------------------

def test_build_exporters_always_includes_jsonl(tmp_path):
    config = _eval_config(results={"phoenix_annotations": False, "results_dir": str(tmp_path)})
    exporters = build_exporters(config)
    assert any(isinstance(e, JSONLResultExporter) for e in exporters)


def test_build_exporters_adds_phoenix_when_enabled(tmp_path):
    config = _eval_config(results={"phoenix_annotations": True, "results_dir": str(tmp_path)})
    exporters = build_exporters(config)
    assert any(isinstance(e, PhoenixTraceAnnotationExporter) for e in exporters)


def test_build_exporters_omits_phoenix_when_disabled(tmp_path):
    config = _eval_config(results={"phoenix_annotations": False, "results_dir": str(tmp_path)})
    exporters = build_exporters(config)
    assert not any(isinstance(e, PhoenixTraceAnnotationExporter) for e in exporters)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def test_runner_returns_run_result():
    runner = _make_runner({"single": [_sample()]}, _eval_result())
    result = runner.run(_eval_config())
    assert isinstance(result, RunResult)


def test_runner_populates_dataset_scores():
    runner = _make_runner({"single": [_sample()]}, _eval_result(score=0.85))
    result = runner.run(_eval_config())
    assert len(result.datasets) == 1
    assert result.datasets[0].dataset == "single"
    assert result.datasets[0].scores["faithfulness"] == pytest.approx(0.85)
    assert result.datasets[0].n_samples == 1


def test_runner_skips_empty_datasets():
    runner = _make_runner(
        {"single": [], "multi_hop": [_sample(query_type="multi_hop")]},
        _eval_result(),
    )
    result = runner.run(_eval_config())
    datasets = [d.dataset for d in result.datasets]
    assert "single" not in datasets
    assert "multi_hop" in datasets


def test_runner_calls_each_exporter():
    exporter1, exporter2 = MagicMock(), MagicMock()
    runner = _make_runner({"single": [_sample()]}, _eval_result(), exporters=[exporter1, exporter2])
    runner.run(_eval_config())
    exporter1.export.assert_called_once()
    exporter2.export.assert_called_once()


def test_runner_golden_used_false_when_no_references():
    runner = _make_runner({"single": [_sample()]}, _eval_result())
    result = runner.run(_eval_config())
    assert result.golden_used is False


def test_runner_golden_used_true_when_reference_attached():
    extractor = MagicMock()
    extractor.extract.return_value = {"single": [_sample()]}
    evaluator = MagicMock()
    evaluator.evaluate.return_value = _eval_result()
    runner = EvaluationRunner(
        extractor=extractor,
        evaluator=evaluator,
        exporters=[],
        golden={"What is NVDA revenue?": "$60.9B"},
    )
    result = runner.run(_eval_config())
    assert result.golden_used is True


def test_runner_extractor_uses_config_datasets():
    extractor = MagicMock()
    extractor.extract.return_value = {}
    evaluator = MagicMock()
    runner = EvaluationRunner(extractor, evaluator, [], golden={})
    runner.run(_eval_config(datasets=["single"]))
    extractor.extract.assert_called_once_with(["single"], since=None)


def test_runner_run_id_is_iso8601_timestamp():
    runner = _make_runner({"single": [_sample()]}, _eval_result())
    result = runner.run(_eval_config())
    # ISO-8601 with microseconds: YYYY-MM-DDTHH:MM:SS.ffffff
    import re
    assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+", result.run_id)


# ---------------------------------------------------------------------------
# _resolve_git_sha
# ---------------------------------------------------------------------------

def test_resolve_git_sha_returns_string():
    sha = _resolve_git_sha()
    assert isinstance(sha, str)
    assert len(sha) > 0


def test_resolve_git_sha_falls_back_to_unknown():
    with patch("evaluation.runner.subprocess.check_output", side_effect=Exception("no git")):
        sha = _resolve_git_sha()
    assert sha == "unknown"
