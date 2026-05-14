"""Tests for evaluation/evaluators/ragas.py."""

from unittest.mock import MagicMock, patch

import pytest

from evaluation.evaluators.ragas import RagasEvaluator, _METRIC_REGISTRY, _REFERENCE_REQUIRED
from evaluation.types import EvalSample, EvaluationResult
from config.loader import JudgeLLMConfig


def _make_config(**kwargs) -> JudgeLLMConfig:
    defaults = {"provider": "openai", "model": "gpt-4o-mini", "api_key": None}
    return JudgeLLMConfig(**{**defaults, **kwargs})


def _sample(reference: str | None = None) -> EvalSample:
    return EvalSample(
        trace_id="t1",
        query_type="single",
        user_input="What is NVDA revenue?",
        retrieved_contexts=["Revenue was $60.9B"],
        response="$60.9B",
        reference=reference,
    )


# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------

def test_all_expected_metrics_in_registry():
    assert set(_METRIC_REGISTRY.keys()) == {
        "faithfulness", "answer_relevancy", "context_precision", "context_recall"
    }


def test_reference_required_is_subset_of_registry():
    assert _REFERENCE_REQUIRED.issubset(_METRIC_REGISTRY.keys())


def test_context_recall_is_reference_required():
    assert "context_recall" in _REFERENCE_REQUIRED


def test_context_precision_is_not_reference_required():
    assert "context_precision" not in _REFERENCE_REQUIRED


# ---------------------------------------------------------------------------
# Unknown metric validation
# ---------------------------------------------------------------------------

def test_unknown_metric_raises_value_error():
    evaluator = RagasEvaluator(_make_config())
    with pytest.raises(ValueError, match="Unknown metric"):
        evaluator.evaluate([_sample()], metrics=["faithfulnes"])


def test_multiple_unknown_metrics_listed_in_error():
    evaluator = RagasEvaluator(_make_config())
    with pytest.raises(ValueError, match="Unknown metric"):
        evaluator.evaluate([_sample()], metrics=["bad1", "bad2"])


# ---------------------------------------------------------------------------
# Empty samples
# ---------------------------------------------------------------------------

def test_empty_samples_returns_empty_evaluation_result():
    evaluator = RagasEvaluator(_make_config())
    result = evaluator.evaluate([], metrics=["faithfulness"])
    assert isinstance(result, EvaluationResult)
    assert result.scores == []
    assert result.aggregate == {}


# ---------------------------------------------------------------------------
# Reference-required metric filtering
# ---------------------------------------------------------------------------

def _patch_ragas(mock_scores: dict):
    """Context manager that stubs out RAGAS internals for unit tests."""
    async def _async_scores(scores):
        return scores

    def make_metric_class(name):
        scores = [MagicMock(value=s) for s in mock_scores.get(name, [])]
        instance = MagicMock()
        instance.abatch_score = MagicMock(return_value=_async_scores(scores))
        return MagicMock(return_value=instance)

    mock_registry = {k: make_metric_class(k) for k in _METRIC_REGISTRY}
    return (
        patch("evaluation.evaluators.ragas._METRIC_REGISTRY", mock_registry),
        patch("evaluation.evaluators.ragas.llm_factory"),
        patch("evaluation.evaluators.ragas.AsyncOpenAI"),
    )


def test_context_recall_dropped_when_no_references():
    evaluator = RagasEvaluator(_make_config())
    patches = _patch_ragas({"faithfulness": [0.9]})
    with patches[0], patches[1], patches[2]:
        result = evaluator.evaluate(
            [_sample(reference=None)],
            metrics=["faithfulness", "context_recall"],
        )
    assert "context_recall" not in result.aggregate
    assert "faithfulness" in result.aggregate
    assert len(result.scores) == 1
    assert "context_recall" not in result.scores[0]


def test_context_recall_included_when_reference_present():
    evaluator = RagasEvaluator(_make_config())
    patches = _patch_ragas({"faithfulness": [0.9], "context_recall": [0.8]})
    with patches[0], patches[1], patches[2]:
        result = evaluator.evaluate(
            [_sample(reference="Expected answer")],
            metrics=["faithfulness", "context_recall"],
        )
    assert "context_recall" in result.aggregate
    assert "faithfulness" in result.aggregate
    assert "context_recall" in result.scores[0]


def test_all_metrics_dropped_returns_empty_scores():
    evaluator = RagasEvaluator(_make_config())
    result = evaluator.evaluate(
        [_sample(reference=None)],
        metrics=["context_recall"],
    )
    assert isinstance(result, EvaluationResult)
    assert result.aggregate == {}
    assert result.scores == [{}]


def test_scores_and_aggregate_are_consistent():
    evaluator = RagasEvaluator(_make_config())
    patches = _patch_ragas({"faithfulness": [0.8, 0.6]})
    with patches[0], patches[1], patches[2]:
        result = evaluator.evaluate(
            [_sample(), _sample()],
            metrics=["faithfulness"],
        )
    assert len(result.scores) == 2
    assert result.scores[0]["faithfulness"] == pytest.approx(0.8)
    assert result.scores[1]["faithfulness"] == pytest.approx(0.6)
    assert result.aggregate["faithfulness"] == pytest.approx(0.7)


def test_nan_scores_are_filtered_from_per_sample_and_aggregate():
    evaluator = RagasEvaluator(_make_config())
    patches = _patch_ragas({"faithfulness": [float("nan")]})
    with patches[0], patches[1], patches[2]:
        result = evaluator.evaluate([_sample()], metrics=["faithfulness"])
    assert "faithfulness" not in result.scores[0]
    assert "faithfulness" not in result.aggregate


# ---------------------------------------------------------------------------
# LLM wiring
# ---------------------------------------------------------------------------

def test_build_llm_raises_for_unsupported_provider():
    cfg = JudgeLLMConfig(provider="openai", model="gpt-4o-mini", api_key=None)
    cfg.provider = "unsupported"  # bypass Literal validation for test
    evaluator = RagasEvaluator(cfg)
    with pytest.raises(ValueError, match="Unsupported judge LLM provider"):
        evaluator._build_llm()


def test_build_llm_passes_api_key():
    from pydantic import SecretStr
    cfg = _make_config(api_key=SecretStr("sk-test"))
    evaluator = RagasEvaluator(cfg)
    with patch("evaluation.evaluators.ragas.AsyncOpenAI") as mock_openai, \
         patch("evaluation.evaluators.ragas.llm_factory"):
        evaluator._build_llm()
        call_kwargs = mock_openai.call_args.kwargs
        assert call_kwargs["api_key"] == "sk-test"
        assert "http_client" in call_kwargs
