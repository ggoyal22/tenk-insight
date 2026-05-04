"""Tests for evaluation/types.py."""

from evaluation.types import DatasetScores, EvalSample, RunResult


def _sample(**kwargs) -> EvalSample:
    defaults = dict(
        trace_id="trace-001",
        query_type="single",
        user_input="What was NVDA revenue in FY2024?",
        retrieved_contexts=["Revenue was $60.9 billion.", "Data Center grew 217%."],
        response="NVDA revenue in FY2024 was $60.9 billion.",
        reference=None,
    )
    return EvalSample(**{**defaults, **kwargs})


# ---------------------------------------------------------------------------
# EvalSample
# ---------------------------------------------------------------------------

def test_eval_sample_stores_all_fields():
    s = _sample(reference="$60.9 billion")
    assert s.trace_id == "trace-001"
    assert s.query_type == "single"
    assert s.user_input == "What was NVDA revenue in FY2024?"
    assert len(s.retrieved_contexts) == 2
    assert s.response == "NVDA revenue in FY2024 was $60.9 billion."
    assert s.reference == "$60.9 billion"


def test_eval_sample_reference_defaults_to_none():
    s = _sample()
    assert s.reference is None


def test_eval_sample_accepts_empty_contexts():
    s = _sample(retrieved_contexts=[])
    assert s.retrieved_contexts == []


def test_eval_sample_accepts_multi_hop_query_type():
    s = _sample(query_type="multi_hop")
    assert s.query_type == "multi_hop"


# ---------------------------------------------------------------------------
# DatasetScores
# ---------------------------------------------------------------------------

def test_dataset_scores_stores_all_fields():
    ds = DatasetScores(
        dataset="single",
        scores={"faithfulness": 0.82, "answer_relevancy": 0.91},
        n_samples=10,
    )
    assert ds.dataset == "single"
    assert ds.scores["faithfulness"] == 0.82
    assert ds.n_samples == 10


def test_dataset_scores_accepts_empty_scores():
    ds = DatasetScores(dataset="comparison", scores={}, n_samples=0)
    assert ds.scores == {}
    assert ds.n_samples == 0


# ---------------------------------------------------------------------------
# RunResult
# ---------------------------------------------------------------------------

def test_run_result_stores_all_fields():
    ds = DatasetScores(dataset="single", scores={"faithfulness": 0.82}, n_samples=5)
    result = RunResult(
        run_id="2026-05-04T10:23:11",
        git_sha="abc1234",
        extractor="phoenix",
        evaluator="ragas",
        golden_used=True,
        datasets=[ds],
    )
    assert result.run_id == "2026-05-04T10:23:11"
    assert result.git_sha == "abc1234"
    assert result.golden_used is True
    assert len(result.datasets) == 1
    assert result.datasets[0].dataset == "single"


def test_run_result_datasets_defaults_to_empty_list():
    result = RunResult(
        run_id="2026-05-04T10:23:11",
        git_sha="unknown",
        extractor="phoenix",
        evaluator="ragas",
        golden_used=False,
    )
    assert result.datasets == []


def test_run_result_unknown_git_sha_is_valid():
    result = RunResult(
        run_id="2026-05-04T10:23:11",
        git_sha="unknown",
        extractor="phoenix",
        evaluator="ragas",
        golden_used=False,
    )
    assert result.git_sha == "unknown"
