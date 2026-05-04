"""Tests for evaluation/exporters/jsonl.py and evaluation/exporters/phoenix.py."""

import json
import sqlite3

import pytest

from evaluation.exporters.jsonl import JSONLResultExporter
from evaluation.exporters.phoenix import PhoenixTraceAnnotationExporter
from evaluation.types import DatasetScores, EvalSample, EvaluationResult, RunResult


def _run_result(**kwargs) -> RunResult:
    defaults = dict(
        run_id="2026-05-04T10:00:00",
        git_sha="abc1234",
        extractor="phoenix",
        evaluator="ragas",
        golden_used=False,
        datasets=[DatasetScores(dataset="single", scores={"faithfulness": 0.85}, n_samples=2)],
    )
    return RunResult(**{**defaults, **kwargs})


def _sample(trace_id: str = "span-001") -> EvalSample:
    return EvalSample(
        trace_id=trace_id,
        query_type="single",
        user_input="What is NVDA revenue?",
        retrieved_contexts=["Revenue was $60.9B"],
        response="$60.9B",
    )


def _evaluation(scores: list[dict]) -> EvaluationResult:
    aggregate = {}
    if scores:
        all_keys = {k for d in scores for k in d}
        aggregate = {k: sum(d.get(k, 0) for d in scores) / len(scores) for k in all_keys}
    return EvaluationResult(scores=scores, aggregate=aggregate)


# ---------------------------------------------------------------------------
# JSONLResultExporter
# ---------------------------------------------------------------------------

def test_jsonl_creates_file_if_missing(tmp_path):
    exporter = JSONLResultExporter(str(tmp_path / "results"))
    result = _run_result()
    exporter.export(result, [_sample()], _evaluation([{"faithfulness": 0.9}]))
    assert (tmp_path / "results" / "runs.jsonl").exists()


def test_jsonl_appends_valid_json_line(tmp_path):
    exporter = JSONLResultExporter(str(tmp_path))
    result = _run_result()
    exporter.export(result, [_sample()], _evaluation([{"faithfulness": 0.9}]))
    lines = (tmp_path / "runs.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["run_id"] == "2026-05-04T10:00:00"
    assert record["git_sha"] == "abc1234"


def test_jsonl_appends_multiple_runs(tmp_path):
    exporter = JSONLResultExporter(str(tmp_path))
    exporter.export(_run_result(run_id="run-1"), [_sample()], _evaluation([{}]))
    exporter.export(_run_result(run_id="run-2"), [_sample()], _evaluation([{}]))
    lines = (tmp_path / "runs.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["run_id"] == "run-1"
    assert json.loads(lines[1])["run_id"] == "run-2"


def test_jsonl_serializes_dataset_scores(tmp_path):
    exporter = JSONLResultExporter(str(tmp_path))
    result = _run_result(datasets=[
        DatasetScores(dataset="single", scores={"faithfulness": 0.9}, n_samples=1)
    ])
    exporter.export(result, [_sample()], _evaluation([{"faithfulness": 0.9}]))
    record = json.loads((tmp_path / "runs.jsonl").read_text())
    assert record["datasets"][0]["scores"]["faithfulness"] == pytest.approx(0.9)


def test_jsonl_creates_nested_results_dir(tmp_path):
    exporter = JSONLResultExporter(str(tmp_path / "deep" / "nested"))
    exporter.export(_run_result(), [_sample()], _evaluation([{}]))
    assert (tmp_path / "deep" / "nested" / "runs.jsonl").exists()


# ---------------------------------------------------------------------------
# PhoenixTraceAnnotationExporter
# ---------------------------------------------------------------------------

def _make_db(tmp_path) -> str:
    db_path = str(tmp_path / "phoenix.db")
    return db_path


def test_phoenix_writes_annotation_rows(tmp_path):
    db_path = _make_db(tmp_path)
    exporter = PhoenixTraceAnnotationExporter(db_path)
    samples = [_sample("span-1"), _sample("span-2")]
    evaluation = _evaluation([
        {"faithfulness": 0.9},
        {"faithfulness": 0.7},
    ])
    exporter.export(_run_result(), samples, evaluation)

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT span_id, name, score FROM span_annotations ORDER BY span_id").fetchall()
    assert len(rows) == 2
    assert rows[0] == ("span-1", "faithfulness", pytest.approx(0.9))
    assert rows[1] == ("span-2", "faithfulness", pytest.approx(0.7))


def test_phoenix_writes_multiple_metrics(tmp_path):
    db_path = _make_db(tmp_path)
    exporter = PhoenixTraceAnnotationExporter(db_path)
    samples = [_sample("span-1")]
    evaluation = _evaluation([{"faithfulness": 0.9, "answer_relevancy": 0.8}])
    exporter.export(_run_result(), samples, evaluation)

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT name FROM span_annotations ORDER BY name").fetchall()
    assert {r[0] for r in rows} == {"faithfulness", "answer_relevancy"}


def test_phoenix_skips_export_when_no_scores(tmp_path):
    db_path = _make_db(tmp_path)
    exporter = PhoenixTraceAnnotationExporter(db_path)
    exporter.export(_run_result(), [], EvaluationResult(scores=[], aggregate={}))

    with sqlite3.connect(db_path) as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='span_annotations'"
        ).fetchall()
    assert tables == []


def test_phoenix_upserts_on_duplicate(tmp_path):
    db_path = _make_db(tmp_path)
    exporter = PhoenixTraceAnnotationExporter(db_path)
    samples = [_sample("span-1")]
    run = _run_result(run_id="run-1")

    exporter.export(run, samples, _evaluation([{"faithfulness": 0.7}]))
    exporter.export(run, samples, _evaluation([{"faithfulness": 0.9}]))

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT score FROM span_annotations").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == pytest.approx(0.9)


def test_phoenix_stores_run_id(tmp_path):
    db_path = _make_db(tmp_path)
    exporter = PhoenixTraceAnnotationExporter(db_path)
    exporter.export(_run_result(run_id="run-abc"), [_sample("s1")], _evaluation([{"faithfulness": 0.8}]))

    with sqlite3.connect(db_path) as conn:
        run_id = conn.execute("SELECT run_id FROM span_annotations").fetchone()[0]
    assert run_id == "run-abc"
