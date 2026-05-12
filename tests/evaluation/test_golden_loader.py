"""Tests for evaluation/golden_loader.py."""

import pytest
import yaml

from evaluation.golden_loader import attach_references, load_golden
from evaluation.types import EvalSample


def _write_yaml(path, data):
    with open(path, "w") as f:
        yaml.dump(data, f)


def _sample(query: str, reference: str | None = None) -> EvalSample:
    return EvalSample(
        trace_id="t1",
        query_type="single",
        user_input=query,
        retrieved_contexts=["ctx"],
        response="answer",
        reference=reference,
    )


# ---------------------------------------------------------------------------
# load_golden — None / missing path
# ---------------------------------------------------------------------------

def test_load_golden_returns_empty_dict_when_path_is_none():
    assert load_golden(None) == {}


def test_load_golden_returns_empty_dict_when_path_missing():
    assert load_golden("/tmp/does_not_exist_xyz.yaml") == {}


# ---------------------------------------------------------------------------
# load_golden — single file
# ---------------------------------------------------------------------------

def test_load_golden_single_file(tmp_path):
    f = tmp_path / "q.yaml"
    _write_yaml(f, [
        {"query": "What is NVDA revenue?", "answer": "$60.9B", "query_type": "single"},
    ])
    result = load_golden(str(f))
    assert result == {"What is NVDA revenue?": "$60.9B"}


def test_load_golden_multiple_entries(tmp_path):
    f = tmp_path / "q.yaml"
    _write_yaml(f, [
        {"query": "Q1", "answer": "A1", "query_type": "single"},
        {"query": "Q2", "answer": "A2", "query_type": "single"},
    ])
    result = load_golden(str(f))
    assert len(result) == 2
    assert result["Q1"] == "A1"
    assert result["Q2"] == "A2"


def test_load_golden_strips_whitespace_from_query(tmp_path):
    f = tmp_path / "q.yaml"
    _write_yaml(f, [{"query": "  padded query  ", "answer": "A", "query_type": "single"}])
    result = load_golden(str(f))
    assert "padded query" in result


# ---------------------------------------------------------------------------
# load_golden — directory
# ---------------------------------------------------------------------------

def test_load_golden_directory_merges_files(tmp_path):
    _write_yaml(tmp_path / "single.yaml", [
        {"query": "Q1", "answer": "A1", "query_type": "single"},
    ])
    _write_yaml(tmp_path / "multi_hop.yaml", [
        {"query": "Q2", "answer": "A2", "query_type": "multi_hop"},
    ])
    result = load_golden(str(tmp_path))
    assert result == {"Q1": "A1", "Q2": "A2"}


def test_load_golden_empty_directory_returns_empty_dict(tmp_path):
    result = load_golden(str(tmp_path))
    assert result == {}


def test_load_golden_ignores_non_yaml_files(tmp_path):
    (tmp_path / "notes.txt").write_text("not yaml")
    _write_yaml(tmp_path / "single.yaml", [
        {"query": "Q1", "answer": "A1", "query_type": "single"},
    ])
    result = load_golden(str(tmp_path))
    assert list(result.keys()) == ["Q1"]


def test_load_golden_raises_on_duplicate_query_across_files(tmp_path):
    _write_yaml(tmp_path / "a.yaml", [{"query": "Same Q", "answer": "A1", "query_type": "single"}])
    _write_yaml(tmp_path / "b.yaml", [{"query": "Same Q", "answer": "A2", "query_type": "single"}])
    with pytest.raises(ValueError, match="Duplicate golden queries"):
        load_golden(str(tmp_path))


# ---------------------------------------------------------------------------
# load_golden — validation errors
# ---------------------------------------------------------------------------

def test_load_golden_raises_on_invalid_yaml(tmp_path):
    f = tmp_path / "bad.yaml"
    f.write_text("key: [unclosed")
    with pytest.raises(ValueError, match="Failed to parse"):
        load_golden(str(f))


def test_load_golden_raises_when_root_is_not_list(tmp_path):
    f = tmp_path / "bad.yaml"
    _write_yaml(f, {"query": "Q", "answer": "A"})
    with pytest.raises(ValueError, match="must contain a YAML list"):
        load_golden(str(f))


def test_load_golden_raises_on_missing_query_field(tmp_path):
    f = tmp_path / "bad.yaml"
    _write_yaml(f, [{"answer": "A", "query_type": "single"}])
    with pytest.raises(ValueError, match="missing required field"):
        load_golden(str(f))


def test_load_golden_skips_entry_with_no_answer_field(tmp_path):
    f = tmp_path / "golden.yaml"
    _write_yaml(f, [{"query": "Q", "query_type": "single"}])
    result = load_golden(str(f))
    assert result == {}


def test_load_golden_raises_on_missing_query_type_field(tmp_path):
    f = tmp_path / "bad.yaml"
    _write_yaml(f, [{"query": "Q", "answer": "A"}])
    with pytest.raises(ValueError, match="missing required field"):
        load_golden(str(f))


def test_load_golden_raises_on_empty_query(tmp_path):
    f = tmp_path / "bad.yaml"
    _write_yaml(f, [{"query": "   ", "answer": "A", "query_type": "single"}])
    with pytest.raises(ValueError, match="empty 'query'"):
        load_golden(str(f))


def test_load_golden_raises_on_duplicate_query_within_file(tmp_path):
    f = tmp_path / "q.yaml"
    _write_yaml(f, [
        {"query": "Same Q", "answer": "A1", "query_type": "single"},
        {"query": "Same Q", "answer": "A2", "query_type": "single"},
    ])
    with pytest.raises(ValueError, match="duplicate query"):
        load_golden(str(f))


# ---------------------------------------------------------------------------
# attach_references
# ---------------------------------------------------------------------------

def test_attach_references_sets_reference_on_match():
    golden = {"What is NVDA revenue?": "$60.9B"}
    samples = [_sample("What is NVDA revenue?")]
    attach_references(samples, golden)
    assert samples[0].reference == "$60.9B"


def test_attach_references_leaves_unmatched_samples_as_none():
    golden = {"Q1": "A1"}
    samples = [_sample("Q2")]
    attach_references(samples, golden)
    assert samples[0].reference is None


def test_attach_references_partial_match():
    golden = {"Q1": "A1"}
    samples = [_sample("Q1"), _sample("Q2")]
    attach_references(samples, golden)
    assert samples[0].reference == "A1"
    assert samples[1].reference is None


def test_attach_references_is_noop_on_empty_golden():
    samples = [_sample("Q1")]
    attach_references(samples, {})
    assert samples[0].reference is None


def test_attach_references_mutates_in_place():
    golden = {"Q": "A"}
    s = _sample("Q")
    original_id = id(s)
    attach_references([s], golden)
    assert id(s) == original_id
    assert s.reference == "A"
