"""Unit tests for PhoenixExtractor — SQLite is mocked; no live Phoenix DB required."""

import json
from unittest.mock import MagicMock, patch

import pytest

from evaluation.extractors.phoenix import PhoenixExtractor
from evaluation.types import EvalSample

# ---------------------------------------------------------------------------
# Helpers — build minimal span attribute payloads
# ---------------------------------------------------------------------------

def _langgraph_attrs(query: str, query_type: str) -> str:
    return json.dumps({
        "input":  {"value": json.dumps({"query": query, "query_type": query_type})},
        "output": {"value": json.dumps({"query": query, "query_type": query_type, "answer": None})},
    })


def _generate_attrs(answer: str) -> str:
    return json.dumps({
        "output": {"value": json.dumps({"answer": {"answer": answer}})},
    })


def _retrieve_attrs(chunks: list[str], use_parent: bool = True) -> str:
    results = []
    for text in chunks:
        if use_parent:
            results.append({"chunk": {"text": "child"}, "parent_chunk": {"text": text}})
        else:
            results.append({"chunk": {"text": text}, "parent_chunk": None})
    return json.dumps({
        "output": {"value": json.dumps({"completed_results": [results]})},
    })


def _make_rows(*spans: tuple[str, str]) -> list[tuple[int, str, str]]:
    """Each span is (name, attrs_json); returns (span_id, name, attrs_json) with auto IDs."""
    return [(i + 1, name, attrs) for i, (name, attrs) in enumerate(spans)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def extractor(tmp_path):
    return PhoenixExtractor(db_path=str(tmp_path / "test.db"))


def _mock_conn(trace_ids: list[int], spans_by_trace: dict[int, list[tuple[str, str]]]):
    """Return a mock sqlite3 connection whose execute() returns canned rows."""
    conn = MagicMock()

    def execute(sql, params=()):
        cur = MagicMock()
        if "DISTINCT spans.trace_rowid" in sql:
            cur.fetchall.return_value = [(tid,) for tid in trace_ids]
        else:
            tid = params[0]
            cur.fetchall.return_value = spans_by_trace.get(tid, [])
        return cur

    conn.execute.side_effect = execute
    conn.__enter__ = lambda s: conn
    conn.__exit__ = MagicMock(return_value=False)
    return conn


# ---------------------------------------------------------------------------
# Happy-path extraction
# ---------------------------------------------------------------------------

def test_extracts_single_trace(extractor):
    spans = _make_rows(
        ("LangGraph", _langgraph_attrs("NVDA revenue?", "single")),
        ("generate",  _generate_attrs("Revenue was $60.9B")),
        ("retrieve",  _retrieve_attrs(["chunk A", "chunk B"])),
    )
    conn = _mock_conn([1], {1: spans})

    with patch("sqlite3.connect", return_value=conn):
        result = extractor.extract(["single"])

    assert len(result["single"]) == 1
    s = result["single"][0]
    assert s.user_input == "NVDA revenue?"
    assert s.response == "Revenue was $60.9B"
    assert s.query_type == "single"
    assert s.reference is None


def test_extracted_contexts_use_parent_chunk(extractor):
    spans = _make_rows(
        ("LangGraph", _langgraph_attrs("NVDA revenue?", "single")),
        ("generate",  _generate_attrs("Revenue was $60.9B")),
        ("retrieve",  _retrieve_attrs(["parent text A", "parent text B"], use_parent=True)),
    )
    conn = _mock_conn([1], {1: spans})

    with patch("sqlite3.connect", return_value=conn):
        result = extractor.extract(["single"])

    assert result["single"][0].retrieved_contexts == ["parent text A", "parent text B"]


def test_extracted_contexts_fall_back_to_child_chunk(extractor):
    spans = _make_rows(
        ("LangGraph", _langgraph_attrs("NVDA revenue?", "single")),
        ("generate",  _generate_attrs("Revenue was $60.9B")),
        ("retrieve",  _retrieve_attrs(["child text A"], use_parent=False)),
    )
    conn = _mock_conn([1], {1: spans})

    with patch("sqlite3.connect", return_value=conn):
        result = extractor.extract(["single"])

    assert result["single"][0].retrieved_contexts == ["child text A"]


def test_flattens_multiple_retrieve_spans(extractor):
    """Multi-hop: two retrieve spans each contributing one chunk."""
    retrieve_1 = _retrieve_attrs(["hop 1 chunk"], use_parent=False)
    retrieve_2 = _retrieve_attrs(["hop 2 chunk"], use_parent=False)
    spans = _make_rows(
        ("LangGraph", _langgraph_attrs("NVDA multi-hop?", "multi_hop")),
        ("retrieve",  retrieve_1),
        ("generate",  _generate_attrs("Answer")),
        ("retrieve",  retrieve_2),
        ("generate",  _generate_attrs("Final answer")),
    )
    conn = _mock_conn([1], {1: spans})

    with patch("sqlite3.connect", return_value=conn):
        result = extractor.extract(["multi_hop"])

    contexts = result["multi_hop"][0].retrieved_contexts
    assert "hop 1 chunk" in contexts
    assert "hop 2 chunk" in contexts


def test_uses_last_generate_span_for_response(extractor):
    spans = _make_rows(
        ("LangGraph", _langgraph_attrs("NVDA revenue?", "multi_hop")),
        ("retrieve",  _retrieve_attrs(["chunk"])),
        ("generate",  _generate_attrs("Intermediate answer")),
        ("retrieve",  _retrieve_attrs(["chunk 2"])),
        ("generate",  _generate_attrs("Final answer")),
    )
    conn = _mock_conn([1], {1: spans})

    with patch("sqlite3.connect", return_value=conn):
        result = extractor.extract(["multi_hop"])

    assert result["multi_hop"][0].response == "Final answer"


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def test_skips_trace_not_in_requested_datasets(extractor):
    spans = _make_rows(
        ("LangGraph", _langgraph_attrs("Compare NVDA and AMD", "comparison")),
        ("generate",  _generate_attrs("NVDA leads")),
        ("retrieve",  _retrieve_attrs(["chunk"])),
    )
    conn = _mock_conn([1], {1: spans})

    with patch("sqlite3.connect", return_value=conn):
        result = extractor.extract(["single"])  # only want single

    assert result["single"] == []


def test_returns_empty_list_for_dataset_with_no_traces(extractor):
    conn = _mock_conn([], {})

    with patch("sqlite3.connect", return_value=conn):
        result = extractor.extract(["multi_hop"])

    assert result["multi_hop"] == []


def test_multiple_datasets_extracted_in_one_call(extractor):
    spans_1 = _make_rows(
        ("LangGraph", _langgraph_attrs("NVDA revenue?", "single")),
        ("generate",  _generate_attrs("$60.9B")),
        ("retrieve",  _retrieve_attrs(["c1"])),
    )
    spans_2 = _make_rows(
        ("LangGraph", _langgraph_attrs("NVDA multi-hop?", "multi_hop")),
        ("generate",  _generate_attrs("Multi answer")),
        ("retrieve",  _retrieve_attrs(["c2"])),
    )
    conn = _mock_conn([1, 2], {1: spans_1, 2: spans_2})

    with patch("sqlite3.connect", return_value=conn):
        result = extractor.extract(["single", "multi_hop"])

    assert len(result["single"]) == 1
    assert len(result["multi_hop"]) == 1


# ---------------------------------------------------------------------------
# Resilience
# ---------------------------------------------------------------------------

def test_skips_trace_missing_generate_span(extractor):
    spans = _make_rows(
        ("LangGraph", _langgraph_attrs("out of scope?", "out_of_scope")),
    )
    conn = _mock_conn([1], {1: spans})

    with patch("sqlite3.connect", return_value=conn):
        result = extractor.extract(["single"])

    assert result["single"] == []


def test_skips_trace_with_malformed_span_json(extractor):
    spans = _make_rows(
        ("LangGraph", "not valid json"),
        ("generate",  _generate_attrs("Answer")),
    )
    conn = _mock_conn([1], {1: spans})

    with patch("sqlite3.connect", return_value=conn):
        result = extractor.extract(["single"])

    assert result["single"] == []
