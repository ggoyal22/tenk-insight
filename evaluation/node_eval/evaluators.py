"""Code-based evaluators for node-level Phoenix experiments.

All evaluators follow the Phoenix convention: named arguments are bound to
special values (output = task result, expected = dataset 'output' field).
Return type may be bool, float, or None (None = skip this metric for this example).
"""


def query_type_accuracy(output: dict, expected: dict) -> bool:
    """Exact match on query_type classification."""
    return output.get("query_type") == expected.get("query_type")


def task_count_match(output: dict, expected: dict) -> tuple | bool:
    """Correct number of retrieval tasks generated.

    Returns (None, "skipped") when expected_tasks is absent from the fixture —
    Phoenix stores a label with no score, leaving the metric blank for that example.
    """
    expected_tasks = expected.get("expected_tasks")
    if expected_tasks is None:
        return (None, "skipped")
    return len(output.get("tasks", [])) == len(expected_tasks)


def task_filter_recall(output: dict, expected: dict) -> tuple | float:
    """Fraction of expected task filters that appear in the output tasks.

    Returns (None, "skipped") when expected_tasks is absent. Out-of-scope entries
    (expected_tasks=[]) score 1.0 only when the output also has no tasks.

    Matching is partial: only non-None fields in the expected filter must match.
    e.g. expected {ticker: AAPL} matches output {ticker: AAPL, form_type: 10-K}.
    """
    expected_tasks = expected.get("expected_tasks")
    if expected_tasks is None:
        return (None, "skipped")

    if not expected_tasks:
        return 1.0 if not output.get("tasks") else 0.0

    output_filters = [t.get("filter") for t in output.get("tasks", [])]
    matched = sum(
        any(_filter_matches(exp["filter"], out_f) for out_f in output_filters)
        for exp in expected_tasks
        if exp.get("filter")
    )
    total = sum(1 for exp in expected_tasks if exp.get("filter"))
    return matched / total if total else 1.0


def _filter_matches(expected: dict | None, actual: dict | None) -> bool:
    if not expected:
        return True
    if actual is None:
        return False
    return all(actual.get(k) == v for k, v in expected.items() if v is not None)
