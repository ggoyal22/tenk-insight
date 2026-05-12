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
    """Fraction of expected tickers that appear in the output tasks.

    Returns (None, "skipped") when expected_tasks is absent. Out-of-scope entries
    (expected_tasks=[]) score 1.0 only when the output also has no tasks.
    """
    expected_tasks = expected.get("expected_tasks")
    if expected_tasks is None:
        return (None, "skipped")

    if not expected_tasks:
        return 1.0 if not output.get("tasks") else 0.0

    expected_tickers = [
        exp["filter"]["ticker"]
        for exp in expected_tasks
        if exp.get("filter") and exp["filter"].get("ticker")
    ]
    if not expected_tickers:
        return 1.0

    output_tickers = {
        (t.get("filter") or {}).get("ticker")
        for t in output.get("tasks", [])
    } - {None}

    matched = sum(1 for t in expected_tickers if t in output_tickers)
    return matched / len(expected_tickers)
