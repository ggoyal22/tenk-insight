"""Node-level evaluation using Phoenix Experiments.

Runs analyze_query against golden fixtures and scores outputs with code-based
evaluators — no judge LLM required.

Metrics written to Phoenix:
  query_type_accuracy  — correct classification
  task_count_match     — correct number of retrieval tasks (fixtures with expected_tasks only)
  task_filter_recall   — expected filters captured in output (fixtures with expected_tasks only)

Usage:
    python scripts/eval_node.py [--golden data/golden] [--name NAME]

Environment:
    OTEL_EXPORTER_OTLP_ENDPOINT — Phoenix base URL (e.g. http://localhost:6006)
    Plus standard LLM/config env vars consumed by load_config().
"""

import argparse
import dataclasses
import logging
import os
import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phoenix.client import Client
from phoenix.client.experiments import run_experiment

from config.loader import load_config
from evaluation.golden_loader import load_node_fixtures
from tracing.setup import flush_spans, setup_tracing
from evaluation.node_eval.evaluators import (
    query_type_accuracy,
    task_count_match,
    task_filter_recall,
)
from generation.nodes.analyze_query import make_analyze_query
from generation.prompt_registry import load_prompts
from llm.factory import build_llm
from retrieval.types import MetadataFilter


def _make_passthrough_filings_repo() -> MagicMock:
    """Stub repo that treats every ticker as indexed — skips DB check in analyze_query."""
    repo = MagicMock()
    repo.list_ids.return_value = [uuid.uuid4()]
    return repo


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate analyze_query node via Phoenix Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--golden", default="data/golden", help="Path to golden dataset dir or file")
    parser.add_argument("--name", help="Experiment name in Phoenix (default: analyze_eval)")
    args = parser.parse_args()

    experiment_name = args.name or "analyze_eval"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    config = load_config()
    setup_tracing(config.tracing)
    llm = build_llm(config.llm)
    prompts = load_prompts(tag=config.prompts.tag)

    analyze_node = make_analyze_query(llm, prompts.analyze, _make_passthrough_filings_repo())

    fixtures = load_node_fixtures(args.golden)
    if not fixtures:
        sys.stderr.write(f"No fixtures found at '{args.golden}'\n")
        sys.exit(1)
    logger.info("Loaded %d fixture(s)", len(fixtures))

    base_url = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:6006")
    client = Client(base_url=base_url)

    examples = [
        {
            "input": {
                "query": f["query"],
                "query_filter": f.get("query_filter"),
                "history": f.get("history", []),
            },
            "output": {
                "query_type": f["query_type"],
                "expected_tasks": f.get("expected_tasks"),
            },
        }
        for f in fixtures
    ]
    dataset = client.datasets.create_dataset(name=f"{experiment_name}_dataset", examples=examples)
    logger.info("Uploaded %d example(s) to Phoenix dataset", len(examples))

    def task(input):
        state = {
            "query": input["query"],
            "query_filter": _build_filter(input.get("query_filter")),
            "history": input.get("history") or [],
            "resolved_query": None,
            "query_type": "",
            "pending_tasks": [],
            "completed_results": [],
            "pipeline_usage": [],
            "hop_count": 0,
            "reflection_count": 0,
            "retrieval_triggered_by": "analysis",
            "answer": None,
        }
        result = analyze_node(state)
        return {
            "query_type": result.get("query_type", ""),
            "tasks": _serialise_tasks(result.get("pending_tasks") or []),
        }

    metadata = {"analyze_prompt_version": prompts.versions.get("analyze")}

    ran = run_experiment(
        dataset=dataset,
        task=task,
        evaluators=[query_type_accuracy, task_count_match, task_filter_recall],
        experiment_name=experiment_name,
        experiment_description=f"golden: {args.golden}",
        experiment_metadata=metadata,
        client=client,
    )
    flush_spans()
    _print_mismatches(ran, dataset)


def _print_mismatches(ran, dataset) -> None:
    example_by_id = {ex["id"]: ex for ex in dataset.examples}
    run_by_id = {run["id"]: run for run in ran["task_runs"]}

    rows = []
    for eval_run in ran["evaluation_runs"]:
        result = eval_run.result
        if isinstance(result, list):
            result = result[0] if result else None
        if not result:
            continue
        score = result.get("score")
        if score is None or score >= 1.0:
            continue

        task_run = run_by_id.get(eval_run.experiment_run_id)
        if not task_run:
            continue
        example = example_by_id.get(task_run["dataset_example_id"])
        if not example:
            continue

        query = example["input"].get("query", "?")
        expected_out = example["output"]
        actual_out = task_run.get("output") or {}

        metric = eval_run.name
        if metric == "query_type_accuracy":
            expected = expected_out.get("query_type", "?")
            actual = actual_out.get("query_type", "?")
        elif metric == "task_count_match":
            n_exp = len(expected_out.get("expected_tasks") or [])
            n_got = len(actual_out.get("tasks") or [])
            expected = f"{n_exp} task(s)"
            actual = f"{n_got} task(s)"
        else:  # task_filter_recall
            expected = "1.00"
            actual = f"{score:.2f}"

        rows.append((metric, query, expected, actual))

    total = len(ran["task_runs"])
    if not rows:
        print(f"\nAll {total} example(s) passed.")
        return

    c1 = max(len("Metric"), max(len(r[0]) for r in rows))
    c2 = max(len("Query"), max(len(r[1]) for r in rows))
    c3 = max(len("Expected"), max(len(r[2]) for r in rows))
    c4 = max(len("Got"), max(len(r[3]) for r in rows))
    sep = f"  {'─' * c1}  {'─' * c2}  {'─' * c3}  {'─' * c4}"

    unique_queries = len({r[1] for r in rows})
    print(f"\nMismatches ({unique_queries} of {total} example(s), {len(rows)} metric failure(s))")
    print(sep)
    print(f"  {'Metric':<{c1}}  {'Query':<{c2}}  {'Expected':<{c3}}  {'Got':<{c4}}")
    print(sep)
    for metric, query, expected, actual in rows:
        print(f"  {metric:<{c1}}  {query:<{c2}}  {expected:<{c3}}  {actual:<{c4}}")
    print(sep)


def _serialise_tasks(tasks: list) -> list[dict]:
    return [
        {
            "keyword_query": t.keyword_query,
            "semantic_query": t.semantic_query,
            "filter": dataclasses.asdict(t.filter) if t.filter else None,
        }
        for t in tasks
    ]


def _build_filter(d: dict | None) -> MetadataFilter | None:
    if not d:
        return None
    fy = d.get("fiscal_year")
    return MetadataFilter(
        ticker=d.get("ticker"),
        form_type=d.get("form_type"),
        fiscal_year=int(fy) if fy is not None else None,
        section=d.get("section"),
    )


if __name__ == "__main__":
    main()
