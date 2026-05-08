"""Node-level evaluation using Phoenix Experiments.

Runs one or both of classify_query / plan_tasks against golden fixtures and
scores outputs with code-based evaluators — no judge LLM required.

Metrics written to Phoenix:
  query_type_accuracy  — correct classification (classify, both)
  task_count_match     — correct number of retrieval tasks (plan, both; fixtures with expected_tasks)
  task_filter_recall   — expected filters captured in output (plan, both; fixtures with expected_tasks)

Usage:
    python scripts/eval_node.py [--node classify|plan|both] [--golden data/golden] [--name NAME]

    --node classify  Run only classify_query; measures query_type_accuracy.
    --node plan      Run only plan_tasks using fixture query_type; measures task quality.
    --node both      Chain classify → plan (default); measures all three metrics.

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
from evaluation.node_eval.evaluators import (
    query_type_accuracy,
    task_count_match,
    task_filter_recall,
)
from generation.nodes.classify import make_classify_query
from generation.nodes.plan import make_plan_tasks
from generation.prompt_registry import load_prompts
from llm.factory import build_llm
from retrieval.types import MetadataFilter


def _make_passthrough_filings_repo() -> MagicMock:
    """Stub repo that treats every ticker as indexed — skips DB check in plan_tasks."""
    repo = MagicMock()
    repo.list_ids.return_value = [uuid.uuid4()]
    return repo


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate classify_query and/or plan_tasks nodes via Phoenix Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--node",
        choices=["classify", "plan", "both"],
        default="both",
        help="Which node(s) to evaluate (default: both)",
    )
    parser.add_argument("--golden", default="data/golden", help="Path to golden dataset dir or file")
    parser.add_argument("--name", help="Experiment name in Phoenix (default: derived from --node)")
    args = parser.parse_args()

    experiment_name = args.name or f"{args.node}_eval"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    config = load_config()
    llm = build_llm(config.llm)
    prompts = load_prompts(tag=config.prompts.tag)

    plan_prompts = {
        "single": prompts.plan_single,
        "comparison": prompts.plan_comparison,
        "time_series": prompts.plan_time_series,
        "multi_hop": prompts.plan_multi_hop,
    }

    classify_node = make_classify_query(llm, prompts.classify) if args.node in ("classify", "both") else None
    plan_node = make_plan_tasks(llm, plan_prompts, _make_passthrough_filings_repo()) if args.node in ("plan", "both") else None

    fixtures = load_node_fixtures(args.golden)
    if not fixtures:
        sys.stderr.write(f"No fixtures found at '{args.golden}'\n")
        sys.exit(1)
    logger.info("Loaded %d fixture(s)", len(fixtures))

    base_url = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:6006")
    client = Client(base_url=base_url)

    # query_type goes in input so plan-only mode can read it directly from the fixture.
    examples = [
        {
            "input": {
                "query": f["query"],
                "query_type": f["query_type"],
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
        }

        if args.node == "classify":
            result = classify_node(state)
            return {"query_type": result["query_type"], "tasks": []}

        if args.node == "plan":
            query_type = input["query_type"]
            state["query_type"] = query_type
            state["resolved_query"] = input["query"]
            pending_tasks = _run_plan(plan_node, state, query_type)
            return {"query_type": query_type, "tasks": _serialise_tasks(pending_tasks)}

        # both: classify first, then plan
        classify_result = classify_node(state)
        state = {**state, **classify_result}
        pending_tasks = _run_plan(plan_node, state, state.get("query_type", ""))
        return {
            "query_type": state["query_type"],
            "tasks": _serialise_tasks(pending_tasks),
        }

    evaluators = _select_evaluators(args.node)
    metadata = _build_metadata(args.node, prompts, plan_prompts)

    run_experiment(
        dataset=dataset,
        task=task,
        evaluators=evaluators,
        experiment_name=experiment_name,
        experiment_metadata=metadata,
        client=client,
    )


def _run_plan(plan_node, state: dict, query_type: str) -> list:
    """Call plan_node and return pending_tasks, or [] for out_of_scope / canned answers."""
    if query_type in ("out_of_scope", "") or state.get("answer"):
        return []
    result = plan_node(state)
    return result.get("pending_tasks", [])


def _serialise_tasks(tasks: list) -> list[dict]:
    return [
        {
            "query": t.query,
            "filter": dataclasses.asdict(t.filter) if t.filter else None,
        }
        for t in tasks
    ]


def _select_evaluators(node: str) -> list:
    if node == "classify":
        return [query_type_accuracy]
    if node == "plan":
        return [task_count_match, task_filter_recall]
    return [query_type_accuracy, task_count_match, task_filter_recall]


def _build_metadata(node: str, prompts, plan_prompts: dict) -> dict:
    meta: dict = {"node": node}
    if node in ("classify", "both"):
        meta["classify_prompt_version"] = prompts.versions.get("classify")
    if node in ("plan", "both"):
        meta["plan_prompt_versions"] = {k: prompts.versions.get(f"plan_{k}") for k in plan_prompts}
    return meta


def _build_filter(d: dict | None) -> MetadataFilter | None:
    if not d:
        return None
    from datetime import date
    fy = d.get("fiscal_year_end")
    return MetadataFilter(
        ticker=d.get("ticker"),
        form_type=d.get("form_type"),
        fiscal_year_end=date.fromisoformat(fy) if isinstance(fy, str) else fy,
        section=d.get("section"),
    )


if __name__ == "__main__":
    main()
