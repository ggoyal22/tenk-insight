"""Node-level evaluation using Phoenix Experiments.

Runs the analyze_query node against golden fixtures and scores outputs with
code-based evaluators — no judge LLM required.

Metrics written to Phoenix:
  query_type_accuracy  — correct classification (all fixtures)
  task_count_match     — correct number of retrieval tasks (fixtures with expected_tasks)
  task_filter_recall   — expected filters captured in output (fixtures with expected_tasks)

Usage:
    python scripts/eval_node.py [--golden data/golden] [--name EXPERIMENT_NAME]

Environment:
    OTEL_EXPORTER_OTLP_ENDPOINT — Phoenix base URL (e.g. http://localhost:6006)
    Plus standard LLM/config env vars consumed by load_config().
"""

import argparse
import dataclasses
import logging
import os
import sys
from pathlib import Path

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
from generation.nodes.analyze import make_analyze_query
from generation.prompt_registry import load_prompts
from llm.factory import build_llm
from retrieval.types import MetadataFilter


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the analyze_query node via Phoenix Experiments")
    parser.add_argument("--golden", default="data/golden", help="Path to golden dataset dir or file")
    parser.add_argument("--name", default="analyze_query_eval", help="Experiment name in Phoenix")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    config = load_config()
    llm = build_llm(config.llm)
    prompts = load_prompts(tag=config.prompts.tag)
    node = make_analyze_query(llm, prompts.query_analysis)

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
    dataset = client.datasets.create_dataset(name="analyze_query_eval", examples=examples)
    logger.info("Uploaded %d example(s) to Phoenix dataset", len(examples))

    def task(input):
        state = {
            "query": input["query"],
            "query_filter": _build_filter(input.get("query_filter")),
            "history": [],
        }
        result = node(state)
        return {
            "query_type": result["query_type"],
            "tasks": [
                {
                    "query": t.query,
                    "filter": dataclasses.asdict(t.filter) if t.filter else None,
                }
                for t in result["pending_tasks"]
            ],
        }

    run_experiment(
        dataset=dataset,
        task=task,
        evaluators=[query_type_accuracy, task_count_match, task_filter_recall],
        experiment_name=args.name,
        experiment_metadata={"prompt_version": prompts.versions.get("query_analysis")},
        client=client,
    )


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
