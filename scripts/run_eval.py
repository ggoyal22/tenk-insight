import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.loader import load_config, load_eval_config
from db.factory import create_db_client
from etl.factory import create_embedder
from evaluation.factory import build_evaluator, build_exporters, build_extractor
from evaluation.golden_loader import load_golden
from evaluation.runner import EvaluationRunner
from evaluation.types import RunResult
from generation.factory import build_generation_pipeline, make_initial_state
from tracing.setup import flush_spans, setup_tracing


def _load_queries(golden_path: str | None, datasets: list[str]) -> list[str]:
    """Load query strings from the golden file(s), filtered to the configured datasets."""
    if not golden_path:
        return []
    p = Path(golden_path)
    if not p.exists():
        return []
    files = [p] if p.is_file() else sorted(p.glob("*.yaml"))
    dataset_set = set(datasets)
    queries: list[str] = []
    for f in files:
        with open(f) as fh:
            entries = yaml.safe_load(fh) or []
        queries.extend(
            str(e["query"]).strip()
            for e in entries
            if isinstance(e, dict)
            and e.get("query_type") in dataset_set
            and e.get("answer") is not None
        )
    return queries


def _print_summary(result: RunResult) -> None:
    golden_label = "yes" if result.golden_used else "no"
    print(f"\nRun ID: {result.run_id}  |  git: {result.git_sha[:8]}  |  golden: {golden_label}\n")

    if not result.datasets:
        print("  No datasets evaluated — no traces found.")
        return

    all_metrics = sorted({m for ds in result.datasets for m in ds.scores})
    col_w = 16
    header = f"  {'Dataset':<12}  {'Samples':>7}  " + "  ".join(f"{m:>{col_w}}" for m in all_metrics)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for ds in result.datasets:
        row = f"  {ds.dataset:<12}  {ds.n_samples:>7}  "
        row += "  ".join(
            f"{ds.scores[m]:>{col_w}.4f}" if m in ds.scores else f"{'—':>{col_w}}"
            for m in all_metrics
        )
        print(row)
    print()


def _wait_for_traces(db_path: str, since: datetime, timeout: int = 30) -> None:
    """Poll Phoenix DB until the generate-span count stops growing (stable for 2s).

    force_flush() ensures spans are sent via HTTP, but Phoenix writes to SQLite
    asynchronously. Polling adapts to however long Phoenix needs rather than
    sleeping a fixed amount.
    """
    logger = logging.getLogger(__name__)
    since_str = since.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    sql = (
        "SELECT COUNT(DISTINCT trace_rowid) FROM spans"
        " WHERE name = 'generate' AND datetime(start_time) >= datetime(?)"
    )
    last_count = -1
    stable_ticks = 0
    for _ in range(timeout):
        time.sleep(1)
        with sqlite3.connect(db_path) as conn:
            count = conn.execute(sql, (since_str,)).fetchone()[0]
        if count == last_count:
            stable_ticks += 1
            if stable_ticks >= 2:
                logger.info("Traces stabilised at %d — proceeding to evaluation", count)
                return
        else:
            last_count = count
            stable_ticks = 0
    logger.warning("Timed out after %ds waiting for traces — proceeding with %d trace(s)", timeout, last_count)


def main() -> None:
    try:
        app_config = load_config()
        eval_config = load_eval_config()
    except Exception as exc:
        sys.stderr.write(f"CRITICAL — Failed to load config: {exc}\n")
        sys.exit(1)

    logging.basicConfig(
        level=eval_config.log_level,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    try:
        setup_tracing(app_config.tracing)
    except RuntimeError as exc:
        sys.stderr.write(f"CRITICAL — Failed to initialise tracing: {exc}\n")
        sys.exit(1)

    queries = _load_queries(eval_config.golden_path, eval_config.datasets)
    if not queries:
        sys.stderr.write(
            f"CRITICAL — No golden queries found for datasets {eval_config.datasets} "
            f"in '{eval_config.golden_path}'\n"
        )
        sys.exit(1)

    client = create_db_client(app_config)
    if not client.health_check():
        sys.stderr.write("CRITICAL — Database health check failed\n")
        client.close()
        sys.exit(1)

    try:
        embedder = create_embedder(app_config)
        graph = build_generation_pipeline(app_config, client, embedder)

        since = datetime.now(timezone.utc)
        logger.info(
            "Executing %d queries across datasets %s", len(queries), eval_config.datasets
        )

        for i, query in enumerate(queries, 1):
            logger.info("[%d/%d] %s", i, len(queries), query)
            try:
                graph.invoke(make_initial_state(query))
            except Exception:
                logger.warning("Query failed, skipping: %s", query, exc_info=True)

    except Exception:
        logging.critical("Execution phase failed", exc_info=True)
        sys.exit(1)
    finally:
        client.close()

    flush_spans()
    _wait_for_traces(os.environ.get("PHOENIX_DB_PATH", ""), since)

    golden = load_golden(eval_config.golden_path)
    runner = EvaluationRunner(
        extractor=build_extractor(eval_config),
        evaluator=build_evaluator(eval_config),
        exporters=build_exporters(eval_config),
        golden=golden,
    )

    try:
        result = runner.run(eval_config, since=since)
    except Exception:
        logging.critical("Evaluation phase failed", exc_info=True)
        sys.exit(1)

    _print_summary(result)


if __name__ == "__main__":
    main()
