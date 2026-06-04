"""
Evaluation entry point for the SEC 10-K RAG pipeline.

Two modes:
  * default     — run the generation pipeline over the golden queries, capturing
                  traces in Phoenix, then score those traces with RAGAS.
  * --eval-only — skip generation and score traces already present in Phoenix.

Examples:
    python scripts/evaluate.py
    python scripts/evaluate.py --difficulty easy
    python scripts/evaluate.py --eval-only --since 2h
"""

import argparse
import logging
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.loader import load_config, load_eval_config
from evaluation.types import RunResult

# Heavy third-party stacks (PyTorch/Transformers via the embedder, RAGAS, LangGraph)
# are imported lazily inside main() so that --help and argument errors return
# instantly, and --eval-only skips loading the embedder and generation pipeline
# altogether.


def _load_queries(
    golden_path: str | None, datasets: list[str], difficulty: str | None = None
) -> list[str]:
    """Load query strings from the golden file(s), filtered to the configured datasets.

    When difficulty is given, only entries with a matching difficulty are kept.
    """
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
            and (difficulty is None or e.get("difficulty") == difficulty)
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


def _parse_since(value: str) -> datetime:
    """Parse a duration string like '1h', '30m', '2h30m' into an absolute UTC datetime."""
    match = re.fullmatch(r'(?:(\d+)h)?(?:(\d+)m)?', value)
    if not match or not any(match.groups()):
        raise ValueError(f"Invalid --since value '{value}'. Use e.g. '1h', '30m', '2h30m'.")
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    return datetime.now(timezone.utc) - timedelta(hours=hours, minutes=minutes)


def _wait_for_traces(db_path: str, since: datetime, timeout: int = 30) -> None:
    """Poll Phoenix DB until the generate-span count stops growing (stable for 2s).

    force_flush() ensures spans are sent via HTTP, but Phoenix writes to SQLite
    asynchronously. Polling adapts to however long Phoenix needs rather than
    sleeping a fixed amount.
    """
    logger = logging.getLogger(__name__)
    if not db_path:
        logger.warning("PHOENIX_DB_PATH is not set — skipping the trace-settle wait")
        return
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
    parser = argparse.ArgumentParser(
        description="Run the RAG generation pipeline over golden queries and/or score traces with RAGAS.",
        epilog=(
            "examples:\n"
            "  python scripts/evaluate.py                    # generate answers, then score them\n"
            "  python scripts/evaluate.py --difficulty easy  # only the easy golden queries\n"
            "  python scripts/evaluate.py --eval-only        # score existing traces in Phoenix\n"
            "  python scripts/evaluate.py --eval-only --since 2h\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-e", "--eval-only", action="store_true",
        help="Skip generation and evaluate existing traces in Phoenix",
    )
    parser.add_argument(
        "-s", "--since", metavar="DURATION",
        help="With --eval-only: limit to traces from the last duration (e.g. 1h, 30m, 2h30m)",
    )
    parser.add_argument(
        "-g", "--golden", metavar="PATH",
        help="Path to a golden YAML file or directory (overrides eval.golden_path in config)",
    )
    parser.add_argument(
        "-d", "--difficulty", choices=["easy", "medium", "hard"],
        help="Run only golden queries of this difficulty (ignored with --eval-only)",
    )
    args = parser.parse_args()

    try:
        app_config = load_config()
        eval_config = load_eval_config()
    except Exception as exc:
        # Logging is not configured yet — basicConfig() below needs the log level
        # from the config we just failed to load — so report directly to stderr.
        sys.stderr.write(f"CRITICAL — Failed to load config: {exc}\n")
        sys.exit(1)

    if args.golden:
        eval_config.golden_path = args.golden

    logging.basicConfig(
        level=eval_config.log_level,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    from tracing.setup import setup_tracing
    try:
        setup_tracing(app_config.tracing)
    except RuntimeError as exc:
        logger.critical("Failed to initialise tracing: %s", exc)
        sys.exit(1)

    since: datetime | None

    if args.eval_only:
        if args.difficulty:
            logger.warning("--difficulty is ignored with --eval-only (no queries are executed)")
        if args.since:
            try:
                since = _parse_since(args.since)
            except ValueError as exc:
                logger.critical("%s", exc)
                sys.exit(1)
            logger.info("--eval-only: evaluating traces since %s", since.isoformat())
        else:
            since = None
            logger.info("--eval-only: evaluating all traces in Phoenix")
    else:
        if args.since:
            logger.warning("--since is ignored without --eval-only (a fresh generation run is timestamped automatically)")

        queries = _load_queries(eval_config.golden_path, eval_config.datasets, args.difficulty)
        if not queries:
            difficulty_note = f" at difficulty '{args.difficulty}'" if args.difficulty else ""
            logger.critical(
                "No golden queries found for datasets %s%s in '%s'",
                eval_config.datasets, difficulty_note, eval_config.golden_path,
            )
            sys.exit(1)

        from db.factory import create_db_client

        client = create_db_client(app_config)
        if not client.health_check():
            logger.critical("Database health check failed")
            client.close()
            sys.exit(1)

        try:
            logger.info("Loading the embedding model and generation pipeline — this can take a few seconds...")
            from etl.factory import create_embedder
            from generation.factory import build_generation_pipeline, make_initial_state

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
            logger.critical("Execution phase failed", exc_info=True)
            sys.exit(1)
        finally:
            client.close()

        from tracing.setup import flush_spans

        flush_spans()
        _wait_for_traces(os.environ.get("PHOENIX_DB_PATH", ""), since)

    logger.info("Loading evaluation libraries (RAGAS)...")
    from evaluation.factory import build_evaluator, build_exporters, build_extractor
    from evaluation.golden_loader import load_golden
    from evaluation.runner import EvaluationRunner

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
        logger.critical("Evaluation phase failed", exc_info=True)
        sys.exit(1)

    _print_summary(result)


if __name__ == "__main__":
    main()
