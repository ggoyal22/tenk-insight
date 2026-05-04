import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.loader import load_eval_config
from evaluation.factory import build_evaluator, build_exporters, build_extractor
from evaluation.golden_loader import load_golden
from evaluation.runner import EvaluationRunner
from evaluation.types import RunResult


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


def main() -> None:
    try:
        config = load_eval_config()
    except Exception as exc:
        sys.stderr.write(f"CRITICAL — Failed to load evaluation config: {exc}\n")
        sys.exit(1)

    logging.basicConfig(
        level=config.log_level,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    golden = load_golden(config.golden_path)

    extractor = build_extractor(config)
    evaluator = build_evaluator(config)
    exporters = build_exporters(config)

    runner = EvaluationRunner(
        extractor=extractor,
        evaluator=evaluator,
        exporters=exporters,
        golden=golden,
    )

    try:
        result = runner.run(config)
    except Exception:
        logging.critical("Evaluation run failed", exc_info=True)
        sys.exit(1)

    _print_summary(result)


if __name__ == "__main__":
    main()
