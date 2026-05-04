"""
Golden dataset loader for evaluation.

Provides ground-truth Q&A pairs used to populate EvalSample.reference,
which unlocks reference-required RAGAS metrics (context_recall, answer_correctness).

golden_path may point to a single YAML file or a directory of YAML files.
A directory is merged into one lookup dict; duplicate queries across files
raise ValueError so ambiguous reference answers fail loudly rather than
silently picking one.
"""

import logging
from pathlib import Path

import yaml

from evaluation.types import EvalSample

logger = logging.getLogger(__name__)


def load_golden(path: str | None) -> dict[str, str]:
    """Load golden Q&A pairs and return {query: reference_answer}.

    path=None or a missing path → returns {} with a warning.
    path is a file          → loads that file.
    path is a directory     → loads and merges all *.yaml files inside.
    Duplicate queries across files raise ValueError.
    """
    if path is None:
        logger.warning(
            "golden_path is not set — reference-required metrics will be skipped"
        )
        return {}

    p = Path(path)
    if not p.exists():
        logger.warning(
            "golden_path '%s' does not exist — reference-required metrics will be skipped",
            path,
        )
        return {}

    if p.is_file():
        return _load_file(p)

    if p.is_dir():
        files = sorted(p.glob("*.yaml"))
        if not files:
            logger.warning(
                "golden_path directory '%s' contains no .yaml files — "
                "reference-required metrics will be skipped",
                path,
            )
            return {}
        merged: dict[str, str] = {}
        for f in files:
            entries = _load_file(f)
            duplicates = merged.keys() & entries.keys()
            if duplicates:
                raise ValueError(
                    f"Duplicate golden queries found when loading '{f.name}': "
                    f"{sorted(duplicates)}. Each query must appear in exactly one file."
                )
            merged.update(entries)
        logger.info(
            "Loaded %d golden entries from %d file(s) in '%s'",
            len(merged), len(files), path,
        )
        return merged

    raise ValueError(f"golden_path '{path}' is neither a file nor a directory.")


def attach_references(samples: list[EvalSample], golden: dict[str, str]) -> None:
    """Set sample.reference for any sample whose user_input matches a golden entry.

    Matching is exact string equality. Samples without a match keep reference=None,
    which causes RAGAS to skip reference-required metrics for that sample only.
    Mutates samples in-place.
    """
    if not golden:
        return
    matched = 0
    for sample in samples:
        ref = golden.get(sample.user_input)
        if ref is not None:
            sample.reference = ref
            matched += 1
    if matched == 0 and samples:
        logger.warning(
            "Attached references to 0 / %d sample(s) — no golden queries matched; "
            "check for exact-string mismatches between golden file and trace queries",
            len(samples),
        )
    else:
        logger.info("Attached references to %d / %d sample(s)", matched, len(samples))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_file(path: Path) -> dict[str, str]:
    """Parse one golden YAML file and return {query: answer}."""
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse golden file '{path}': {exc}") from exc

    if not isinstance(data, list):
        raise ValueError(
            f"Golden file '{path}' must contain a YAML list of entries, "
            f"got {type(data).__name__}."
        )

    result: dict[str, str] = {}
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Golden file '{path}': entry {i} must be a dict, "
                f"got {type(entry).__name__}."
            )
        missing = [f for f in ("query", "answer", "query_type") if f not in entry]
        if missing:
            raise ValueError(
                f"Golden file '{path}': entry {i} is missing required field(s) "
                f"{missing}. Got keys: {list(entry.keys())}."
            )
        query = str(entry["query"]).strip()
        if not query:
            raise ValueError(
                f"Golden file '{path}': entry {i} has an empty 'query' field."
            )
        if query in result:
            raise ValueError(
                f"Golden file '{path}': duplicate query at entry {i}: '{query}'."
            )
        result[query] = str(entry["answer"])

    logger.info("Loaded %d golden entries from '%s'", len(result), path)
    return result
