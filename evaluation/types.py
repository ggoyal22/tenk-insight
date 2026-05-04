"""
Core data types shared across the evaluation pipeline.

EvalSample   — one trace worth of input to the evaluator
DatasetScores — aggregated metric scores for one query-type group
RunResult    — full output of one evaluation run, written by result writers
"""

from dataclasses import dataclass, field


@dataclass
class EvaluationResult:
    scores: list[dict[str, float]]  # one score dict per EvalSample; used for Phoenix span annotations
    aggregate: dict[str, float]     # mean of scores across all samples; used by DatasetScores


@dataclass
class EvalSample:
    trace_id:            str
    query_type:          str
    user_input:          str
    retrieved_contexts:  list[str]
    response:            str
    # reference=None → reference-required metrics (context_recall, answer_correctness)
    # are skipped automatically for this sample
    reference:           str | None = None


@dataclass
class DatasetScores:
    dataset:   str               # query_type group, e.g. "single", "multi_hop"
    scores:    dict[str, float]  # metric name → mean score across all samples
    n_samples: int               # number of traces evaluated


@dataclass
class RunResult:
    run_id:      str                      # ISO-8601 timestamp, e.g. "2026-05-04T10:23:11"
    git_sha:     str                      # git rev-parse HEAD; "unknown" if not in a repo
    extractor:   str                      # backend used, e.g. "phoenix"
    evaluator:   str                      # backend used, e.g. "ragas"
    golden_used: bool                     # True if any sample had a non-None reference
    datasets:    list[DatasetScores] = field(default_factory=list)
