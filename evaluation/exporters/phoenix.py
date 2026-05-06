"""Phoenix trace annotation exporter.

Writes evaluation results back to the Phoenix SQLite database using two tables:

  evaluation_runs   — one row per evaluation run (run_id, git_sha, extractor,
                      evaluator, golden_used). Makes run metadata queryable
                      alongside trace annotations without repeating it per row.

  span_annotations  — one row per metric per trace. Links to evaluation_runs
                      via run_id. Annotations appear in the Phoenix UI under
                      the Annotations tab for each trace.

Both tables are written in a single transaction. If a metric is absent from a
sample's scores (e.g. context_recall was skipped), no annotation is written
for that metric on that trace.
"""

import logging
import sqlite3
from datetime import datetime, timezone

from evaluation.exporters.base import BaseResultExporter
from evaluation.types import EvalSample, EvaluationResult, RunResult

logger = logging.getLogger(__name__)



class PhoenixTraceAnnotationExporter(BaseResultExporter):
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    def export(
        self,
        result: RunResult,
        samples: list[EvalSample],
        evaluation: EvaluationResult,
    ) -> None:
        if not evaluation.scores:
            logger.warning("No per-sample scores to annotate — skipping Phoenix export")
            return

        annotation_rows = self._build_annotation_rows(result.run_id, samples, evaluation)
        if not annotation_rows:
            logger.warning("No annotation rows produced — skipping Phoenix export")
            return

        with sqlite3.connect(self._db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode=WAL")
            self._ensure_schema(conn)
            conn.execute(
                """
                INSERT OR REPLACE INTO evaluation_runs
                    (run_id, git_sha, extractor, evaluator, golden_used, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    result.run_id,
                    result.git_sha,
                    result.extractor,
                    result.evaluator,
                    int(result.golden_used),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.executemany(
                """
                INSERT OR REPLACE INTO span_annotations
                    (span_rowid, name, score, annotator_kind, source, metadata, identifier)
                VALUES (?, ?, ?, 'CODE', 'APP', '{}', ?)
                """,
                annotation_rows,
            )
            conn.commit()

        logger.info(
            "Wrote %d annotation(s) for run '%s' to Phoenix DB at '%s'",
            len(annotation_rows), result.run_id, self._db_path,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_annotation_rows(
        run_id: str,
        samples: list[EvalSample],
        evaluation: EvaluationResult,
    ) -> list[tuple]:
        rows = []
        for sample, sample_scores in zip(samples, evaluation.scores):
            for metric_name, score in sample_scores.items():
                rows.append((sample.trace_id, metric_name, score, run_id))
        return rows

    @staticmethod
    def _ensure_schema(conn: sqlite3.Connection) -> None:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_runs (
                run_id      TEXT PRIMARY KEY,
                git_sha     TEXT NOT NULL,
                extractor   TEXT NOT NULL,
                evaluator   TEXT NOT NULL,
                golden_used INTEGER NOT NULL,
                created_at  TEXT NOT NULL
            )
        """)
