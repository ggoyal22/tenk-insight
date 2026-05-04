"""Phoenix trace annotation exporter.

Writes per-sample evaluation scores back to the Phoenix SQLite database as
span annotations on the root span of each trace. Annotations appear in the
Phoenix UI under the Annotations tab for each trace.

One annotation record is written per metric per trace. If a metric is absent
from a sample's scores (e.g. context_recall was skipped for that sample),
no annotation is written for that metric on that trace.
"""

import json
import logging
import sqlite3

from evaluation.exporters.base import BaseResultExporter
from evaluation.types import EvalSample, EvaluationResult, RunResult

logger = logging.getLogger(__name__)

_ANNOTATION_SOURCE = "ragas_evaluation"


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

        rows = self._build_annotation_rows(result.run_id, samples, evaluation)
        if not rows:
            logger.warning("No annotation rows produced — skipping Phoenix export")
            return

        with sqlite3.connect(self._db_path) as conn:
            self._ensure_annotations_table(conn)
            conn.executemany(
                """
                INSERT OR REPLACE INTO span_annotations
                    (span_id, run_id, name, score, source)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()

        logger.info(
            "Wrote %d annotation(s) for run '%s' to Phoenix DB at '%s'",
            len(rows), result.run_id, self._db_path,
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
                rows.append((
                    sample.trace_id,
                    run_id,
                    metric_name,
                    score,
                    _ANNOTATION_SOURCE,
                ))
        return rows

    @staticmethod
    def _ensure_annotations_table(conn: sqlite3.Connection) -> None:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS span_annotations (
                span_id TEXT NOT NULL,
                run_id  TEXT NOT NULL,
                name    TEXT NOT NULL,
                score   REAL NOT NULL,
                source  TEXT NOT NULL,
                PRIMARY KEY (span_id, run_id, name)
            )
        """)
