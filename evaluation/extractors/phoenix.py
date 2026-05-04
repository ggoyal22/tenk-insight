import json
import logging
import sqlite3
from collections import defaultdict

from evaluation.extractors.base import BaseExtractor
from evaluation.types import EvalSample

logger = logging.getLogger(__name__)


class PhoenixExtractor(BaseExtractor):
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    def extract(self, datasets: list[str]) -> dict[str, list[EvalSample]]:
        result: dict[str, list[EvalSample]] = {ds: [] for ds in datasets}

        with sqlite3.connect(self._db_path) as conn:
            trace_ids = self._complete_trace_ids(conn)
            if not trace_ids:
                logger.warning("No complete traces found in Phoenix DB at %s", self._db_path)
                return result

            for trace_rowid in trace_ids:
                sample = self._extract_sample(conn, trace_rowid)
                if sample is None:
                    continue
                if sample.query_type not in datasets:
                    continue
                result[sample.query_type].append(sample)

        for ds, samples in result.items():
            if not samples:
                logger.warning("No traces found for dataset '%s'", ds)
            else:
                logger.info("Extracted %d sample(s) for dataset '%s'", len(samples), ds)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _complete_trace_ids(self, conn: sqlite3.Connection) -> list[int]:
        """Return trace_rowids for traces that reached the generate node.

        Out-of-scope queries terminate before generate and are excluded —
        RAGAS metrics require both a response and retrieved contexts.
        """
        cur = conn.execute("SELECT DISTINCT trace_rowid FROM spans WHERE name = 'generate'")
        return [row[0] for row in cur.fetchall()]

    def _extract_sample(self, conn: sqlite3.Connection, trace_rowid: int) -> EvalSample | None:
        cur = conn.execute(
            "SELECT name, attributes FROM spans WHERE trace_rowid = ? ORDER BY start_time",
            (trace_rowid,),
        )
        # Collect all spans grouped by name; preserves insertion order (start_time ASC)
        # so generate[-1] is always the final generation node.
        spans: dict[str, list[dict]] = defaultdict(list)
        for name, attrs_json in cur.fetchall():
            try:
                spans[name].append(json.loads(attrs_json))
            except (json.JSONDecodeError, TypeError):
                logger.warning("Skipping malformed span '%s' in trace %s", name, trace_rowid)
                continue

        if not spans.get("LangGraph") or not spans.get("generate"):
            logger.debug("Skipping trace %s — missing root or generate span", trace_rowid)
            return None

        try:
            root_input = json.loads(spans["LangGraph"][0]["input"]["value"])
            query_type: str = root_input.get("query_type", "")
            user_input: str = root_input["query"]

            response_output = json.loads(spans["generate"][-1]["output"]["value"])
            response: str = response_output["answer"]["answer"]

            contexts = self._extract_contexts(spans.get("retrieve", []))
        except (KeyError, json.JSONDecodeError, TypeError) as exc:
            logger.warning("Skipping trace %s — failed to parse spans: %s", trace_rowid, exc)
            return None

        # Derive trace_id from the span's own span_id on the root span for a stable identifier
        root_span_id = spans["LangGraph"][0].get("span_id", str(trace_rowid))

        return EvalSample(
            trace_id=root_span_id,
            query_type=query_type,
            user_input=user_input,
            retrieved_contexts=contexts,
            response=response,
            reference=None,
        )

    @staticmethod
    def _extract_contexts(retrieve_spans: list[dict]) -> list[str]:
        """Flatten all retrieve spans across all hops into a single context list.

        Uses parent_chunk.text when available (the larger window the LLM sees),
        falling back to chunk.text.
        """
        contexts: list[str] = []
        for span_attrs in retrieve_spans:
            try:
                output = json.loads(span_attrs["output"]["value"])
                for hop in output.get("completed_results", []):
                    for r in hop:
                        node = r.get("parent_chunk") or r.get("chunk")
                        if node and node.get("text"):
                            contexts.append(node["text"])
            except (KeyError, json.JSONDecodeError, TypeError):
                continue
        return contexts
