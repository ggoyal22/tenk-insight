import json
import logging
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone

from evaluation.extractors.base import BaseExtractor
from evaluation.types import EvalSample

logger = logging.getLogger(__name__)


class PhoenixExtractor(BaseExtractor):
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    def extract(
        self, datasets: list[str], since: datetime | None = None
    ) -> dict[str, list[EvalSample]]:
        result: dict[str, list[EvalSample]] = {ds: [] for ds in datasets}

        with sqlite3.connect(self._db_path) as conn:
            trace_ids = self._complete_trace_ids(conn, since=since)
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

    def _complete_trace_ids(
        self, conn: sqlite3.Connection, since: datetime | None = None
    ) -> list[int]:
        """Return trace_rowids for traces that reached the generate node.

        Out-of-scope queries terminate before generate and are excluded —
        RAGAS metrics require both a response and retrieved contexts.
        since: when set, restricts to traces whose start_time >= since (UTC).
        """
        sql = "SELECT DISTINCT trace_rowid FROM spans WHERE name = 'generate'"
        params: tuple = ()
        if since is not None:
            sql += " AND datetime(start_time) >= datetime(?)"
            params = (since.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),)
        return [row[0] for row in conn.execute(sql, params).fetchall()]

    def _extract_sample(self, conn: sqlite3.Connection, trace_rowid: int) -> EvalSample | None:
        cur = conn.execute(
            "SELECT id, name, attributes FROM spans WHERE trace_rowid = ? ORDER BY start_time",
            (trace_rowid,),
        )
        # Collect spans grouped by name; span_rowids tracks spans.id for annotation writes.
        # Insertion order (start_time ASC) is preserved so generate[-1] is the final node.
        spans: dict[str, list[dict]] = defaultdict(list)
        span_rowids: dict[str, list[int]] = defaultdict(list)
        for span_id, name, attrs_json in cur.fetchall():
            try:
                spans[name].append(json.loads(attrs_json))
                span_rowids[name].append(span_id)
            except (json.JSONDecodeError, TypeError):
                logger.warning("Skipping malformed span '%s' in trace %s", name, trace_rowid)
                continue

        if not spans.get("LangGraph") or not spans.get("generate"):
            logger.debug("Skipping trace %s — missing root or generate span", trace_rowid)
            return None

        try:
            root_input = json.loads(spans["LangGraph"][0]["input"]["value"])
            user_input: str = root_input["query"]

            # query_type is set by analyze_query mid-run; read from output, not input
            root_output = json.loads(spans["LangGraph"][0]["output"]["value"])
            query_type: str = root_output.get("query_type", "")

            response_output = json.loads(spans["generate"][-1]["output"]["value"])
            response: str = response_output["answer"]["answer"]

            contexts = self._extract_contexts(spans.get("retrieve", []))
        except (KeyError, json.JSONDecodeError, TypeError) as exc:
            logger.warning("Skipping trace %s — failed to parse spans: %s", trace_rowid, exc)
            return None

        return EvalSample(
            trace_id=span_rowids["LangGraph"][0],
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
