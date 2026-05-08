"""Runtime prompt registry with per-prompt Phoenix fallback.

Fetches each prompt from Phoenix at startup using OTEL_EXPORTER_OTLP_ENDPOINT
as the base URL (same env var used by tracing). Falls back to the hardcoded
constant for any prompt that cannot be retrieved (Phoenix down, prompt not yet
created, etc.).

Logs the source (Phoenix or constant) for each prompt at INFO level.
"""

import dataclasses
import logging
import os
from dataclasses import dataclass

from generation.prompts import (
    CHECK_HOP_PROMPT,
    CLASSIFY_PROMPT,
    COMPARISON_PROMPT,
    HYDE_PROMPT,
    PLAN_COMPARISON_PROMPT,
    PLAN_MULTI_HOP_PROMPT,
    PLAN_SINGLE_PROMPT,
    PLAN_TIME_SERIES_PROMPT,
    QA_PROMPT,
    REFLECTION_PROMPT,
    TIME_SERIES_PROMPT,
)

logger = logging.getLogger(__name__)

_DEFAULTS: dict[str, str] = {
    "classify": CLASSIFY_PROMPT,
    "plan_single": PLAN_SINGLE_PROMPT,
    "plan_comparison": PLAN_COMPARISON_PROMPT,
    "plan_time_series": PLAN_TIME_SERIES_PROMPT,
    "plan_multi_hop": PLAN_MULTI_HOP_PROMPT,
    "hyde": HYDE_PROMPT,
    "qa": QA_PROMPT,
    "comparison": COMPARISON_PROMPT,
    "time_series": TIME_SERIES_PROMPT,
    "check_hop": CHECK_HOP_PROMPT,
    "reflection": REFLECTION_PROMPT,
}


@dataclass(frozen=True)
class Prompts:
    classify: str
    plan_single: str
    plan_comparison: str
    plan_time_series: str
    plan_multi_hop: str
    hyde: str
    qa: str
    comparison: str
    time_series: str
    check_hop: str
    reflection: str
    versions: dict[str, str | None] = dataclasses.field(default_factory=dict)


def load_prompts(tag: str | None = None) -> Prompts:
    """Fetch all prompts from Phoenix, falling back to hardcoded constants per prompt."""
    base_url = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not base_url:
        logger.debug("OTEL_EXPORTER_OTLP_ENDPOINT not set — using hardcoded prompt constants")
        return Prompts(**_DEFAULTS)

    try:
        from phoenix.client import Client
        client = Client(base_url=base_url)
    except Exception as exc:
        logger.info("Could not initialise Phoenix client (%s) — using hardcoded prompt constants", exc)
        return Prompts(**_DEFAULTS)

    resolved: dict[str, str] = {}
    versions: dict[str, str | None] = {}
    tag_label = f" (tag={tag})" if tag else " (latest)"

    for name, fallback in _DEFAULTS.items():
        try:
            kwargs: dict = {"prompt_identifier": name}
            if tag:
                kwargs["tag"] = tag
            pv = client.prompts.get(**kwargs)
            resolved[name] = _extract_system_message(pv)
            versions[name] = pv.id
            logger.info("Loaded prompt '%s' from Phoenix%s", name, tag_label)
        except Exception as exc:
            resolved[name] = fallback
            versions[name] = None
            logger.info(
                "Prompt '%s' not in Phoenix (%s) — using hardcoded constant", name, exc
            )

    return Prompts(**resolved, versions=versions)


def _extract_system_message(prompt_version) -> str:
    for msg in prompt_version._template["messages"]:
        if msg["role"] == "system":
            content = msg["content"]
            if isinstance(content, str):
                return content
            return "".join(part["text"] for part in content if part.get("type") == "text")
    raise ValueError("No system message found in prompt version")
