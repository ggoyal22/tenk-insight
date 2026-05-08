#!/usr/bin/env python3
"""Bidirectional prompt sync between local constants and Phoenix.

Push: reads constants from generation/prompts.py → creates a new version in
Phoenix for each prompt (immutable — never overwrites an existing version).

Pull: fetches prompt versions from Phoenix → rewrites the constant values in
generation/prompts.py in place. Use --tag or --version to target a specific
version; omit both for the latest.

Usage:
    python scripts/sync_prompts.py push
    python scripts/sync_prompts.py pull
    python scripts/sync_prompts.py pull --tag production
"""

import argparse
import os
import re
import sys
from pathlib import Path

# Ensure project root on path regardless of working directory.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

from phoenix.client import Client
from phoenix.client.__generated__ import v1
from phoenix.client.types.prompts import PromptVersion

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

# Maps Phoenix prompt name → (Python constant name, current value)
_PROMPTS: dict[str, tuple[str, str]] = {
    "classify":          ("CLASSIFY_PROMPT",          CLASSIFY_PROMPT),
    "plan_single":       ("PLAN_SINGLE_PROMPT",       PLAN_SINGLE_PROMPT),
    "plan_comparison":   ("PLAN_COMPARISON_PROMPT",   PLAN_COMPARISON_PROMPT),
    "plan_time_series":  ("PLAN_TIME_SERIES_PROMPT",  PLAN_TIME_SERIES_PROMPT),
    "plan_multi_hop":    ("PLAN_MULTI_HOP_PROMPT",    PLAN_MULTI_HOP_PROMPT),
    "hyde":              ("HYDE_PROMPT",               HYDE_PROMPT),
    "qa":                ("QA_PROMPT",                 QA_PROMPT),
    "comparison":        ("COMPARISON_PROMPT",         COMPARISON_PROMPT),
    "time_series":       ("TIME_SERIES_PROMPT",        TIME_SERIES_PROMPT),
    "check_hop":         ("CHECK_HOP_PROMPT",          CHECK_HOP_PROMPT),
    "reflection":        ("REFLECTION_PROMPT",         REFLECTION_PROMPT),
}

_PROMPTS_FILE = _ROOT / "generation" / "prompts.py"


def _client() -> Client:
    base_url = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not base_url:
        sys.exit("OTEL_EXPORTER_OTLP_ENDPOINT is not set — cannot connect to Phoenix.")
    return Client(base_url=base_url)


def _make_version(model_name: str, model_provider: str, text: str) -> PromptVersion:
    provider = model_provider.upper()
    # Phoenix client accepts a subset of provider names; map ollama → OLLAMA etc.
    return PromptVersion(
        [v1.PromptMessage(role="system", content=text)],
        model_name=model_name,
        model_provider=provider,  # type: ignore[arg-type]
        template_format="NONE",
    )


def _extract_system_message(prompt_version: PromptVersion) -> str:
    for msg in prompt_version._template["messages"]:
        if msg["role"] == "system":
            content = msg["content"]
            if isinstance(content, str):
                return content
            return "".join(part["text"] for part in content if part.get("type") == "text")
    raise ValueError("No system message found in prompt version")


# ---------------------------------------------------------------------------
# Push
# ---------------------------------------------------------------------------

def cmd_push(args: argparse.Namespace) -> None:
    from config.loader import load_config
    config = load_config()
    client = _client()

    for name, (const_name, text) in _PROMPTS.items():
        version = client.prompts.create(
            name=name,
            version=_make_version(config.llm.model, config.llm.provider, text),
        )
        print(f"  pushed '{name}' → version id {version.id}")

    print(f"Done. {len(_PROMPTS)} prompt(s) pushed to Phoenix.")


# ---------------------------------------------------------------------------
# Pull
# ---------------------------------------------------------------------------

def _replace_constant(source: str, const_name: str, new_text: str) -> str:
    """Replace a triple-quoted string constant in Python source."""
    pattern = rf'({re.escape(const_name)}\s*=\s*)""".*?"""'
    # Escape any triple quotes in the new text (unlikely but safe)
    safe_text = new_text.replace('"""', r'\"\"\"')
    replacement = rf'\g<1>"""{safe_text}"""'
    updated, count = re.subn(pattern, replacement, source, count=1, flags=re.DOTALL)
    if count == 0:
        raise ValueError(f"Could not locate '{const_name}' triple-quoted assignment in {_PROMPTS_FILE}")
    return updated


def cmd_pull(args: argparse.Namespace) -> None:
    client = _client()
    tag: str | None = args.tag
    tag_label = f" (tag={tag})" if tag else " (latest)"

    source = _PROMPTS_FILE.read_text()
    updated = source
    pulled = 0

    for name, (const_name, _) in _PROMPTS.items():
        kwargs: dict = {"prompt_identifier": name}
        if tag:
            kwargs["tag"] = tag
        try:
            pv = client.prompts.get(**kwargs)
            text = _extract_system_message(pv)
            updated = _replace_constant(updated, const_name, text)
            print(f"  pulled '{name}'{tag_label}")
            pulled += 1
        except Exception as exc:
            print(f"  skipped '{name}': {exc}")

    if pulled:
        _PROMPTS_FILE.write_text(updated)
        print(f"Done. {pulled} prompt(s) written to {_PROMPTS_FILE}.")
    else:
        print("No prompts were pulled — file unchanged.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync prompts between local constants and Phoenix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("push", help="Push local constants to Phoenix as new versions")

    pull_p = sub.add_parser("pull", help="Pull from Phoenix into local constants")
    pull_p.add_argument("--tag", metavar="TAG", help="Pull versions with this tag (e.g. 'production')")

    args = parser.parse_args()

    if args.command == "push":
        cmd_push(args)
    elif args.command == "pull":
        cmd_pull(args)


if __name__ == "__main__":
    main()
