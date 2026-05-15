import logging

import tiktoken

from config.loader import ChunkingConfig
from db.models import FilingRecord
from etl.chunker.base import Chunker
from etl.types import ChildChunk, ParentChunk, ParsedSection

logger = logging.getLogger(__name__)


class RecursiveChunker(Chunker):
    def __init__(self, config: ChunkingConfig) -> None:
        self._config = config
        self._enc = tiktoken.get_encoding("cl100k_base")

    def chunk(
        self, sections: list[ParsedSection], filing: FilingRecord
    ) -> tuple[list[ParentChunk], list[ChildChunk]]:
        self._validate_sections(sections)

        parents: list[ParentChunk] = []
        children: list[ChildChunk] = []
        parent_idx = 0
        child_idx = 0

        for section in sections:
            if not section.text.strip():
                continue

            prefix = self._build_prefix(filing, section.section_name)

            if section.content_type == "table":
                parent = ParentChunk(
                    section_name=section.section_name,
                    content_type="table",
                    text=section.text,
                    token_count=self._count_tokens(section.text),
                    filing_chunk_index=parent_idx,
                )
                parents.append(parent)
                parent_idx += 1

                for raw_child in self._split_table_rows(section.text, prefix):
                    text = prefix + "\n\n" + raw_child
                    children.append(ChildChunk(
                        section_name=section.section_name,
                        content_type="table",
                        text=text,
                        token_count=self._count_tokens(text),
                        filing_chunk_index=child_idx,
                        parent_chunk_index=parent.filing_chunk_index,
                    ))
                    child_idx += 1

            else:
                prefix_tokens = self._count_tokens(prefix + "\n\n")
                eff_child = self._config.child_chunk_size - prefix_tokens
                if eff_child <= 0:
                    logger.warning(
                        "Prefix for section '%s' consumes %d tokens, leaving no room in "
                        "child_chunk_size=%d — clamping to 1",
                        section.section_name, prefix_tokens, self._config.child_chunk_size,
                    )
                    eff_child = 1

                for raw_parent in self._split(
                    section.text,
                    self._config.parent_chunk_size,
                    self._config.parent_chunk_overlap,
                    ["\n\n", "\n", " ", ""],
                ):
                    parent = ParentChunk(
                        section_name=section.section_name,
                        content_type="narrative",
                        text=raw_parent,
                        token_count=self._count_tokens(raw_parent),
                        filing_chunk_index=parent_idx,
                    )
                    parents.append(parent)
                    parent_idx += 1

                    for raw_child in self._split(
                        raw_parent,
                        eff_child,
                        self._config.child_chunk_overlap,
                        ["\n\n", "\n", " ", ""],
                    ):
                        text = prefix + "\n\n" + raw_child
                        children.append(ChildChunk(
                            section_name=section.section_name,
                            content_type="narrative",
                            text=text,
                            token_count=self._count_tokens(text),
                            filing_chunk_index=child_idx,
                            parent_chunk_index=parent.filing_chunk_index,
                        ))
                        child_idx += 1

        logger.info(
            "Chunked %d sections → %d parents, %d children",
            len(sections), len(parents), len(children),
        )
        return parents, children

    def _build_prefix(self, filing: FilingRecord, section_name: str) -> str:
        year = filing.fiscal_year_end.year if filing.fiscal_year_end else filing.filing_date.year
        return (
            f"{filing.company_name} ({filing.ticker}) | "
            f"{filing.form_type} FY{year} | "
            f"{section_name}"
        )

    def _count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    def _split(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
        seps: list[str],
    ) -> list[str]:
        """Recursively split text on separators until every chunk fits in chunk_size tokens."""
        if self._count_tokens(text) <= chunk_size:
            return [text.strip()] if text.strip() else []

        sep, *rest = seps

        # Last resort: split at raw token boundaries
        if sep == "":
            toks = self._enc.encode(text)
            out, i = [], 0
            while i < len(toks):
                j = min(i + chunk_size, len(toks))
                out.append(self._enc.decode(toks[i:j]))
                if j >= len(toks):
                    break
                i = j - overlap
            return out

        parts = [p for p in text.split(sep) if p.strip()]
        out: list[str] = []
        window: list[str] = []

        for part in parts:
            # Part alone is too large — recurse with a finer separator
            if self._count_tokens(part) > chunk_size:
                if window:
                    out.append(sep.join(window))
                    window = []
                out.extend(self._split(part, chunk_size, overlap, rest or [""]))
                continue

            candidate = sep.join(window + [part]) if window else part
            if self._count_tokens(candidate) > chunk_size and window:
                out.append(sep.join(window))
                # Retain a tail of the window within the overlap budget
                tail: list[str] = []
                for p in reversed(window):
                    probe = sep.join([p] + tail) if tail else p
                    if self._count_tokens(probe) <= overlap:
                        tail.insert(0, p)
                    else:
                        break
                window = tail

            window.append(part)

        if window:
            out.append(sep.join(window))

        return [c for c in out if c.strip()]

    def _split_table_rows(self, text: str, prefix: str) -> list[str]:
        """Split a table (markdown or reformatted plain-text) into row-groups.

        Markdown tables: header + separator prepended to every group.
        Reformatted plain-text (no --- separator): context line (first line)
        prepended to every group; data rows are already self-descriptive.
        """
        lines = text.strip().splitlines()

        has_separator = len(lines) >= 3 and any("---" in l for l in lines[:3])
        if has_separator:
            header, separator, *data_rows = lines
            sticky = header + "\n" + separator
            def _make_group(rows: list[str]) -> str:
                return sticky + "\n" + "\n".join(rows)
        else:
            if len(lines) < 2:
                return [text]
            sticky = lines[0]
            data_rows = lines[1:]
            def _make_group(rows: list[str]) -> str:
                return sticky + "\n" + "\n".join(rows)

        prefix_tokens = self._count_tokens(prefix + "\n\n")
        sticky_tokens = self._count_tokens(sticky + "\n")
        available = self._config.child_chunk_size - prefix_tokens - sticky_tokens

        if available <= 0:
            return [text]

        groups: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for row in data_rows:
            row_tokens = self._count_tokens(row + "\n")
            if current_tokens + row_tokens > available and current:
                groups.append(_make_group(current))
                current = []
                current_tokens = 0
            current.append(row)
            current_tokens += row_tokens

        if current:
            groups.append(_make_group(current))

        return groups or [text]
