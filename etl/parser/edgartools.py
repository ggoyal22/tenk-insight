import logging
import re
from pathlib import Path
from typing import Literal

from edgar.files.html_documents import TableBlock
from edgar.files.htmltools import ChunkedDocument

from etl.parser.base import Parser
from etl.types import ParsedSection

logger = logging.getLogger(__name__)


def _reformat_table(md: str) -> str:
    """Reformat a fragmented SEC markdown table into readable key-value text.

    SEC XBRL tables split dollar signs, numbers, and percent signs across
    separate cells. This collapses them into value groups and pairs each group
    with its column header label by position.

    Output format:
        <context line joining all header row values>
        <row label>: <header0>=<value0>, <header1>=<value1>, ...
    """
    lines = [l for l in md.strip().splitlines() if l.strip()]
    if not lines:
        return md

    def parse_row(line: str) -> list[str]:
        parts = line.split("|")
        return [p.strip() for p in parts[1:-1]]

    rows = []
    for line in lines:
        if not line.startswith("|"):
            continue
        if "---" in line:
            continue
        rows.append(parse_row(line))

    if not rows:
        return md

    # Header rows have an empty label column (col 1); data rows have a non-empty one.
    header_rows = [r for r in rows if len(r) > 1 and r[1] == ""]
    data_rows   = [r for r in rows if len(r) > 1 and r[1] != ""]

    if not data_rows:
        return md

    # Build group-position labels by joining non-empty values from every header
    # row in order. Values at the same group position are joined with a space so
    # that "Year Ended", "Jan 26, 2025", and "($ in millions)" all end up together.
    group_buckets: list[list[str]] = []
    for hr in header_rows:
        pos = 0
        for cell in hr[2:]:
            if cell:
                while len(group_buckets) <= pos:
                    group_buckets.append([])
                group_buckets[pos].append(cell)
                pos += 1

    group_labels = [" ".join(vals) for vals in group_buckets]
    context_line = " | ".join(group_labels) if group_labels else ""

    def _is_number(s: str) -> bool:
        if not s:
            return False
        cleaned = re.sub(r"[,.\-—()]", "", s)
        return bool(cleaned) and cleaned.isdigit()

    output_rows: list[str] = []
    for row in data_rows:
        if len(row) < 2 or not row[1]:
            continue
        label = row[1]

        # Collapse cells into value groups:
        #   "$ | 12,914"  ->  "$12,914"
        #   "9.9  | %"    ->  "9.9%"
        #   empty cells   ->  skipped
        groups: list[str] = []
        cells = row[2:]
        i = 0
        while i < len(cells):
            cell = cells[i]
            if not cell:
                i += 1
                continue
            nxt = cells[i + 1] if i + 1 < len(cells) else ""
            if cell == "$" and _is_number(nxt):
                groups.append(f"${nxt}")
                i += 2
            elif _is_number(cell) and nxt == "%":
                groups.append(f"{cell}%")
                i += 2
            else:
                groups.append(cell)
                i += 1

        if not groups:
            continue

        parts: list[str] = []
        for gi, val in enumerate(groups):
            lbl = group_labels[gi] if gi < len(group_labels) else ""
            parts.append(f"{lbl}={val}" if lbl else val)

        output_rows.append(f"{label}: {', '.join(parts)}")

    if not output_rows:
        return md

    result = []
    if context_line:
        result.append(context_line)
    result.extend(output_rows)
    return "\n".join(result)


class EdgarToolsParser(Parser):
    def __init__(self, items_filter: list[str] | None = None) -> None:
        """
        Args:
            items_filter: if provided, only parse these items (e.g. ["Item 7", "Item 1A"]).
                          None means parse all items — use for production.
        """
        self._items_filter = items_filter

    def parse(self, raw_path: Path) -> list[ParsedSection]:
        self._validate_raw_file(raw_path)

        html = raw_path.read_text(encoding="utf-8")
        cd = ChunkedDocument(html)

        available = cd.list_items()
        items_to_parse = self._resolve_items(available)

        sections: list[ParsedSection] = []
        order = 0

        for item_name in items_to_parse:
            groups = list(cd.chunks_for_item(item_name))
            item_had_content = False

            for group in groups:
                table_blocks = [b for b in group if isinstance(b, TableBlock)]
                text_blocks = [b for b in group
                               if not isinstance(b, TableBlock)
                               and b.text is not None
                               and not b.is_empty()]

                if not table_blocks and not text_blocks:
                    continue

                if table_blocks:
                    text = "\n\n".join(_reformat_table(b.to_markdown()) for b in table_blocks)
                    ctype: Literal["narrative", "table"] = "table"
                else:
                    text = "\n".join(b.text for b in text_blocks)
                    ctype = "narrative"

                sections.append(ParsedSection(
                    section_name=item_name,
                    content_type=ctype,
                    text=text,
                    section_order=order,
                ))
                order += 1
                item_had_content = True

            if not item_had_content:
                # Section header present in TOC but no extractable content blocks
                sections.append(ParsedSection(
                    section_name=item_name,
                    content_type="narrative",
                    text="",
                    section_order=order,
                ))
                order += 1

        logger.info(
            "Parsed %s → %d sections (%d items)",
            raw_path.name, len(sections), len(items_to_parse),
        )
        return sections

    def _resolve_items(self, available: list[str]) -> list[str]:
        if self._items_filter is None:
            return available
        unknown = set(self._items_filter) - set(available)
        if unknown:
            logger.warning("items_filter contains items not found in document: %s", unknown)
        return [i for i in self._items_filter if i in set(available)]
