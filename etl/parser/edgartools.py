import logging
from pathlib import Path
from typing import Literal

from edgar.files.html_documents import TableBlock
from edgar.files.htmltools import ChunkedDocument

from etl.parser.base import Parser
from etl.types import ParsedSection

logger = logging.getLogger(__name__)


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
                    text = "\n\n".join(b.to_markdown() for b in table_blocks)
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
