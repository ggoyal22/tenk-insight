from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from edgar.files.html_documents import TableBlock
from etl.parser.edgartools import EdgarToolsParser, _reformat_table


# ---------------------------------------------------------------------------
# Helpers for mocking edgartools document blocks
# ---------------------------------------------------------------------------

class FakeTableBlock(TableBlock):
    """Minimal TableBlock subclass that bypasses the BeautifulSoup Tag init."""
    def __init__(self, md: str) -> None:
        self._md = md

    def to_markdown(self) -> str:
        return self._md


class FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.text = text

    def is_empty(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# _reformat_table — pure function tests
# ---------------------------------------------------------------------------

def test_reformat_table_dollar_number_collapse():
    md = (
        "| | | FY2025 | | FY2024 | |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        "| | Revenue | $ | 12,914 | $ | 10,000 |"
    )
    result = _reformat_table(md)
    assert "Revenue: FY2025=$12,914, FY2024=$10,000" in result
    assert "FY2025 | FY2024" in result


def test_reformat_table_number_percent_collapse():
    md = (
        "| | | FY2025 | | FY2024 | |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        "| | Gross Margin | 74 | % | 73 | % |"
    )
    result = _reformat_table(md)
    assert "Gross Margin: FY2025=74%, FY2024=73%" in result


def test_reformat_table_multi_row_header_stacks_labels():
    md = (
        "| | | Year Ended | | Year Ended |\n"
        "| | | Jan 2025 | | Jan 2024 |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| | Revenue | $ | 12,914 | $ | 10,000 |"
    )
    result = _reformat_table(md)
    assert "Year Ended Jan 2025" in result
    assert "Year Ended Jan 2024" in result


def test_reformat_table_empty_input_returns_original():
    assert _reformat_table("") == ""


def test_reformat_table_no_pipe_lines_returns_original():
    md = "Just plain text without pipes"
    assert _reformat_table(md) == md


def test_reformat_table_no_data_rows_returns_original():
    md = (
        "| | | FY2025 |\n"
        "| --- | --- | --- |"
    )
    assert _reformat_table(md) == md


# ---------------------------------------------------------------------------
# EdgarToolsParser._resolve_items — pure method tests
# ---------------------------------------------------------------------------

def test_resolve_items_no_filter_returns_all():
    parser = EdgarToolsParser(items_filter=None)
    available = ["Item 1", "Item 1A", "Item 7"]
    assert parser._resolve_items(available) == available


def test_resolve_items_filter_returns_subset_in_filter_order():
    parser = EdgarToolsParser(items_filter=["Item 7", "Item 1"])
    available = ["Item 1", "Item 1A", "Item 7", "Item 8"]
    assert parser._resolve_items(available) == ["Item 7", "Item 1"]


def test_resolve_items_unknown_items_silently_excluded():
    parser = EdgarToolsParser(items_filter=["Item 1", "Item 99"])
    available = ["Item 1", "Item 7"]
    assert parser._resolve_items(available) == ["Item 1"]


# ---------------------------------------------------------------------------
# EdgarToolsParser.parse — tests using mocked ChunkedDocument
# ---------------------------------------------------------------------------

def test_parse_raises_on_missing_file(tmp_path):
    parser = EdgarToolsParser()
    with pytest.raises(FileNotFoundError):
        parser.parse(tmp_path / "nonexistent.html")


def test_parse_raises_on_empty_file(tmp_path):
    empty = tmp_path / "empty.html"
    empty.write_bytes(b"")
    parser = EdgarToolsParser()
    with pytest.raises(ValueError):
        parser.parse(empty)


def test_parse_narrative_section(tmp_path):
    filing = tmp_path / "filing.html"
    filing.write_text("<html>placeholder</html>", encoding="utf-8")

    text_block = FakeTextBlock("NVIDIA is a technology company.")
    mock_cd = MagicMock()
    mock_cd.list_items.return_value = ["Item 1"]
    mock_cd.chunks_for_item.return_value = iter([[text_block]])

    with patch("etl.parser.edgartools.ChunkedDocument", return_value=mock_cd):
        parser = EdgarToolsParser()
        sections = parser.parse(filing)

    assert len(sections) == 1
    assert sections[0].section_name == "Item 1"
    assert sections[0].content_type == "narrative"
    assert "NVIDIA" in sections[0].text
    assert sections[0].section_order == 0


def test_parse_table_section(tmp_path):
    filing = tmp_path / "filing.html"
    filing.write_text("<html>placeholder</html>", encoding="utf-8")

    md = "| | | FY2025 |\n| --- | --- | --- |\n| | Revenue | $ | 1,000 |"
    table_block = FakeTableBlock(md)
    mock_cd = MagicMock()
    mock_cd.list_items.return_value = ["Item 8"]
    mock_cd.chunks_for_item.return_value = iter([[table_block]])

    with patch("etl.parser.edgartools.ChunkedDocument", return_value=mock_cd):
        parser = EdgarToolsParser()
        sections = parser.parse(filing)

    assert len(sections) == 1
    assert sections[0].content_type == "table"


def test_parse_empty_item_produces_empty_section(tmp_path):
    filing = tmp_path / "filing.html"
    filing.write_text("<html>placeholder</html>", encoding="utf-8")

    mock_cd = MagicMock()
    mock_cd.list_items.return_value = ["Item 1", "Item 2"]
    # Item 1 has no blocks; Item 2 has content
    text_block = FakeTextBlock("Some content.")
    mock_cd.chunks_for_item.side_effect = [iter([[]]), iter([[text_block]])]

    with patch("etl.parser.edgartools.ChunkedDocument", return_value=mock_cd):
        parser = EdgarToolsParser()
        sections = parser.parse(filing)

    assert len(sections) == 2
    empty_section = next(s for s in sections if s.section_name == "Item 1")
    assert empty_section.text == ""


def test_parse_section_order_is_sequential(tmp_path):
    filing = tmp_path / "filing.html"
    filing.write_text("<html>placeholder</html>", encoding="utf-8")

    mock_cd = MagicMock()
    mock_cd.list_items.return_value = ["Item 1", "Item 1A", "Item 7"]
    mock_cd.chunks_for_item.side_effect = [
        iter([[FakeTextBlock("Business.")]]),
        iter([[FakeTextBlock("Risk factors.")]]),
        iter([[FakeTextBlock("MD&A.")]]),
    ]

    with patch("etl.parser.edgartools.ChunkedDocument", return_value=mock_cd):
        parser = EdgarToolsParser()
        sections = parser.parse(filing)

    orders = [s.section_order for s in sections]
    assert orders == list(range(len(sections)))


def test_parse_respects_items_filter(tmp_path):
    filing = tmp_path / "filing.html"
    filing.write_text("<html>placeholder</html>", encoding="utf-8")

    mock_cd = MagicMock()
    mock_cd.list_items.return_value = ["Item 1", "Item 1A", "Item 7"]
    mock_cd.chunks_for_item.return_value = iter([[FakeTextBlock("Content.")]])

    with patch("etl.parser.edgartools.ChunkedDocument", return_value=mock_cd):
        parser = EdgarToolsParser(items_filter=["Item 7"])
        sections = parser.parse(filing)

    assert len(sections) == 1
    assert sections[0].section_name == "Item 7"
