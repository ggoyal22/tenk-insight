from datetime import date, datetime, timezone
from uuid import uuid4

import pytest

from db.models import FilingRecord
from etl.types import ParsedSection


def test_empty_sections_raises(chunker, filing):
    with pytest.raises(ValueError, match="empty"):
        chunker.chunk([], filing)


def test_single_short_narrative_produces_one_parent_one_child(chunker, filing):
    sections = [ParsedSection(section_name="Item 1", content_type="narrative", text="NVIDIA makes GPUs.", section_order=0)]
    parents, children = chunker.chunk(sections, filing)
    assert len(parents) == 1
    assert len(children) == 1


def test_child_text_contains_prefix(chunker, filing):
    sections = [ParsedSection(section_name="Item 1", content_type="narrative", text="NVIDIA makes GPUs.", section_order=0)]
    _, children = chunker.chunk(sections, filing)
    assert "NVIDIA Corporation (NVDA)" in children[0].text
    assert "10-K" in children[0].text
    assert "Item 1" in children[0].text


def test_prefix_uses_fiscal_year_end_year(chunker, filing):
    # filing.fiscal_year_end = 2024-01-28 → FY2024
    sections = [ParsedSection(section_name="Item 1A", content_type="narrative", text="Risk text.", section_order=0)]
    _, children = chunker.chunk(sections, filing)
    assert "FY2024" in children[0].text


def test_prefix_falls_back_to_filing_date_year(chunker):
    no_fye = FilingRecord(
        id=uuid4(),
        ticker="AAPL",
        company_name="Apple Inc.",
        cik="320193",
        accession_number="0000320193-23-000001",
        form_type="10-K",
        filing_date=date(2023, 11, 3),
        fiscal_year_end=None,
        source_url="https://www.sec.gov/test",
        downloaded_at=datetime.now(timezone.utc),
    )
    sections = [ParsedSection(section_name="Item 1", content_type="narrative", text="Apple business.", section_order=0)]
    _, children = chunker.chunk(sections, no_fye)
    assert "FY2023" in children[0].text


def test_whitespace_only_section_is_skipped(chunker, filing):
    sections = [
        ParsedSection(section_name="Item 1", content_type="narrative", text="   \n  ", section_order=0),
        ParsedSection(section_name="Item 2", content_type="narrative", text="Real content here.", section_order=1),
    ]
    parents, children = chunker.chunk(sections, filing)
    assert len(parents) == 1
    assert parents[0].section_name == "Item 2"


def test_chunk_indices_are_globally_continuous(chunker, filing):
    text = "word " * 10
    sections = [
        ParsedSection(section_name="Item 1", content_type="narrative", text=text, section_order=0),
        ParsedSection(section_name="Item 2", content_type="narrative", text=text, section_order=1),
    ]
    parents, children = chunker.chunk(sections, filing)
    assert [p.filing_chunk_index for p in parents] == list(range(len(parents)))
    assert [c.filing_chunk_index for c in children] == list(range(len(children)))


def test_child_parent_chunk_index_links_to_valid_parent(chunker, filing):
    text = "word " * 10
    sections = [
        ParsedSection(section_name="Item 1", content_type="narrative", text=text, section_order=0),
        ParsedSection(section_name="Item 2", content_type="narrative", text=text, section_order=1),
    ]
    parents, children = chunker.chunk(sections, filing)
    valid_parent_indices = {p.filing_chunk_index for p in parents}
    for child in children:
        assert child.parent_chunk_index in valid_parent_indices


def test_narrative_splits_into_multiple_children_when_text_is_large(chunker, filing):
    # Default child_chunk_size=256; prefix ≈ 16 tokens → eff_child ≈ 240.
    # A single paragraph of ~500 words exceeds eff_child and forces at least 3 children.
    text = " ".join(["word"] * 500)
    sections = [ParsedSection(section_name="Item 1", content_type="narrative", text=text, section_order=0)]
    _, children = chunker.chunk(sections, filing)
    assert len(children) > 1


def test_table_section_produces_correct_content_type(chunker, filing):
    table_text = (
        "Year Ended Jan 2025 | Year Ended Jan 2024\n"
        "Revenue: FY2025=12,914, FY2024=10,000\n"
        "Net Income: FY2025=5,000, FY2024=4,000"
    )
    sections = [ParsedSection(section_name="Item 8", content_type="table", text=table_text, section_order=0)]
    parents, children = chunker.chunk(sections, filing)
    assert len(parents) == 1
    assert parents[0].content_type == "table"
    assert all(c.content_type == "table" for c in children)


def test_markdown_table_children_each_contain_header_and_separator(chunker, filing):
    table_text = (
        "| Metric | FY2025 | FY2024 |\n"
        "| --- | --- | --- |\n"
        + "\n".join(f"| Row {i} | {i*10} | {i*9} |" for i in range(1, 10))
    )
    sections = [ParsedSection(section_name="Item 8", content_type="table", text=table_text, section_order=0)]
    _, children = chunker.chunk(sections, filing)
    assert len(children) >= 1
    for child in children:
        raw = child.text.split("\n\n", 1)[-1]
        assert "| Metric | FY2025 | FY2024 |" in raw
        assert "| --- | --- | --- |" in raw


def test_plain_text_table_children_each_contain_context_line(chunker, filing):
    context_line = "Year Ended Jan 2025 | Year Ended Jan 2024"
    table_text = context_line + "\n" + "\n".join(f"Row {i}: val={i}" for i in range(1, 10))
    sections = [ParsedSection(section_name="Item 8", content_type="table", text=table_text, section_order=0)]
    _, children = chunker.chunk(sections, filing)
    assert len(children) >= 1
    for child in children:
        raw = child.text.split("\n\n", 1)[-1]
        assert context_line in raw
