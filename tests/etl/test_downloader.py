from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from etl.downloader.base import FilingNotFoundError
from etl.downloader.edgartools import EdgarToolsDownloader, _parse_period


# ---------------------------------------------------------------------------
# _parse_period — pure function tests
# ---------------------------------------------------------------------------

def test_parse_period_yyyymmdd_format():
    result = _parse_period("20240128", "NVDA", 2024)
    assert result == date(2024, 1, 28)


def test_parse_period_yyyy_mm_dd_format():
    result = _parse_period("2024-01-28", "NVDA", 2024)
    assert result == date(2024, 1, 28)


def test_parse_period_none_raises():
    with pytest.raises(FilingNotFoundError):
        _parse_period(None, "NVDA", 2024)


def test_parse_period_invalid_format_raises():
    with pytest.raises(FilingNotFoundError):
        _parse_period("28/01/2024", "NVDA", 2024)


# ---------------------------------------------------------------------------
# EdgarToolsDownloader.fetch — tests using mocked edgar library
# ---------------------------------------------------------------------------

def _make_config(raw_dir: Path) -> MagicMock:
    config = MagicMock()
    config.edgar.user_agent = "Test Suite test@example.com"
    config.edgar.raw_data_dir = raw_dir
    return config


def _make_filing_obj(year: int = 2024) -> MagicMock:
    f = MagicMock()
    f.period_of_report = f"{year}0128"
    f.form = "10-K"
    f.filing_date = date(year, 2, 21)
    f.accession_no = f"0001045810-{year}-000001"
    f.cik = "1045810"
    f.filing_url = "https://www.sec.gov/test"
    f.html.return_value = "<html>10-K content</html>"
    return f


@patch("etl.downloader.edgartools.set_identity")
@patch("etl.downloader.edgartools.Company")
def test_fetch_raises_when_ticker_not_found(mock_company_cls, mock_set_identity, tmp_path):
    mock_company = MagicMock()
    mock_company.not_found = True
    mock_company_cls.return_value = mock_company

    downloader = EdgarToolsDownloader(_make_config(tmp_path))
    with pytest.raises(FilingNotFoundError, match="not found"):
        downloader.fetch("FAKE", "10-K", 2024)


@patch("etl.downloader.edgartools.set_identity")
@patch("etl.downloader.edgartools.Company")
def test_fetch_raises_when_no_matching_filings(mock_company_cls, mock_set_identity, tmp_path):
    mock_company = MagicMock()
    mock_company.not_found = False
    mock_company.get_filings.return_value = []
    mock_company_cls.return_value = mock_company

    downloader = EdgarToolsDownloader(_make_config(tmp_path))
    with pytest.raises(FilingNotFoundError, match="No 10-K filing"):
        downloader.fetch("NVDA", "10-K", 2024)


@patch("etl.downloader.edgartools.set_identity")
@patch("etl.downloader.edgartools.Company")
def test_fetch_raises_when_html_is_empty(mock_company_cls, mock_set_identity, tmp_path):
    filing_obj = _make_filing_obj(2024)
    filing_obj.html.return_value = None

    mock_company = MagicMock()
    mock_company.not_found = False
    mock_company.get_filings.return_value = [filing_obj]
    mock_company_cls.return_value = mock_company

    downloader = EdgarToolsDownloader(_make_config(tmp_path))
    with pytest.raises(FilingNotFoundError, match="empty"):
        downloader.fetch("NVDA", "10-K", 2024)


@patch("etl.downloader.edgartools.set_identity")
@patch("etl.downloader.edgartools.Company")
def test_fetch_returns_filing_record_and_path(mock_company_cls, mock_set_identity, tmp_path):
    filing_obj = _make_filing_obj(2024)

    mock_company = MagicMock()
    mock_company.not_found = False
    mock_company.name = "NVIDIA Corporation"
    mock_company.sic = "3674"
    mock_company.get_filings.return_value = [filing_obj]
    mock_company_cls.return_value = mock_company

    downloader = EdgarToolsDownloader(_make_config(tmp_path))
    record, raw_path = downloader.fetch("NVDA", "10-K", 2024)

    assert record.ticker == "NVDA"
    assert record.company_name == "NVIDIA Corporation"
    assert record.form_type == "10-K"
    assert record.fiscal_year_end == date(2024, 1, 28)
    assert raw_path.exists()
    assert raw_path.read_text(encoding="utf-8") == "<html>10-K content</html>"


@patch("etl.downloader.edgartools.set_identity")
@patch("etl.downloader.edgartools.Company")
def test_fetch_selects_most_recent_filing_when_multiple_match(mock_company_cls, mock_set_identity, tmp_path):
    # Two filings for the same year — fetch should pick the later filing_date.
    early = _make_filing_obj(2024)
    early.filing_date = date(2024, 2, 1)
    early.accession_no = "0001045810-24-000001"
    early.html.return_value = "<html>early</html>"

    late = _make_filing_obj(2024)
    late.filing_date = date(2024, 2, 21)
    late.accession_no = "0001045810-24-000002"
    late.html.return_value = "<html>late</html>"

    mock_company = MagicMock()
    mock_company.not_found = False
    mock_company.name = "NVIDIA Corporation"
    mock_company.sic = "3674"
    mock_company.get_filings.return_value = [early, late]
    mock_company_cls.return_value = mock_company

    downloader = EdgarToolsDownloader(_make_config(tmp_path))
    record, raw_path = downloader.fetch("NVDA", "10-K", 2024)

    assert record.accession_number == "0001045810-24-000002"
    assert raw_path.read_text() == "<html>late</html>"


@patch("etl.downloader.edgartools.set_identity")
@patch("etl.downloader.edgartools.Company")
def test_fetch_saves_raw_file_under_ticker_subdirectory(mock_company_cls, mock_set_identity, tmp_path):
    filing_obj = _make_filing_obj(2024)

    mock_company = MagicMock()
    mock_company.not_found = False
    mock_company.name = "NVIDIA Corporation"
    mock_company.sic = None
    mock_company.get_filings.return_value = [filing_obj]
    mock_company_cls.return_value = mock_company

    downloader = EdgarToolsDownloader(_make_config(tmp_path))
    _, raw_path = downloader.fetch("nvda", "10-K", 2024)  # lowercase ticker

    assert raw_path.parent == tmp_path / "NVDA"
    assert raw_path.suffix == ".html"
