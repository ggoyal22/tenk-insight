import logging
import re
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

from edgar import Company, set_identity

from config.loader import AppConfig
from db.models import FilingRecord
from etl.downloader.base import Downloader, FilingNotFoundError


def _parse_period(raw: str | None, ticker: str, year: int) -> date:
    """Parse EDGAR period_of_report to date, handling both YYYYMMDD and YYYY-MM-DD formats."""
    if raw is None:
        raise FilingNotFoundError(
            f"period_of_report is None for {ticker} fiscal year {year} — filing metadata incomplete."
        )
    fmt = "%Y%m%d" if re.fullmatch(r"\d{8}", raw) else "%Y-%m-%d"
    try:
        return datetime.strptime(raw, fmt).date()
    except ValueError:
        raise FilingNotFoundError(
            f"Cannot parse period_of_report {raw!r} for {ticker} fiscal year {year}."
        )

logger = logging.getLogger(__name__)


class EdgarToolsDownloader(Downloader):
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        set_identity(config.edgar.user_agent)

    def fetch(self, ticker: str, form_type: str, year: int) -> tuple[FilingRecord, Path]:
        logger.info("Fetching %s %s fiscal year %d", ticker, form_type, year)

        company = Company(ticker)
        if company.not_found:
            raise FilingNotFoundError(f"Ticker '{ticker}' not found on EDGAR.")

        all_filings = company.get_filings(form=form_type)

        matches = [
            f for f in all_filings
            if f.period_of_report and int(f.period_of_report[:4]) == year
        ]

        if not matches:
            raise FilingNotFoundError(
                f"No {form_type} filing found for {ticker} with fiscal year {year}."
            )

        # take the most recent filing date in case of amendments
        filing = max(matches, key=lambda f: f.filing_date)

        raw_html = filing.html()
        if not raw_html:
            raise FilingNotFoundError(
                f"Primary document HTML is empty for {ticker} {form_type} {year} "
                f"(accession: {filing.accession_no})."
            )

        raw_path = self._save_raw(ticker, filing.accession_no, raw_html)

        record = FilingRecord(
            id=uuid4(),  # placeholder — DB generates the real UUID on insert
            ticker=ticker.upper(),
            company_name=str(company.name),
            cik=str(filing.cik),
            accession_number=filing.accession_no,
            form_type=form_type,
            filing_date=filing.filing_date,
            fiscal_year_end=_parse_period(filing.period_of_report, ticker, year),
            fiscal_year=year,
            sic_code=str(company.sic) if company.sic else None,
            source_url=filing.filing_url,
            downloaded_at=datetime.now(timezone.utc),
        )

        logger.info("Saved %s %s %d → %s", ticker, form_type, year, raw_path)
        return record, raw_path

    def _save_raw(self, ticker: str, accession_number: str, html: str) -> Path:
        raw_dir = self._config.edgar.raw_data_dir / ticker.upper()
        raw_dir.mkdir(parents=True, exist_ok=True)
        path = raw_dir / f"{accession_number}.html"
        try:
            path.write_text(html, encoding="utf-8")
        except OSError as e:
            raise OSError(
                f"Failed to save raw filing for {ticker} "
                f"(accession: {accession_number}) to {path}: {e}"
            ) from e
        return path
