from pathlib import Path
from unittest.mock import MagicMock

import pytest

from etl.downloader.base import FilingNotFoundError
from etl.pipeline import Pipeline
from etl.types import ChildChunk, ParentChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipeline() -> tuple[Pipeline, MagicMock, MagicMock, MagicMock, MagicMock, MagicMock]:
    downloader = MagicMock()
    parser = MagicMock()
    chunker = MagicMock()
    embedder = MagicMock()
    loader = MagicMock()
    pipeline = Pipeline(downloader, parser, chunker, embedder, loader)
    return pipeline, downloader, parser, chunker, embedder, loader


def _make_parent() -> ParentChunk:
    return ParentChunk(
        section_name="Item 1",
        content_type="narrative",
        text="parent text",
        token_count=2,
        filing_chunk_index=0,
    )


def _make_child(idx: int = 0) -> ChildChunk:
    return ChildChunk(
        section_name="Item 1",
        content_type="narrative",
        text="child text",
        token_count=2,
        filing_chunk_index=idx,
        parent_chunk_index=0,
    )


def _wire_success(downloader, parser, chunker, embedder, loader, n_children: int = 1):
    filing = MagicMock()
    downloader.fetch.return_value = (filing, Path("/tmp/filing.html"))
    parser.parse.return_value = [MagicMock()]
    children = [_make_child(i) for i in range(n_children)]
    chunker.chunk.return_value = ([_make_parent()], children)
    embedder.embed.return_value = [[0.1] * 4] * n_children
    embedder.model_name = "test-model"
    loader.is_fully_ingested.return_value = False
    loader.load.return_value = MagicMock()
    return filing, children


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_run_processes_all_ticker_form_year_combinations():
    pipeline, downloader, parser, chunker, embedder, loader = _make_pipeline()
    _wire_success(downloader, parser, chunker, embedder, loader)

    pipeline.run(tickers=["NVDA", "AAPL"], form_types=["10-K"], years=[2024])

    assert downloader.fetch.call_count == 2
    assert loader.load.call_count == 2


def test_run_skips_when_filing_not_found():
    pipeline, downloader, parser, chunker, embedder, loader = _make_pipeline()
    downloader.fetch.side_effect = FilingNotFoundError("not found")

    pipeline.run(tickers=["FAKE"], form_types=["10-K"], years=[2024])

    parser.parse.assert_not_called()
    loader.load.assert_not_called()


def test_run_skips_before_embedding_when_already_fully_ingested():
    pipeline, downloader, parser, chunker, embedder, loader = _make_pipeline()
    _wire_success(downloader, parser, chunker, embedder, loader)
    loader.is_fully_ingested.return_value = True  # filing already complete in DB

    pipeline.run(tickers=["NVDA"], form_types=["10-K"], years=[2024])

    # The skip happens right after fetch — no expensive work runs.
    parser.parse.assert_not_called()
    chunker.chunk.assert_not_called()
    embedder.embed.assert_not_called()
    loader.load.assert_not_called()


def test_run_processes_when_partially_embedded():
    pipeline, downloader, parser, chunker, embedder, loader = _make_pipeline()
    _wire_success(downloader, parser, chunker, embedder, loader)
    loader.is_fully_ingested.return_value = False  # interrupted run, missing embeddings

    pipeline.run(tickers=["NVDA"], form_types=["10-K"], years=[2024])

    embedder.embed.assert_called_once()
    loader.load.assert_called_once()


def test_run_skips_when_chunker_returns_empty():
    pipeline, downloader, parser, chunker, embedder, loader = _make_pipeline()
    filing = MagicMock()
    downloader.fetch.return_value = (filing, Path("/tmp/filing.html"))
    parser.parse.return_value = [MagicMock()]
    chunker.chunk.return_value = ([], [])  # no content extracted

    pipeline.run(tickers=["NVDA"], form_types=["10-K"], years=[2024])

    embedder.embed.assert_not_called()
    loader.load.assert_not_called()


def test_run_marks_as_failed_on_unexpected_exception():
    pipeline, downloader, parser, chunker, embedder, loader = _make_pipeline()
    downloader.fetch.side_effect = RuntimeError("unexpected")

    # Should not raise — pipeline swallows and logs the error.
    pipeline.run(tickers=["NVDA"], form_types=["10-K"], years=[2024])

    loader.load.assert_not_called()


def test_run_raises_when_embedder_returns_wrong_count():
    pipeline, downloader, parser, chunker, embedder, loader = _make_pipeline()
    _wire_success(downloader, parser, chunker, embedder, loader, n_children=2)
    # Return only 1 vector for 2 children → length mismatch → RuntimeError → failure
    embedder.embed.return_value = [[0.1] * 4]  # only 1 vector

    pipeline.run(tickers=["NVDA"], form_types=["10-K"], years=[2024])

    loader.load.assert_not_called()


def test_run_sets_embedding_and_model_on_each_child():
    pipeline, downloader, parser, chunker, embedder, loader = _make_pipeline()
    _, children = _wire_success(downloader, parser, chunker, embedder, loader, n_children=2)
    embedder.embed.return_value = [[0.1] * 4, [0.2] * 4]
    embedder.model_name = "bge-model"

    pipeline.run(tickers=["NVDA"], form_types=["10-K"], years=[2024])

    assert children[0].embedding == [0.1] * 4
    assert children[0].embedding_model == "bge-model"
    assert children[1].embedding == [0.2] * 4
    assert children[1].embedding_model == "bge-model"
