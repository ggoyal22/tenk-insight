import logging
from itertools import product

from etl.chunker.base import Chunker
from etl.downloader.base import Downloader, FilingNotFoundError
from etl.embedder.base import Embedder
from etl.loader import Loader
from etl.parser.base import Parser

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(
        self,
        downloader: Downloader,
        parser: Parser,
        chunker: Chunker,
        embedder: Embedder,
        loader: Loader,
    ) -> None:
        self._downloader = downloader
        self._parser = parser
        self._chunker = chunker
        self._embedder = embedder
        self._loader = loader

    def run(self, tickers: list[str], form_types: list[str], years: list[int]) -> None:
        combos = list(product(tickers, form_types, years))
        logger.info("Pipeline starting — %d filing(s) to process", len(combos))
        succeeded = failed = skipped = 0
        for ticker, form_type, year in combos:
            status = self._process_one(ticker, form_type, year)
            if status == "ok":
                succeeded += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
        logger.info(
            "Pipeline complete — %d succeeded, %d skipped, %d failed",
            succeeded, skipped, failed,
        )

    def _process_one(self, ticker: str, form_type: str, year: int) -> str:
        label = f"{ticker} {form_type} {year}"
        try:
            logger.info("Processing %s", label)
            filing, raw_path = self._downloader.fetch(ticker, form_type, year)
            sections = self._parser.parse(raw_path)
            parents, children = self._chunker.chunk(sections, filing)
            if not parents or not children:
                logger.warning("No extractable content for %s — skipping", label)
                return "skipped"

            vectors = self._embedder.embed([c.text for c in children])
            if len(vectors) != len(children):
                raise RuntimeError(
                    f"Embedder returned {len(vectors)} vectors for {len(children)} chunks — "
                    "length mismatch would corrupt embeddings"
                )
            for child, vector in zip(children, vectors):
                child.embedding = vector
                child.embedding_model = self._embedder.model_name

            filing_uuid = self._loader.load(filing, parents, children)
            logger.info("Loaded %s → %s", label, filing_uuid)
            return "ok"
        except FilingNotFoundError:
            logger.warning("No filing found for %s — skipping", label)
            return "skipped"
        except Exception:
            logger.exception("Failed to process %s", label)
            return "failed"
