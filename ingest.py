import logging
import sys

from config.loader import ensure_directories, load_config
from db.factory import (
    create_chunks_repo,
    create_db_client,
    create_filings_repo,
    create_parent_chunks_repo,
    create_vector_store,
)
from etl.factory import (
    create_chunker,
    create_downloader,
    create_embedder,
    create_loader,
    create_parser,
)
from etl.pipeline import Pipeline


def main() -> None:
    try:
        config = load_config()
    except Exception as exc:
        sys.stderr.write(f"CRITICAL — Failed to load config: {exc}\n")
        sys.exit(1)

    logging.basicConfig(
        level=config.logging.level,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    ensure_directories(config)

    client = create_db_client(config)
    if not client.health_check():
        logging.critical("Database health check failed — aborting ingestion")
        client.close()
        sys.exit(1)

    try:
        filings_repo = create_filings_repo(client)
        parent_chunks_repo = create_parent_chunks_repo(client)
        chunks_repo = create_chunks_repo(client)
        vector_store = create_vector_store(config, client)

        downloader = create_downloader(config)
        parser = create_parser()
        chunker = create_chunker(config)
        embedder = create_embedder(config)
        loader = create_loader(client, filings_repo, parent_chunks_repo, chunks_repo, vector_store)

        pipeline = Pipeline(downloader, parser, chunker, embedder, loader)
        pipeline.run(
            tickers=config.edgar.tickers,
            form_types=config.edgar.form_types,
            years=config.edgar.years,
        )
    except Exception:
        logging.critical("Ingestion failed with unhandled error", exc_info=True)
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()
