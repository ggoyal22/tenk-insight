import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.loader import load_config
from db.factory import create_db_client
from etl.factory import create_embedder
from generation.factory import build_generation_pipeline, make_initial_state
from generation.types import Citation, total_pipeline_usage
from llm.types import Message
from tracing.setup import setup_tracing


def _format_citations(citations: list[Citation]) -> str:
    if not citations:
        return ""
    lines = ["\nSources:"]
    for c in citations:
        lines.append(
            f"  [{c.index}] {c.company_name} ({c.ticker}) · {c.form_type} "
            f"filed {c.filing_date} · {c.section}"
        )
    return "\n".join(lines)


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

    try:
        setup_tracing(config.tracing)
    except RuntimeError as exc:
        sys.stderr.write(f"CRITICAL — Failed to initialise tracing: {exc}\n")
        sys.exit(1)

    client = create_db_client(config)
    if not client.health_check():
        sys.stderr.write("CRITICAL — Database health check failed\n")
        client.close()
        sys.exit(1)

    try:
        embedder = create_embedder(config)
        graph = build_generation_pipeline(config, client, embedder)

        tickers = ", ".join(config.edgar.tickers)
        print(f"SEC EDGAR Q&A  |  indexed tickers: {tickers}")
        print("Ask a question, or type 'exit' to quit.\n")

        history: list[Message] = []

        while True:
            try:
                query = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not query:
                continue
            if query.lower() in {"exit", "quit"}:
                break

            state = make_initial_state(query, history=history)
            result_state = graph.invoke(state)
            result = result_state.get("answer")

            if result is None:
                print("No answer could be generated for that question.\n")
                continue

            usage = total_pipeline_usage(result_state.get("pipeline_usage") or [])
            print(f"\n{result.answer}")
            print(_format_citations(result.citations))
            print(f"\n[{usage.input_tokens} in / {usage.output_tokens} out tokens]\n")

            history.append(Message(role="user", content=query))
            history.append(Message(role="assistant", content=result.answer))

    except Exception:
        logging.critical("Query session failed", exc_info=True)
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()
