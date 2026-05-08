"""
Streamlit chat UI for the SEC EDGAR RAG pipeline.

Run with:
    streamlit run app.py
"""

import logging
import sys
from pathlib import Path

# Must be first — ensures all project-local imports resolve correctly regardless
# of where streamlit run is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from config.loader import load_config
from db.factory import create_db_client, create_filings_repo
from etl.factory import create_embedder
from generation.factory import build_generation_pipeline, make_initial_state
from generation.types import Citation, GenerationResult
from llm.types import Message
from tracing.setup import setup_tracing

logger = logging.getLogger(__name__)

SUGGESTED_QUESTIONS = [
    "What were NVIDIA's main risk factors in its most recent 10-K?",
    "How did NVIDIA describe demand for its AI and data center chips?",
    "What did NVIDIA say about export controls and geopolitical risks?",
    "How has NVIDIA's revenue and gross margin changed over recent years?",
    "What competition does NVIDIA face in the GPU and AI accelerator market?",
    "What did NVIDIA disclose about its supply chain and manufacturing partners?",
]


def _citation_to_dict(c: Citation) -> dict:
    return {
        "ticker": c.ticker,
        "company_name": c.company_name,
        "form_type": c.form_type,
        "fiscal_year_end": str(c.fiscal_year_end) if c.fiscal_year_end else None,
        "filing_date": str(c.filing_date),
        "accession_number": c.accession_number,
        "source_url": c.source_url,
        "section": c.section,
        "chunk_text": c.chunk_text,
    }


@st.cache_resource(show_spinner="Loading pipeline...")
def load_pipeline():
    config = load_config()
    setup_tracing(config.tracing)
    client = create_db_client(config)
    if not client.health_check():
        raise RuntimeError("Database health check failed — is PostgreSQL running?")
    embedder = create_embedder(config)
    graph = build_generation_pipeline(config, client, embedder)
    indexed = create_filings_repo(client).list_indexed_summary()
    return config, client, graph, indexed


def render_citations(citations: list[dict]) -> None:
    if not citations:
        return
    st.markdown(f"**Sources ({len(citations)})**")
    for i, c in enumerate(citations, 1):
        label = (
            f"[{i}] {c['company_name']} ({c['ticker']})"
            f" · {c['form_type']}"
            f" · filed {c['filing_date']}"
            f" · {c['section']}"
        )
        with st.expander(label):
            url = c.get("source_url", "")
            if url and url.startswith("https://"):
                st.markdown(f"[View filing on SEC EDGAR]({url})")
            st.text(c["chunk_text"])


def submit_query(query: str, graph) -> None:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            with st.status("Processing...", expanded=True) as status:
                state = make_initial_state(query, history=st.session_state.history)
                result: GenerationResult | None = None
                for mode, data in graph.stream(state, stream_mode=["updates", "custom"]):
                    if mode == "custom":
                        status.write(data)
                    elif mode == "updates":
                        for node_output in data.values():
                            if isinstance(node_output, dict) and "answer" in node_output:
                                result = node_output["answer"]
                status.update(label="Done", state="complete", expanded=False)
        except Exception:
            logger.exception("Pipeline error for query: %r", query)
            content = "Something went wrong while processing your question. Please try again."
            st.error(content)
            st.session_state.messages.append({"role": "assistant", "content": content})
            return

        if result is None or not result.answer.strip():
            content = "I couldn't generate an answer for that question. Try rephrasing."
            st.markdown(content)
            st.session_state.messages.append({"role": "assistant", "content": content})
            return

        st.markdown(result.answer)
        citations = [_citation_to_dict(c) for c in result.citations]
        render_citations(citations)
        st.caption(
            f"{result.usage.input_tokens} in / {result.usage.output_tokens} out tokens"
        )

        st.session_state.messages.append({
            "role": "assistant",
            "content": result.answer,
            "citations": citations,
            "usage": {
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
            },
        })
        st.session_state.history.append(Message(role="user", content=query))
        st.session_state.history.append(Message(role="assistant", content=result.answer))


def main() -> None:
    # st.set_page_config must be the first st.* call in the script.
    st.set_page_config(
        page_title="Financial 10-K Q&A",
        page_icon="📊",
        layout="wide",
    )

    try:
        config, _db_client, graph, indexed = load_pipeline()
    except Exception:
        logger.exception("Failed to initialize pipeline")
        st.error("Failed to initialize the pipeline. Check server logs for details.")
        st.stop()

    # Initialize session state before any rendering so sidebar and other
    # components can safely read from it.
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history" not in st.session_state:
        st.session_state.history = []

    with st.sidebar:
        st.title("Financial 10-K Q&A")
        st.markdown("**Indexed filings**")
        for ticker, company_name, years in indexed:
            years_str = ", ".join(str(y) for y in years)
            st.markdown(f"- `{ticker}` — {company_name} ({years_str})")
        st.divider()
        if st.button("New chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.history = []
            st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                render_citations(msg["citations"])
            if msg.get("usage"):
                u = msg["usage"]
                st.caption(f"{u['input_tokens']} in / {u['output_tokens']} out tokens")

    if not st.session_state.messages:
        st.markdown("#### Try asking:")
        cols = st.columns(2)
        clicked = None
        for i, q in enumerate(SUGGESTED_QUESTIONS):
            if cols[i % 2].button(q, key=f"sug_{i}", use_container_width=True):
                clicked = q
        if clicked:
            submit_query(clicked, graph)
            st.rerun()

    if prompt := st.chat_input("Ask a question about SEC filings..."):
        submit_query(prompt, graph)


if __name__ == "__main__":
    main()
