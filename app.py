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
from db.factory import create_db_client, create_feedback_repo, create_filings_repo
from db.repositories.feedback import FeedbackRepo
from etl.factory import create_embedder
from generation.factory import build_generation_pipeline, make_initial_state
from generation.types import Citation, GenerationResult, total_pipeline_usage
from llm.pricing import compute_cost
from llm.types import LLMUsage, Message
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
        "index": c.index,
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
    setup_tracing(config.tracing, project_name=f"ui-{config.environment}")
    client = create_db_client(config)
    if not client.health_check():
        raise RuntimeError("Database health check failed — is PostgreSQL running?")
    embedder = create_embedder(config)
    graph = build_generation_pipeline(config, client, embedder)
    indexed = create_filings_repo(client).list_indexed_summary()
    feedback_repo = create_feedback_repo(client)
    feedback_repo.create_table()
    return config, client, graph, indexed, feedback_repo


def render_citations(citations: list[dict]) -> None:
    if not citations:
        return
    st.markdown(f"**Sources ({len(citations)})**")
    for c in citations:
        label = (
            f"[{c['index']}] {c['company_name']} ({c['ticker']})"
            f" · {c['form_type']}"
            f" · filed {c['filing_date']}"
            f" · {c['section']}"
        )
        with st.expander(label):
            url = c.get("source_url", "")
            if url and url.startswith("https://"):
                st.markdown(f"[View filing on SEC EDGAR]({url})")
            st.text(c["chunk_text"])


def render_feedback(msg_index: int, query: str, answer: str, repo: FeedbackRepo) -> None:
    submitted_key = f"fb_submitted_{msg_index}"

    if st.session_state.get(submitted_key):
        return

    st.caption("Rate this answer")
    rating = st.feedback("thumbs", key=f"fb_{msg_index}")

    if rating is not None:
        comment = st.text_area(
            "comment", key=f"fb_comment_{msg_index}", height=68,
            label_visibility="collapsed", placeholder="Add a comment (optional)",
        )
        if st.button("Submit", key=f"fb_submit_{msg_index}", type="tertiary"):
            repo.insert_feedback(
                query=query,
                answer=answer,
                rating=bool(rating),
                comment=comment.strip() or None,
            )
            st.session_state[submitted_key] = True
            st.rerun()


def submit_query(query: str, graph, config, feedback_repo: FeedbackRepo) -> None:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            with st.status("Processing...", expanded=True) as status:
                state = make_initial_state(query, history=st.session_state.history)
                result: GenerationResult | None = None
                pipeline_usage: list[LLMUsage] = []
                for mode, data in graph.stream(state, stream_mode=["updates", "custom"]):
                    if mode == "custom":
                        status.write(data)
                    elif mode == "updates":
                        for node_output in data.values():
                            if not isinstance(node_output, dict):
                                continue
                            if "answer" in node_output:
                                result = node_output["answer"]
                            if "pipeline_usage" in node_output:
                                pipeline_usage.extend(node_output["pipeline_usage"])
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

        usage = total_pipeline_usage(pipeline_usage)
        cost = compute_cost(usage, config.llm.model, config.llm.provider)
        cost_str = f"${cost:.4f}" if isinstance(cost, float) else cost
        st.markdown(result.answer)
        citations = [_citation_to_dict(c) for c in result.citations]
        render_citations(citations)
        total = usage.input_tokens + usage.output_tokens
        st.caption(
            f"Cost: {cost_str}  ·  {total:,} tokens"
            f"  ({usage.input_tokens:,} prompt / {usage.output_tokens:,} completion)"
        )

        st.session_state.messages.append({
            "role": "assistant",
            "content": result.answer,
            "citations": citations,
            "query": query,
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cost_str": cost_str,
            },
        })
        msg_index = len(st.session_state.messages) - 1
        render_feedback(msg_index, query, result.answer, feedback_repo)
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
        config, _db_client, graph, indexed, feedback_repo = load_pipeline()
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
            for key in [k for k in st.session_state if k.startswith("fb_")]:
                del st.session_state[key]
            st.rerun()

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                render_citations(msg["citations"])
            if msg.get("usage"):
                u = msg["usage"]
                total = u["input_tokens"] + u["output_tokens"]
                cost_part = f"Cost: {u['cost_str']}  ·  " if u.get("cost_str") else ""
                st.caption(
                    f"{cost_part}{total:,} tokens"
                    f"  ({u['input_tokens']:,} prompt / {u['output_tokens']:,} completion)"
                )
            if msg["role"] == "assistant" and msg.get("query"):
                render_feedback(i, msg["query"], msg["content"], feedback_repo)

    if not st.session_state.messages:
        st.markdown("#### Try asking:")
        cols = st.columns(2)
        clicked = None
        for i, q in enumerate(SUGGESTED_QUESTIONS):
            if cols[i % 2].button(q, key=f"sug_{i}", use_container_width=True):
                clicked = q
        if clicked:
            submit_query(clicked, graph, config, feedback_repo)
            st.rerun()

    st.caption(f"Generation model: {config.llm.model}")
    if prompt := st.chat_input("Ask a question about SEC filings..."):
        submit_query(prompt, graph, config, feedback_repo)


if __name__ == "__main__":
    main()
