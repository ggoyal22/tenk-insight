from retrieval.types import RetrievalResult


def build_context(results: list[RetrievalResult]) -> str:
    """Format retrieval results into a labelled context string for the LLM.

    Each result is prefixed with its source filing so the LLM can cite it.
    Expects a flat list — callers are responsible for deduplication and ordering.
    """
    parts = []
    for r in results:
        f = r.filing
        label = (
            f"[{f.ticker} | {f.form_type} | "
            f"{f.fiscal_year_end or f.filing_date} | {r.parent_chunk.section}]"
        )
        parts.append(f"{label}\n{r.parent_chunk.text}")
    return "\n\n---\n\n".join(parts) if parts else "No context retrieved."
