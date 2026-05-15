from retrieval.types import RetrievalResult


def build_context(results: list[RetrievalResult]) -> str:
    """Format retrieval results into a labelled context string for the LLM.

    Each result is prefixed with its source filing so the LLM can cite it.
    Expects a flat list — callers are responsible for deduplication and ordering.
    """
    parts = []
    for i, r in enumerate(results, start=1):
        f = r.filing
        label = (
            f"[{i}] [{f.ticker} | {f.form_type} | "
            f"{f.fiscal_year_end or f.filing_date} | {r.parent_chunk.section}]"
        )
        parts.append(f"{label}\n{r.parent_chunk.text}")
    return "\n\n---\n\n".join(parts) if parts else "No context retrieved."


def build_hop_context(results: list[RetrievalResult]) -> str:
    """Compact context for the check_hop node using child chunk text.

    Uses the matched child chunk (~256 tokens) rather than the full parent
    (~1024 tokens) — enough to assess coverage gaps without ballooning the prompt.
    """
    parts = []
    for i, r in enumerate(results, start=1):
        f = r.filing
        label = (
            f"[{i}] [{f.ticker} | {f.form_type} | "
            f"{f.fiscal_year_end or f.filing_date} | {r.chunk.section}]"
        )
        parts.append(f"{label}\n{r.chunk.text}")
    return "\n\n---\n\n".join(parts) if parts else "No context retrieved."
