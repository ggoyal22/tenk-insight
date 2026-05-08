from collections.abc import Callable

from langgraph.config import get_stream_writer


def get_writer() -> Callable[[str], None]:
    """Return the LangGraph stream writer, or a no-op when called outside a graph context (e.g. in tests)."""
    try:
        return get_stream_writer()
    except RuntimeError:
        return lambda _: None
