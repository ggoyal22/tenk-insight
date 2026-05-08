import logging
from collections.abc import Callable

from langgraph.graph import END, StateGraph
from langgraph.types import Send

from config.loader import GenerationConfig
from generation.nodes.retrieve import RetrieveInput
from generation.types import GenerationState

logger = logging.getLogger(__name__)


def build_graph(
    analyze_query_fn: Callable,
    hyde_expand_fn: Callable,
    retrieve_fn: Callable,
    generate_fn: Callable,
    check_hop_fn: Callable,
    reflect_fn: Callable,
    config: GenerationConfig,
):
    """Wire nodes and edge routing into a compiled LangGraph graph.

    All node functions are passed in (rather than imported directly) so the
    graph has no knowledge of how dependencies are injected — that is the
    factory's responsibility.
    """

    # ── Edge routing functions ────────────────────────────────────────────────
    # Each function reads state and returns either a node name (str), END, or
    # a list of Send objects for fan-out to the retrieve node.

    def route_after_analyze(state: GenerationState):
        if config.eval_stop_after == "analyze_query":
            return END
        tasks = state["pending_tasks"]
        if not tasks:
            logger.debug("analyze_query produced no tasks — terminating early.")
            return END
        if config.hyde.enabled:
            return "hyde_expand"
        return [Send("retrieve", {"task": t, "hyde_query": None}) for t in tasks]

    def route_after_hyde(state: GenerationState):
        if config.eval_stop_after == "hyde_expand":
            return END
        tasks = state["pending_tasks"]
        if not tasks:
            logger.warning(
                "Query type '%s' produced no retrieval tasks after HyDE expansion — terminating without answer.",
                state["query_type"],
            )
            return END
        return [Send("retrieve", {"task": t, "hyde_query": state["hyde_query"]}) for t in tasks]

    def route_after_retrieve(state: GenerationState):
        if config.eval_stop_after == "retrieve":
            return END
        # Route to check_hop only for multi-hop queries where we haven't exceeded
        # the hop limit and retrieval wasn't triggered by reflection (which already
        # has its own quality loop and doesn't need multi-hop chaining on top).
        if (
            state["query_type"] == "multi_hop"
            and state["hop_count"] < config.multi_hop.max_hops
            and state.get("retrieval_triggered_by") != "reflect"
        ):
            return "check_hop"
        return "generate"

    def route_after_check_hop(state: GenerationState):
        if config.eval_stop_after == "check_hop":
            return END
        tasks = state["pending_tasks"]
        if tasks:
            return [Send("retrieve", {"task": t, "hyde_query": state.get("hyde_query")}) for t in tasks]
        return "generate"

    def route_after_generate(state: GenerationState):
        if config.eval_stop_after == "generate":
            return END
        if config.reflection.enabled:
            return "reflect"
        return END

    def route_after_reflect(state: GenerationState):
        if config.eval_stop_after == "reflect":
            return END
        tasks = state["pending_tasks"]
        if tasks and state["reflection_count"] < config.reflection.max_iterations:
            return [Send("retrieve", {"task": t, "hyde_query": state.get("hyde_query")}) for t in tasks]
        return END

    # ── Graph assembly ────────────────────────────────────────────────────────

    g = StateGraph(GenerationState)

    g.add_node("analyze_query", analyze_query_fn)
    g.add_node("hyde_expand", hyde_expand_fn)
    g.add_node("retrieve", retrieve_fn)
    g.add_node("generate", generate_fn)
    g.add_node("check_hop", check_hop_fn)
    g.add_node("reflect", reflect_fn)

    g.set_entry_point("analyze_query")
    g.add_conditional_edges("analyze_query", route_after_analyze)
    g.add_conditional_edges("hyde_expand", route_after_hyde)
    g.add_conditional_edges("retrieve", route_after_retrieve)
    g.add_conditional_edges("check_hop", route_after_check_hop)
    g.add_conditional_edges("generate", route_after_generate)
    g.add_conditional_edges("reflect", route_after_reflect)

    return g.compile()
