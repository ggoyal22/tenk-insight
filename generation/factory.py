from config.loader import AppConfig
from db.client.base import DatabaseClient
from db.factory import create_filings_repo
from etl.embedder.base import Embedder
from generation.graph import build_graph
from generation.nodes import (
    make_check_hop,
    make_classify_query,
    make_generate,
    make_hyde_expand,
    make_plan_tasks,
    make_reflect,
    make_retrieve,
)
from generation.prompt_registry import load_prompts
from llm.factory import build_llm
from retrieval.factory import build_retriever_from_config


def build_generation_pipeline(config: AppConfig, db_client: DatabaseClient, embedder: Embedder):
    """Build and return the compiled generation graph.

    This is the single entry point for assembling the pipeline. It wires together
    the LLM, retriever, embedder, and all node functions, then compiles the graph.

    Args:
        config:     Full application config (llm, generation, and retrieval sections are used).
        db_client:  Live database client used to build the retriever.
        embedder:   Loaded embedding model used to embed queries before retrieval.

    Returns:
        A compiled LangGraph graph that accepts a GenerationState dict and returns
        the updated state with an answer field populated.
    """
    llm = build_llm(config.llm)
    filings_repo = create_filings_repo(db_client)
    retriever = build_retriever_from_config(config, db_client)
    prompts = load_prompts(tag=config.prompts.tag)

    plan_prompts = {
        "single": prompts.plan_single,
        "comparison": prompts.plan_comparison,
        "time_series": prompts.plan_time_series,
        "multi_hop": prompts.plan_multi_hop,
    }

    return build_graph(
        classify_query_fn=make_classify_query(llm, prompts.classify),
        plan_tasks_fn=make_plan_tasks(llm, plan_prompts, filings_repo),
        hyde_expand_fn=make_hyde_expand(llm, prompts.hyde),
        retrieve_fn=make_retrieve(retriever, embedder),
        generate_fn=make_generate(llm, prompts.qa, prompts.comparison, prompts.time_series),
        check_hop_fn=make_check_hop(llm, config.generation, prompts.check_hop),
        reflect_fn=make_reflect(llm, config.generation, prompts.reflection),
        config=config.generation,
    )


def make_initial_state(query: str, history=None, query_filter=None) -> dict:
    """Build the initial state dict for invoking the generation graph."""
    return {
        "query": query,
        "history": history or [],
        "query_filter": query_filter,
        "query_type": "",
        "resolved_query": None,
        "pending_tasks": [],
        "hyde_query": None,
        "completed_results": [],
        "pipeline_usage": [],
        "hop_count": 0,
        "reflection_count": 0,
        "retrieval_triggered_by": "analysis",
        "answer": None,
    }
