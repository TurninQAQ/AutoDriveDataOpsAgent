"""Assertions that the integration suite is using real LangGraph, not shim code."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path

import pytest


pytestmark = pytest.mark.real_langgraph


def test_real_langgraph_1_2_11_is_loaded_from_site_packages() -> None:
    package_version = importlib.metadata.version("langgraph")
    assert package_version == "1.2.11"
    import langgraph
    import langgraph.graph

    assert not getattr(langgraph, "__v2_test_compat__", False)
    module_path = Path(langgraph.graph.__file__).resolve()
    assert "site-packages" in module_path.parts
    assert "tests" not in module_path.parts


def test_real_langgraph_interrupt_resume_api_is_available() -> None:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command, interrupt

    graph_builder = StateGraph(dict)

    def pause(state: dict) -> dict:
        answer = interrupt({"kind": "integration-test"})
        return {"answer": answer}

    graph_builder.add_node("pause", pause)
    graph_builder.add_edge(START, "pause")
    graph_builder.add_edge("pause", END)
    graph = graph_builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "real-langgraph-test"}}
    first = graph.invoke({}, config=config)
    assert "__interrupt__" in first
    second = graph.invoke(Command(resume={"approved": True}), config=config)
    assert second["answer"] == {"approved": True}
