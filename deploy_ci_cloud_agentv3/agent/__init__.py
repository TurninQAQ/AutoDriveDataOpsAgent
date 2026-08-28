"""V3 Agent package with lazy imports so non-Agent utilities do not require MCP/LangGraph at import time."""

__all__ = ["AgentRuntime", "AgentState", "GraphDependencies", "build_graph"]


def __getattr__(name):
    if name == "AgentRuntime":
        from .runtime import AgentRuntime
        return AgentRuntime
    if name == "AgentState":
        from .state import AgentState
        return AgentState
    if name in {"GraphDependencies", "build_graph"}:
        from .graph import GraphDependencies, build_graph
        return {"GraphDependencies": GraphDependencies, "build_graph": build_graph}[name]
    raise AttributeError(name)
