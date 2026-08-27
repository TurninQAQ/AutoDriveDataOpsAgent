from __future__ import annotations

from typing import Any

from deploy_ci_cloud_agentv2.platform_backend.runtime import build_platform_facade
from deploy_ci_cloud_agentv3.agent.context_builder import ContextBuilder
from deploy_ci_cloud_agentv3.agent.final_guard import FinalGuard
from deploy_ci_cloud_agentv3.agent.graph import GraphDependencies, build_graph
from deploy_ci_cloud_agentv3.agent.prompts import SYSTEM_PROMPT
from deploy_ci_cloud_agentv3.mcp.client import OfficialMCPClient
from deploy_ci_cloud_agentv3.mcp.server import build_mcp_servers
from deploy_ci_cloud_agentv3.services.audit import AuditStore
from deploy_ci_cloud_agentv3.services.pending_action import PendingActionFactory
from deploy_ci_cloud_agentv3.services.verification import VerificationService
from deploy_ci_cloud_agentv3.services.write_service import WriteService


def _ensure_checkpointer(checkpointer: Any | None):
    if checkpointer is not None:
        return checkpointer
    try:
        from langgraph.checkpoint.memory import InMemorySaver
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("langgraph is required for resumable HITL execution") from exc
    return InMemorySaver()


def _build_runtime(
    cls,
    provider: Any,
    *,
    agent_mcp: Any,
    runtime_mcp: Any,
    checkpointer: Any | None,
    audit_path: str | None,
):
    deps = GraphDependencies(
        provider=provider,
        agent_mcp=agent_mcp,
        pending_factory=PendingActionFactory(runtime_mcp),
        write_service=WriteService(
            runtime_mcp, VerificationService(runtime_mcp), AuditStore(audit_path)
        ),
        context_builder=ContextBuilder(SYSTEM_PROMPT),
        final_guard=FinalGuard(),
    )
    return cls(build_graph(deps, checkpointer=_ensure_checkpointer(checkpointer)))


class AgentRuntime:
    def __init__(self, graph: Any) -> None:
        self.graph = graph

    @classmethod
    def local(
        cls,
        provider: Any,
        *,
        facade: Any | None = None,
        checkpointer: Any | None = None,
        audit_path: str | None = None,
    ):
        """Local mainline: official MCP Client(MCPServer) in-process transport."""
        facade = facade or build_platform_facade()
        agent_server, runtime_server, _ = build_mcp_servers(facade)
        return _build_runtime(
            cls,
            provider,
            agent_mcp=OfficialMCPClient(agent_server),
            runtime_mcp=OfficialMCPClient(runtime_server),
            checkpointer=checkpointer,
            audit_path=audit_path,
        )

    @classmethod
    def in_process(
        cls,
        provider: Any,
        *,
        facade: Any | None = None,
        checkpointer: Any | None = None,
        audit_path: str | None = None,
    ):
        """Backward-compatible alias for the official MCP local mainline."""
        return cls.local(
            provider,
            facade=facade,
            checkpointer=checkpointer,
            audit_path=audit_path,
        )

    @classmethod
    def remote(
        cls,
        provider: Any,
        *,
        agent_mcp_url: str,
        runtime_mcp_url: str,
        checkpointer: Any | None = None,
        audit_path: str | None = None,
    ):
        """Remote mainline: official MCP Client(URL) over Streamable HTTP."""
        return _build_runtime(
            cls,
            provider,
            agent_mcp=OfficialMCPClient(agent_mcp_url),
            runtime_mcp=OfficialMCPClient(runtime_mcp_url),
            checkpointer=checkpointer,
            audit_path=audit_path,
        )

    async def start(self, thread_id: str, user_message: str) -> dict[str, Any]:
        state = {
            "thread_id": thread_id,
            "messages": [{"role": "user", "content": user_message}],
            "tool_results": [],
            "pending_action": None,
            "last_write_result": None,
            "prepared_artifact": None,
            "approved_fingerprint": None,
            "review_route": None,
            "final_response": None,
            "step_count": 0,
        }
        return await self.graph.ainvoke(state, {"configurable": {"thread_id": thread_id}})

    async def review(self, thread_id: str, decision: dict[str, Any] | str | bool) -> dict[str, Any]:
        try:
            from langgraph.types import Command
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("langgraph is required") from exc
        return await self.graph.ainvoke(
            Command(resume=decision), {"configurable": {"thread_id": thread_id}}
        )
