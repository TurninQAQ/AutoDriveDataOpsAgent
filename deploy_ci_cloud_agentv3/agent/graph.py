from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from deploy_ci_cloud_agentv3.agent.context_builder import ContextBuilder
from deploy_ci_cloud_agentv3.agent.final_guard import FinalGuard
from deploy_ci_cloud_agentv3.agent.state import AgentState
from deploy_ci_cloud_agentv3.models.pending_action import PendingAction
from deploy_ci_cloud_agentv3.models.final_response import FinalCandidate
from deploy_ci_cloud_agentv3.models.proposal import ProposalResult
from deploy_ci_cloud_agentv3.models.tool_result import ToolResult
from deploy_ci_cloud_agentv3.models.write_result import WriteResult
from deploy_ci_cloud_agentv3.providers.base import AssistantMessage, ToolCall
from deploy_ci_cloud_agentv3.providers.tool_adapter import mcp_tools_to_native
from deploy_ci_cloud_agentv3.services.pending_action import PendingActionFactory
from deploy_ci_cloud_agentv3.services.write_service import WriteService


@dataclass
class GraphDependencies:
    provider: Any
    agent_mcp: Any
    pending_factory: PendingActionFactory
    write_service: WriteService
    context_builder: ContextBuilder
    final_guard: FinalGuard
    max_steps: int = 32


def build_graph(deps: GraphDependencies, *, checkpointer: Any | None = None):
    try:
        from langgraph.graph import END, START, StateGraph
        from langgraph.types import interrupt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("langgraph is required to build the V3.5 graph") from exc

    async def agent(state: AgentState) -> dict[str, Any]:
        step = int(state.get("step_count") or 0) + 1
        if step > deps.max_steps:
            final = deps.final_guard.build(FinalCandidate(status="write_uncertain", message="Agent step budget exhausted before a safe completion."), state.get("last_write_result"))
            return {"step_count": step, "final_response": final.model_dump(mode="json")}

        mcp_tools = await deps.agent_mcp.list_tools()
        response: AssistantMessage = await deps.provider.invoke(
            deps.context_builder.build(state), mcp_tools_to_native(mcp_tools)
        )
        messages = list(state.get("messages") or [])
        messages.append(_assistant_message(response))
        update: dict[str, Any] = {"messages": messages, "step_count": step, "final_response": None}
        if not response.tool_calls:
            final = deps.final_guard.build(response.content, state.get("last_write_result"))
            update["final_response"] = final.model_dump(mode="json")
        return update

    def route_agent(state: AgentState) -> str:
        last = (state.get("messages") or [])[-1] if state.get("messages") else {}
        return "model_tools" if last.get("tool_calls") else END

    async def model_tools(state: AgentState) -> dict[str, Any]:
        last = (state.get("messages") or [])[-1]
        calls = [_tool_call_from_wire(item) for item in last.get("tool_calls") or []]
        proposal_calls = [call for call in calls if call.name.startswith("propose_")]
        messages = list(state.get("messages") or [])
        tool_results = list(state.get("tool_results") or [])

        if proposal_calls and (len(proposal_calls) != 1 or len(calls) != 1):
            # Native Function Calling requires one Tool message for every tool_call_id
            # emitted by the assistant, even when policy rejects the whole round.
            rejected_results, rejected_messages = _proposal_policy_rejection(calls)
            tool_results.extend(item.model_dump(mode="json") for item in rejected_results)
            messages.extend(rejected_messages)
            return {"messages": messages, "tool_results": tool_results, "pending_action": None}

        if proposal_calls:
            call = proposal_calls[0]
            try:
                raw = await deps.agent_mcp.call_tool(call.name, call.arguments)
                proposal = ProposalResult.model_validate(raw)
                pending = await deps.pending_factory.build(proposal)
                deps.write_service.audit.append("ProposalCreated", pending.model_dump(mode="json"))
                result = ToolResult(kind="ACTION_PROPOSAL", tool_name=call.name, call_id=call.id, data=proposal.model_dump(mode="json"))
                tool_results.append(result.model_dump(mode="json"))
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result.model_dump_json()})
                return {"messages": messages, "tool_results": tool_results, "pending_action": pending.model_dump(mode="json")}
            except Exception as exc:
                error = ToolResult(kind="TOOL_ERROR", tool_name=call.name, call_id=call.id, error=f"{type(exc).__name__}: {exc}")
                tool_results.append(error.model_dump(mode="json"))
                messages.append({"role": "tool", "tool_call_id": call.id, "content": error.model_dump_json()})
                return {"messages": messages, "tool_results": tool_results, "pending_action": None}

        async def execute(call: ToolCall) -> tuple[ToolCall, Any, Exception | None]:
            try:
                return call, await deps.agent_mcp.call_tool(call.name, call.arguments), None
            except Exception as exc:
                return call, None, exc

        completed = await asyncio.gather(*(execute(call) for call in calls))
        prepared = state.get("prepared_artifact")
        for call, raw, exc in completed:
            if exc is not None:
                result = ToolResult(kind="TOOL_ERROR", tool_name=call.name, call_id=call.id, error=f"{type(exc).__name__}: {exc}")
            else:
                kind = "PREPARED_ARTIFACT" if call.name == "prepare_task_spec" else "OBSERVATION"
                data = raw if isinstance(raw, dict) else {"result": raw}
                result = ToolResult(kind=kind, tool_name=call.name, call_id=call.id, data=data)
                if call.name == "prepare_task_spec" and isinstance(raw, dict):
                    prepared = raw
            tool_results.append(result.model_dump(mode="json"))
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result.model_dump_json()})
        return {"messages": messages, "tool_results": tool_results, "prepared_artifact": prepared}

    def route_tools(state: AgentState) -> str:
        return "review" if state.get("pending_action") else "agent"

    async def review(state: AgentState) -> dict[str, Any]:
        pending = PendingAction.model_validate(state["pending_action"])
        decision = interrupt(
            {
                "kind": "WRITE_REVIEW",
                "proposal_id": pending.proposal_id,
                "action": pending.action,
                "args": pending.args,
                "before": pending.before,
                "action_precondition": pending.action_precondition,
                "artifact": pending.artifact,
                "reason": pending.reason,
                "expected_effect": pending.expected_effect,
                "fingerprint": pending.fingerprint,
                "options": ["approve", "reject", "edit"],
            }
        )
        parsed = _parse_review(decision)
        if parsed["decision"] == "approve":
            approved = str(parsed.get("fingerprint") or "")
            if approved != pending.fingerprint:
                deps.write_service.audit.append("ReviewApprovalMismatch", {"proposal_id": pending.proposal_id, "approved_fingerprint": approved, "pending_fingerprint": pending.fingerprint})
                return {"approved_fingerprint": None, "review_route": "review"}
            deps.write_service.audit.append("ReviewApproved", {"proposal_id": pending.proposal_id, "fingerprint": approved})
            return {"approved_fingerprint": approved, "review_route": "execute_write"}
        if parsed["decision"] == "edit":
            edited = parsed.get("args")
            if not isinstance(edited, dict):
                return {"review_route": "review"}
            rebuilt = await deps.pending_factory.rebuild_from_edit(pending, edited)
            deps.write_service.audit.append("ReviewEdited", {"old_proposal_id": pending.proposal_id, "old_fingerprint": pending.fingerprint, "new_proposal_id": rebuilt.proposal_id, "new_fingerprint": rebuilt.fingerprint})
            return {"pending_action": rebuilt.model_dump(mode="json"), "approved_fingerprint": None, "review_route": "review"}

        deps.write_service.audit.append("ReviewRejected", {"proposal_id": pending.proposal_id, "fingerprint": pending.fingerprint, "reason": str(parsed.get("reason") or "rejected by human reviewer")})
        rejected = WriteResult(
            id=f"write_{uuid.uuid4().hex}", action=pending.action, status="REJECTED",
            verified=False, before=pending.before, after={}, error=str(parsed.get("reason") or "rejected by human reviewer"),
        )
        return {
            "pending_action": None, "approved_fingerprint": None,
            "last_write_result": rejected.model_dump(mode="json"), "review_route": "agent",
        }

    def route_review(state: AgentState) -> str:
        return str(state.get("review_route") or "review")

    async def execute_write(state: AgentState) -> dict[str, Any]:
        pending = PendingAction.model_validate(state["pending_action"])
        try:
            result = await deps.write_service.execute(pending, str(state.get("approved_fingerprint") or ""))
        except PermissionError as exc:
            result = WriteResult(
                id=f"write_{uuid.uuid4().hex}", action=pending.action, status="FAILED", verified=False,
                before=pending.before, after={}, error=f"approval validation failed: {exc}",
            )
            deps.write_service.audit.append("WriteBlocked", {"proposal_id": pending.proposal_id, "fingerprint": pending.fingerprint, "error": str(exc)})
        messages = list(state.get("messages") or [])
        messages.append({"role": "system", "content": f"Deterministic WriteResult: {result.model_dump(mode='json')}"})
        tool_results = list(state.get("tool_results") or [])
        tool_results.append(ToolResult(kind="WRITE_RESULT", tool_name=pending.action, data=result.model_dump(mode="json")).model_dump(mode="json"))
        return {
            "messages": messages,
            "tool_results": tool_results,
            "pending_action": None,
            "approved_fingerprint": None,
            "last_write_result": result.model_dump(mode="json"),
        }

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent)
    builder.add_node("model_tools", model_tools)
    builder.add_node("review", review)
    builder.add_node("execute_write", execute_write)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_agent, {"model_tools": "model_tools", END: END})
    builder.add_conditional_edges("model_tools", route_tools, {"review": "review", "agent": "agent"})
    builder.add_conditional_edges("review", route_review, {"review": "review", "execute_write": "execute_write", "agent": "agent"})
    builder.add_edge("execute_write", "agent")
    return builder.compile(checkpointer=checkpointer)



def _proposal_policy_rejection(calls: list[ToolCall]) -> tuple[list[ToolResult], list[dict[str, Any]]]:
    policy_error = (
        "Proposal must be the only tool call in its round; "
        "multiple/mixed proposal calls are rejected."
    )
    results = [
        ToolResult(
            kind="TOOL_ERROR",
            tool_name=call.name,
            call_id=call.id,
            error=policy_error,
        )
        for call in calls
    ]
    messages = [
        {
            "role": "tool",
            "tool_call_id": result.call_id,
            "content": result.model_dump_json(),
        }
        for result in results
    ]
    return results, messages

def _assistant_message(message: AssistantMessage) -> dict[str, Any]:
    wire: dict[str, Any] = {"role": "assistant", "content": message.content}
    if message.tool_calls:
        wire["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)},
            }
            for call in message.tool_calls
        ]
    return wire


def _tool_call_from_wire(item: dict[str, Any]) -> ToolCall:
    function = item.get("function") or {}
    args = function.get("arguments") or {}
    if isinstance(args, str):
        args = json.loads(args)
    return ToolCall(id=str(item.get("id") or ""), name=str(function.get("name") or ""), arguments=dict(args))


def _parse_review(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return {"decision": value.strip().lower()}
    if isinstance(value, bool):
        return {"decision": "approve" if value else "reject"}
    if isinstance(value, dict):
        decision = str(value.get("decision") or value.get("type") or "").strip().lower()
        return {**value, "decision": decision}
    return {"decision": "reject", "reason": "invalid review response"}
