"""Stable host boundary for the Phase B graph."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .budgets import RuntimeBudgets
from .context import ContextBuilder
from .contracts import CompletionContractCompiler
from .events import EventProvenance, EventStore
from .evidence import EvidenceTracker
from .gate import ResponseCompletionGate
from .graph import GraphDependencies, build_graph
from .principles import load_operating_principles
from .state import AgentState, InMemoryCheckpointer, new_state
from ..platform.facade import InMemoryReadFacade, ReadFacade
from ..providers.deterministic import DeterministicReadAgent
from ..providers.model import AgentProvider
from ..tools.catalog import build_read_registry
from ..tools.registry import ToolRegistry
from ..tools.runtime import ReadToolRuntime


@dataclass(frozen=True)
class SystemContext:
    runtime_version: str
    environment: str
    operator_id: str
    trust_domain: str
    tool_catalog_hash: str
    policy_version: str
    event_store: EventStore
    checkpointer: InMemoryCheckpointer
    provider: AgentProvider
    read_facade: ReadFacade
    tool_registry: ToolRegistry
    principles_path: str
    budgets: RuntimeBudgets


@dataclass(frozen=True)
class AgentRunResult:
    thread_id: str
    request_id: str
    status: str
    response: str | None
    goal_outcomes: tuple[Any, ...]
    pending_interrupt: object | None = None
    terminal_outcome: object | None = None
    state: AgentState | None = None


def build_system_context(
    provider: AgentProvider | None = None,
    *,
    read_facade: ReadFacade | None = None,
    event_store: EventStore | None = None,
    checkpointer: InMemoryCheckpointer | None = None,
    budgets: RuntimeBudgets | None = None,
    principles_path: str | Path | None = None,
    environment: str = "offline",
    operator_id: str = "phase-b-test-operator",
    trust_domain: str = "phase-b-test-domain",
) -> SystemContext:
    facade = read_facade or InMemoryReadFacade()
    registry = build_read_registry(facade)
    selected_provider = provider or DeterministicReadAgent()
    source = principles_path or Path(__file__).resolve().parents[1] / "doc" / "Luna_OPERATING_PRINCIPLES.md"
    return SystemContext(
        runtime_version="autodrive-dataops-agent-v2-phase-b",
        environment=environment,
        operator_id=operator_id,
        trust_domain=trust_domain,
        tool_catalog_hash=registry.catalog_hash(),
        policy_version="read-only-v2",
        event_store=event_store or EventStore(),
        checkpointer=checkpointer or InMemoryCheckpointer(),
        provider=selected_provider,
        read_facade=facade,
        tool_registry=registry,
        principles_path=str(source),
        budgets=budgets or RuntimeBudgets(),
    )


async def invoke(
    user_input: str,
    *,
    thread_id: str,
    system_context: SystemContext,
) -> AgentRunResult:
    """Initialize/load state, freeze principles, execute the graph, and return the result."""
    if not isinstance(user_input, str) or not user_input.strip():
        raise ValueError("user_input must not be empty")
    snapshot = load_operating_principles(system_context.principles_path)
    prior = system_context.checkpointer.load(thread_id)
    state = new_state(
        user_input=user_input,
        thread_id=thread_id,
        snapshot=snapshot,
        budgets=system_context.budgets,
        prior=prior,
    )
    provenance = EventProvenance(
        model_version=getattr(system_context.provider, "model_version", "unknown"),
        prompt_version=getattr(system_context.provider, "prompt_version", "unknown"),
        tool_catalog_hash=system_context.tool_catalog_hash,
        operating_principles_version=snapshot.version,
        operating_principles_hash=snapshot.content_hash,
        policy_version=system_context.policy_version,
    )
    started = system_context.event_store.append(
        event_type="AgentRunStarted",
        request_id=state["request_id"],
        thread_id=thread_id,
        payload={"user_input_length": len(user_input)},
        provenance=provenance,
    )
    state["last_event_id"] = started.event_id
    dependencies = GraphDependencies(
        provider=system_context.provider,
        read_runtime=ReadToolRuntime(system_context.tool_registry),
        compiler=CompletionContractCompiler(),
        evidence_tracker=EvidenceTracker(),
        completion_gate=ResponseCompletionGate(),
        context_builder=ContextBuilder(),
        event_store=system_context.event_store,
        model_version=getattr(system_context.provider, "model_version", "unknown"),
        prompt_version=getattr(system_context.provider, "prompt_version", "unknown"),
        tool_catalog_hash=system_context.tool_catalog_hash,
        policy_version=system_context.policy_version,
    )
    try:
        graph = build_graph(dependencies)
        final_state = await graph.ainvoke(
            state,
            config={"recursion_limit": system_context.budgets.max_agent_steps * 4 + 12},
        )
    except Exception as exc:
        from .outcomes import ControlledTerminalOutcome, TerminalCode

        terminal = ControlledTerminalOutcome(
            code=TerminalCode.UNRECOVERABLE_RUNTIME_ERROR,
            safe_facts={"graph_error_type": type(exc).__name__},
            message_template="The runtime could not safely complete this interaction.",
        )
        final_state = dict(state)
        final_state.update(
            {
                "terminal_state": terminal,
                "termination_reason": terminal.code.value,
            }
        )
        system_context.event_store.append(
            event_type="ControlledTerminalOutcomeProduced",
            request_id=state["request_id"],
            thread_id=thread_id,
            payload={"code": terminal.code.value, "safe_facts": terminal.safe_facts},
            provenance=provenance,
        )
    system_context.checkpointer.save(final_state)
    terminal = final_state.get("terminal_state")
    passed = bool(final_state.get("gate_passed"))
    status = "COMPLETED" if passed else "CONTROLLED_TERMINAL" if terminal else "ERROR"
    completed_event = system_context.event_store.append(
        event_type="AgentRunCompleted",
        request_id=state["request_id"],
        thread_id=thread_id,
        payload={"status": status, "termination_reason": final_state.get("termination_reason")},
        provenance=provenance,
    )
    final_state["last_event_id"] = completed_event.event_id
    candidate = final_state.get("final_candidate")
    return AgentRunResult(
        thread_id=thread_id,
        request_id=state["request_id"],
        status=status,
        response=candidate.response if passed and candidate is not None else None,
        goal_outcomes=tuple(final_state.get("goal_outcomes", {}).values()),
        terminal_outcome=terminal,
        state=final_state,
    )


async def resume(*, thread_id: str, resume_input: object, system_context: SystemContext) -> AgentRunResult:
    """Phase B has no interrupting WRITE path; approval resume is intentionally not active yet."""
    raise NotImplementedError(
        "Phase B has no HITL interrupt. resume() is reserved for the later approved-WRITE phase."
    )
