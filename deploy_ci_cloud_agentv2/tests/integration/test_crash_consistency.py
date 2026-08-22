import asyncio

from deploy_ci_cloud_agentv2 import build_system_context, invoke
from deploy_ci_cloud_agentv2.agent.budgets import RuntimeBudgets
from deploy_ci_cloud_agentv2.agent.decisions import FinalCandidate, SingleToolCall, ToolCall
from deploy_ci_cloud_agentv2.agent.events import EventStore
from deploy_ci_cloud_agentv2.agent.goals import GoalDescriptor, ReadTaskState
from deploy_ci_cloud_agentv2.agent.outcomes import TerminalCode
from deploy_ci_cloud_agentv2.platform import InMemoryReadFacade
from deploy_ci_cloud_agentv2.providers import ScriptedProvider


class FailOnEventStore(EventStore):
    def __init__(self, event_type, *, always=False):
        super().__init__()
        self.event_type = event_type
        self.always = always
        self.failed = False

    def append(self, **kwargs):
        if kwargs["event_type"] == self.event_type and (self.always or not self.failed):
            self.failed = True
            raise RuntimeError(f"injected append failure: {self.event_type}")
        return super().append(**kwargs)


def _provider(*decisions):
    descriptor = GoalDescriptor(1, (ReadTaskState("g1", "task_A"),))
    normalized = []
    for decision in decisions:
        if isinstance(decision, str) and decision == "read":
            normalized.append(SingleToolCall(ToolCall("task", "get_task_detail", {"task_name": "task_A"}), descriptor))
        elif isinstance(decision, str) and decision == "final":
            normalized.append(FinalCandidate("done", referenced_goal_ids=("g1",)))
        else:
            normalized.append(decision)
    return normalized


def _run(thread_id, store, decisions, *, budgets=None):
    context = build_system_context(
        ScriptedProvider(decisions, repeat_last=True),
        read_facade=InMemoryReadFacade(
            responses={"get_task_detail": {"task_name": "task_A", "state": "RUNNING"}}
        ),
        event_store=store,
        budgets=budgets or RuntimeBudgets(),
    )
    result = asyncio.run(invoke("task_A", thread_id=thread_id, system_context=context))
    return result, context


def test_evidence_append_failure_never_checkpoints_undurable_evidence():
    result, context = _run(
        "crash-evidence",
        FailOnEventStore("EvidenceRecorded"),
        _provider("read"),
    )
    checkpoint = context.checkpointer.load("crash-evidence")
    events = context.event_store.for_thread("crash-evidence")
    assert result.status == "CONTROLLED_TERMINAL"
    assert checkpoint["last_event_id"] == events[-1].event_id
    assert checkpoint["current_request"].evidence.records == ()
    assert not any(event.event_type == "EvidenceRecorded" for event in events)


def test_goal_outcome_append_failure_keeps_durable_evidence_only():
    result, context = _run(
        "crash-outcome",
        FailOnEventStore("GoalOutcomeUpdated"),
        _provider("read"),
    )
    checkpoint = context.checkpointer.load("crash-outcome")
    events = context.event_store.for_thread("crash-outcome")
    assert result.status == "CONTROLLED_TERMINAL"
    assert checkpoint["last_event_id"] == events[-1].event_id
    assert checkpoint["current_request"].evidence.records
    assert not any(
        event.event_type == "GoalOutcomeUpdated" and event.payload.get("status") == "SATISFIED"
        for event in events
    )


def test_completion_gate_append_failure_cannot_checkpoint_gate_success():
    result, context = _run(
        "crash-gate",
        FailOnEventStore("CompletionGateEvaluated"),
        _provider("read", "final"),
    )
    checkpoint = context.checkpointer.load("crash-gate")
    events = context.event_store.for_thread("crash-gate")
    assert result.status == "CONTROLLED_TERMINAL"
    assert checkpoint["last_event_id"] == events[-1].event_id
    assert checkpoint["current_request"].gate_passed is None
    assert not any(event.event_type == "CompletionGateEvaluated" for event in events)


def test_completed_append_failure_returns_last_durable_state_without_checkpoint():
    result, context = _run(
        "crash-completed",
        FailOnEventStore("AgentRunCompleted", always=True),
        _provider("read", "final"),
    )
    events = context.event_store.for_thread("crash-completed")
    assert result.status == "CONTROLLED_TERMINAL"
    assert result.terminal_outcome.code is TerminalCode.UNRECOVERABLE_RUNTIME_ERROR
    assert context.checkpointer.load("crash-completed") is None
    assert events[-1].event_type != "AgentRunCompleted"
    assert result.state["last_event_id"] == events[-1].event_id


def test_terminal_append_failure_is_runtime_fail_closed_without_synthetic_completion():
    descriptor = GoalDescriptor(1, (ReadTaskState("g1", "task_A"),))
    candidate = FinalCandidate(
        "premature",
        proposed_goal_descriptor=descriptor,
        referenced_goal_ids=("g1",),
    )
    result, context = _run(
        "crash-terminal",
        FailOnEventStore("ControlledTerminalOutcomeProduced", always=True),
        [candidate],
        budgets=RuntimeBudgets(max_completion_gate_rejections=1, max_agent_steps=10),
    )
    events = context.event_store.for_thread("crash-terminal")
    assert result.status == "CONTROLLED_TERMINAL"
    assert context.checkpointer.load("crash-terminal") is None
    assert events[-1].event_type != "ControlledTerminalOutcomeProduced"
    assert result.state["last_event_id"] == events[-1].event_id
