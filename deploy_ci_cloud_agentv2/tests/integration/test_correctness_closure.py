import asyncio

from deploy_ci_cloud_agentv2 import build_system_context, invoke
from deploy_ci_cloud_agentv2.agent.budgets import RuntimeBudgets
from deploy_ci_cloud_agentv2.agent.decisions import (
    FinalCandidate,
    ReadToolBatch,
    SingleToolCall,
    ToolCall,
)
from deploy_ci_cloud_agentv2.agent.goals import DiagnoseTask, GoalDescriptor, ReadTaskState
from deploy_ci_cloud_agentv2.agent.outcomes import TerminalCode
from deploy_ci_cloud_agentv2.platform import InMemoryReadFacade
from deploy_ci_cloud_agentv2.providers import ScriptedProvider
from deploy_ci_cloud_agentv2.tools.runtime import ReadFailure


def _run(provider, facade, *, thread_id, budgets=None):
    context = build_system_context(
        provider,
        read_facade=facade,
        budgets=budgets or RuntimeBudgets(),
    )
    result = asyncio.run(invoke("test request", thread_id=thread_id, system_context=context))
    return result, context


def test_entity_mismatch_cannot_satisfy_task_or_diagnosis_goal():
    task_descriptor = GoalDescriptor(1, (ReadTaskState("g1", "task_A"),))
    task_provider = ScriptedProvider(
        [
            SingleToolCall(
                ToolCall("task", "get_task_detail", {"task_name": "task_A"}),
                task_descriptor,
            ),
            FinalCandidate("wrong target", referenced_goal_ids=("g1",)),
        ]
    )
    result, context = _run(
        task_provider,
        InMemoryReadFacade(
            responses={"get_task_detail": {"task_name": "other_task", "state": "RUNNING"}}
        ),
        thread_id="mismatch-task",
    )
    assert result.status != "COMPLETED"
    assert result.state["evidence"].records == ()
    assert result.state["goal_outcomes"]["g1"].status.value == "PENDING"
    assert not any(
        event.payload.get("passed")
        for event in context.event_store.for_thread("mismatch-task")
        if event.event_type == "CompletionGateEvaluated"
    )

    diagnosis_descriptor = GoalDescriptor(1, (DiagnoseTask("g1", "task_A"),))
    diagnosis_provider = ScriptedProvider(
        [
            SingleToolCall(
                ToolCall("detail", "get_task_detail", {"task_name": "task_A"}),
                diagnosis_descriptor,
            ),
            SingleToolCall(
                ToolCall("diagnosis", "diagnose_task", {"task_name": "task_A"})
            ),
            FinalCandidate("wrong diagnosis target", referenced_goal_ids=("g1",)),
        ]
    )
    result, _ = _run(
        diagnosis_provider,
        InMemoryReadFacade(
            responses={
                "get_task_detail": {"task_name": "task_A", "state": "FAILED"},
                "diagnose_task": {"task_name": "task_B", "root_cause": "CUDA OOM"},
            }
        ),
        thread_id="mismatch-diagnosis",
    )
    assert result.status != "COMPLETED"
    assert not any(
        record.kind == "DIAGNOSTIC_CONTEXT"
        for record in result.state["evidence"].records
    )


def test_invalid_single_read_decision_is_bounded_and_agent_recovers():
    descriptor = GoalDescriptor(1, (ReadTaskState("g1", "task_A"),))
    provider = ScriptedProvider(
        [
            SingleToolCall(
                ToolCall("bad", "unknown_tool", {}),
                descriptor,
            ),
            SingleToolCall(ToolCall("good", "get_task_detail", {"task_name": "task_A"})),
            FinalCandidate("recovered", referenced_goal_ids=("g1",)),
        ]
    )
    result, context = _run(
        provider,
        InMemoryReadFacade(
            responses={"get_task_detail": {"task_name": "task_A", "state": "RUNNING"}}
        ),
        thread_id="invalid-single",
    )
    assert result.status == "COMPLETED"
    assert result.terminal_outcome is None
    assert any(
        item.status == "READ_GUARD_REJECTED" for item in result.state["observations"]
    )
    assert not any(
        event.payload.get("code") == TerminalCode.UNRECOVERABLE_RUNTIME_ERROR.value
        for event in context.event_store.for_thread("invalid-single")
        if event.event_type == "ControlledTerminalOutcomeProduced"
    )


def test_invalid_batch_read_decision_is_bounded_and_agent_recovers():
    descriptor = GoalDescriptor(1, (ReadTaskState("g1", "task_A"),))
    provider = ScriptedProvider(
        [
            ReadToolBatch(
                (ToolCall("bad", "diagnose_task", {"task_name": "task_A"}),),
                descriptor,
            ),
            SingleToolCall(ToolCall("good", "get_task_detail", {"task_name": "task_A"})),
            FinalCandidate("recovered", referenced_goal_ids=("g1",)),
        ]
    )
    result, _ = _run(
        provider,
        InMemoryReadFacade(
            responses={"get_task_detail": {"task_name": "task_A", "state": "RUNNING"}}
        ),
        thread_id="invalid-batch",
    )
    assert result.status == "COMPLETED"
    assert any(item.status == "READ_GUARD_REJECTED" for item in result.state["observations"])


def test_partial_batch_success_is_preserved_and_retry_budget_is_per_call():
    facade = InMemoryReadFacade(
        responses={
            "get_task_detail": {"task_name": "task_A", "state": "RUNNING"},
            "get_queue_state": {"task_name": "task_A", "position": 1},
            "get_gpu_pool": {"devices": [{"gpu_id": "0"}]},
        },
        failures={
            "get_gpu_pool": [
                ReadFailure("READ_TIMEOUT", "timeout", retryable=True),
                ReadFailure("READ_TIMEOUT", "timeout", retryable=True),
            ]
        },
    )
    descriptor = GoalDescriptor(1, (ReadTaskState("g1", "task_A"),))
    provider = ScriptedProvider(
        [
            ReadToolBatch(
                (
                    ToolCall("task", "get_task_detail", {"task_name": "task_A"}),
                    ToolCall("gpu", "get_gpu_pool", {}),
                ),
                descriptor,
            ),
            FinalCandidate("task is available", referenced_goal_ids=("g1",)),
        ]
    )
    result, _ = _run(
        provider,
        facade,
        thread_id="partial-preserve",
        budgets=RuntimeBudgets(max_runtime_read_retries_per_call=1),
    )
    assert result.status == "COMPLETED"
    assert [item.status for item in result.state["observations"]] == [
        "SUCCESS",
        "READ_FAILURE",
    ]
    assert any(record.kind == "LIVE_TASK" for record in result.state["evidence"].records)
    assert result.state["budgets"].runtime_read_retries_used == 1


def test_completed_checkpoint_points_to_event_store_tail():
    descriptor = GoalDescriptor(1, (ReadTaskState("g1", "task_A"),))
    provider = ScriptedProvider(
        [
            SingleToolCall(
                ToolCall("task", "get_task_detail", {"task_name": "task_A"}),
                descriptor,
            ),
            FinalCandidate("done", referenced_goal_ids=("g1",)),
        ]
    )
    result, context = _run(
        provider,
        InMemoryReadFacade(
            responses={"get_task_detail": {"task_name": "task_A", "state": "RUNNING"}}
        ),
        thread_id="checkpoint-order",
    )
    trace = context.event_store.for_thread("checkpoint-order")
    persisted = context.checkpointer.load("checkpoint-order")
    assert result.status == "COMPLETED"
    assert trace[-1].event_type == "AgentRunCompleted"
    assert persisted["last_event_id"] == trace[-1].event_id
    assert result.state["last_event_id"] == trace[-1].event_id
