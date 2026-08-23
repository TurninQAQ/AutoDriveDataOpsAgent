import asyncio
import threading

import pytest

from deploy_ci_cloud_agentv2 import build_system_context, invoke, resume
from deploy_ci_cloud_agentv2.agent.decisions import FinalCandidate, SingleToolCall, ToolCall
from deploy_ci_cloud_agentv2.agent.goals import GoalDescriptor, ResumeTask
from deploy_ci_cloud_agentv2.platform.facade import InMemoryPlatformFacade
from deploy_ci_cloud_agentv2.providers import ScriptedProvider
from deploy_ci_cloud_agentv2.safety.approval import ApprovalDecision, ResumeInput


def test_resume_task_requires_explicit_approval_and_executes_once():
    descriptor = GoalDescriptor(1, (ResumeTask("g1", "task_A"),))
    provider = ScriptedProvider([
        SingleToolCall(ToolCall("w1", "resume_task", {"task_name": "task_A"}), descriptor),
        FinalCandidate("task_A was resumed", referenced_goal_ids=("g1",)),
    ])
    facade = InMemoryPlatformFacade(tasks={"task_A": {"state": "STOPPED", "revision": 1}})
    context = build_system_context(provider, read_facade=facade)

    first = asyncio.run(invoke("resume task_A", thread_id="write-resume", system_context=context))
    assert first.status == "INTERRUPTED"
    assert facade.mutation_count == 0
    pending = first.pending_interrupt
    assert pending.tool_name == "resume_task"
    assert pending.arguments["task_name"] == "task_A"

    second = asyncio.run(resume(
        thread_id="write-resume",
        resume_input=ResumeInput(
            ApprovalDecision.APPROVE,
            pending.approval_request_id,
            pending.transaction_id,
            pending.fingerprint,
        ),
        system_context=context,
    ))
    assert second.status == "COMPLETED"
    assert second.response == "task_A was resumed"
    assert facade.mutation_count == 1
    assert facade.tasks["task_A"]["state"] == "RUNNING"
    assert second.goal_outcomes[0].status.value == "SATISFIED"

import pytest

from deploy_ci_cloud_agentv2.agent.goals import (
    DeleteTask, SetTaskPriority, StopTask, SubmitTask,
)
from deploy_ci_cloud_agentv2.safety.policy import WriteAdmissionPolicy


def _approval_input(pending, decision=ApprovalDecision.APPROVE):
    return ResumeInput(
        decision,
        pending.approval_request_id,
        pending.transaction_id,
        pending.fingerprint,
    )


@pytest.mark.parametrize(
    "goal,tool,args,initial_tasks,expected",
    [
        (StopTask("g1", "task_A"), "stop_task", {"task_name": "task_A"}, {"task_A": {"state": "RUNNING", "revision": 1}}, lambda f: f.tasks["task_A"]["state"] == "STOPPED"),
        (DeleteTask("g1", "task_A"), "delete_task", {"task_name": "task_A"}, {"task_A": {"state": "RUNNING", "revision": 1}}, lambda f: "task_A" not in f.tasks),
        (SetTaskPriority("g1", "task_A", 7), "set_task_priority", {"task_name": "task_A", "priority": 7}, {"task_A": {"state": "RUNNING", "revision": 1}}, lambda f: f.tasks["task_A"]["priority"] == 7),
        (SubmitTask("g1", "task_B", {"dataset": "d1"}), "submit_task", {"task_name": "task_B", "config": {"dataset": "d1"}}, {}, lambda f: f.tasks["task_B"]["state"] == "SUBMITTED"),
    ],
)
def test_every_write_type_runs_through_approval_and_verification(goal, tool, args, initial_tasks, expected):
    descriptor = GoalDescriptor(1, (goal,))
    provider = ScriptedProvider([
        SingleToolCall(ToolCall("w", tool, args), descriptor),
        FinalCandidate("resolved", referenced_goal_ids=("g1",)),
    ])
    facade = InMemoryPlatformFacade(tasks=initial_tasks)
    context = build_system_context(provider, read_facade=facade)
    first = asyncio.run(invoke("write", thread_id=f"write-{tool}", system_context=context))
    assert first.status == "INTERRUPTED"
    assert facade.mutation_count == 0
    second = asyncio.run(resume(thread_id=f"write-{tool}", resume_input=_approval_input(first.pending_interrupt), system_context=context))
    assert second.status == "COMPLETED"
    assert facade.mutation_count == 1
    assert expected(facade)
    assert second.goal_outcomes[0].status.value == "SATISFIED"


def test_human_reject_executes_no_mutation_and_goal_is_rejected():
    descriptor = GoalDescriptor(1, (ResumeTask("g1", "task_A"),))
    provider = ScriptedProvider([
        SingleToolCall(ToolCall("w", "resume_task", {"task_name": "task_A"}), descriptor),
        FinalCandidate("not executed", referenced_goal_ids=("g1",)),
    ])
    facade = InMemoryPlatformFacade(tasks={"task_A": {"state": "STOPPED", "revision": 1}})
    context = build_system_context(provider, read_facade=facade)
    first = asyncio.run(invoke("resume", thread_id="reject", system_context=context))
    second = asyncio.run(resume(thread_id="reject", resume_input=_approval_input(first.pending_interrupt, ApprovalDecision.REJECT), system_context=context))
    assert second.status == "COMPLETED"
    assert facade.mutation_count == 0
    assert second.goal_outcomes[0].status.value == "REJECTED"
    assert second.goal_outcomes[0].reason_code == "USER_REJECTED_WRITE"


def test_reject_does_not_allow_same_request_to_reopen_same_approval():
    descriptor = GoalDescriptor(1, (ResumeTask("g1", "task_A"),))
    same_write = SingleToolCall(ToolCall("w", "resume_task", {"task_name": "task_A"}), descriptor)
    provider = ScriptedProvider([
        same_write,
        same_write,
        FinalCandidate("rejected", referenced_goal_ids=("g1",)),
    ])
    facade = InMemoryPlatformFacade(tasks={"task_A": {"state": "STOPPED", "revision": 1}})
    context = build_system_context(provider, read_facade=facade)
    first = asyncio.run(invoke("resume", thread_id="reject-repeat", system_context=context))
    second = asyncio.run(resume(thread_id="reject-repeat", resume_input=_approval_input(first.pending_interrupt, ApprovalDecision.REJECT), system_context=context))
    assert second.status == "COMPLETED"
    assert facade.mutation_count == 0
    approvals = [e for e in context.event_store.for_thread("reject-repeat") if e.event_type == "ApprovalRequested"]
    assert len(approvals) == 1


def test_policy_denial_is_goal_level_and_never_interrupts_or_mutates():
    descriptor = GoalDescriptor(1, (DeleteTask("g1", "task_A"),))
    provider = ScriptedProvider([
        SingleToolCall(ToolCall("w", "delete_task", {"task_name": "task_A"}), descriptor),
        FinalCandidate("denied", referenced_goal_ids=("g1",)),
    ])
    facade = InMemoryPlatformFacade(tasks={"task_A": {"state": "RUNNING", "revision": 1}})
    context = build_system_context(
        provider,
        read_facade=facade,
        write_policy=WriteAdmissionPolicy(protected_targets=frozenset({"task_A"})),
    )
    result = asyncio.run(invoke("delete", thread_id="deny", system_context=context))
    assert result.status == "COMPLETED"
    assert result.pending_interrupt is None
    assert facade.mutation_count == 0
    assert result.goal_outcomes[0].status.value == "DENIED"
    assert result.goal_outcomes[0].reason_code == "POLICY_DENIED_WRITE"


def test_forged_resume_input_is_rejected_before_graph_and_keeps_pending_transaction():
    descriptor = GoalDescriptor(1, (ResumeTask("g1", "task_A"),))
    provider = ScriptedProvider([
        SingleToolCall(ToolCall("w", "resume_task", {"task_name": "task_A"}), descriptor),
        FinalCandidate("done", referenced_goal_ids=("g1",)),
    ])
    facade = InMemoryPlatformFacade(tasks={"task_A": {"state": "STOPPED", "revision": 1}})
    context = build_system_context(provider, read_facade=facade)
    first = asyncio.run(invoke("resume", thread_id="forged", system_context=context))
    pending = first.pending_interrupt
    with pytest.raises(ValueError, match="fingerprint"):
        asyncio.run(resume(
            thread_id="forged",
            resume_input=ResumeInput(ApprovalDecision.APPROVE, pending.approval_request_id, pending.transaction_id, "forged"),
            system_context=context,
        ))
    assert facade.mutation_count == 0
    checkpoint = context.checkpointer.load("forged")
    assert checkpoint["current_request"].pending_interrupt == pending


def test_second_resume_cannot_reuse_consumed_approval():
    descriptor = GoalDescriptor(1, (ResumeTask("g1", "task_A"),))
    provider = ScriptedProvider([
        SingleToolCall(ToolCall("w", "resume_task", {"task_name": "task_A"}), descriptor),
        FinalCandidate("done", referenced_goal_ids=("g1",)),
    ])
    facade = InMemoryPlatformFacade(tasks={"task_A": {"state": "STOPPED", "revision": 1}})
    context = build_system_context(provider, read_facade=facade)
    first = asyncio.run(invoke("resume", thread_id="double-resume", system_context=context))
    approval = _approval_input(first.pending_interrupt)
    second = asyncio.run(resume(thread_id="double-resume", resume_input=approval, system_context=context))
    assert second.status == "COMPLETED"
    with pytest.raises(ValueError, match="not suspended"):
        asyncio.run(resume(thread_id="double-resume", resume_input=approval, system_context=context))
    assert facade.mutation_count == 1


def test_toctou_change_after_approval_invalidates_transaction_without_mutation():
    descriptor = GoalDescriptor(1, (ResumeTask("g1", "task_A"),))
    provider = ScriptedProvider([
        SingleToolCall(ToolCall("w", "resume_task", {"task_name": "task_A"}), descriptor),
        FinalCandidate("could not execute", referenced_goal_ids=("g1",)),
    ], repeat_last=True)
    facade = InMemoryPlatformFacade(tasks={"task_A": {"state": "STOPPED", "revision": 1}})
    context = build_system_context(provider, read_facade=facade)
    first = asyncio.run(invoke("resume", thread_id="toctou", system_context=context))
    facade.tasks["task_A"]["revision"] = 2
    facade.tasks["task_A"]["state"] = "RUNNING"
    second = asyncio.run(resume(thread_id="toctou", resume_input=_approval_input(first.pending_interrupt), system_context=context))
    assert facade.mutation_count == 0
    assert second.state["current_request"].write_transaction.status.value == "INVALIDATED"


def test_pending_approval_survives_system_context_restart_with_durable_checkpoint(tmp_path):
    db = tmp_path / "runtime.sqlite3"
    descriptor = GoalDescriptor(1, (ResumeTask("g1", "task_A"),))
    facade = InMemoryPlatformFacade(tasks={"task_A": {"state": "STOPPED", "revision": 1}})
    first_context = build_system_context(
        ScriptedProvider([SingleToolCall(ToolCall("w", "resume_task", {"task_name": "task_A"}), descriptor)]),
        read_facade=facade,
        durable_path=db,
    )
    first = asyncio.run(invoke("resume", thread_id="restart-write", system_context=first_context))
    assert first.status == "INTERRUPTED"
    pending = first.pending_interrupt
    assert facade.mutation_count == 0

    # New Runtime objects simulate a process restart; LangGraph's in-memory
    # checkpointer is intentionally not shared.
    second_context = build_system_context(
        ScriptedProvider([FinalCandidate("done", referenced_goal_ids=("g1",))]),
        read_facade=facade,
        durable_path=db,
    )
    second = asyncio.run(resume(
        thread_id="restart-write",
        resume_input=_approval_input(pending),
        system_context=second_context,
    ))
    assert second.status == "COMPLETED"
    assert facade.mutation_count == 1
    assert second.goal_outcomes[0].status.value == "SATISFIED"

from deploy_ci_cloud_agentv2.agent.outcomes import TerminalCode
from deploy_ci_cloud_agentv2.tools.write_runtime import MutationOutcomeUnknown


def test_unknown_mutation_outcome_blocks_replay_and_future_invoke():
    descriptor = GoalDescriptor(1, (ResumeTask("g1", "task_A"),))
    provider = ScriptedProvider([
        SingleToolCall(ToolCall("w", "resume_task", {"task_name": "task_A"}), descriptor),
    ], repeat_last=True)
    facade = InMemoryPlatformFacade(tasks={"task_A": {"state": "STOPPED", "revision": 1}})
    facade.set_mutation_failures({"resume_task": [MutationOutcomeUnknown("lost response")]})
    context = build_system_context(provider, read_facade=facade)
    first = asyncio.run(invoke("resume", thread_id="unknown", system_context=context))
    second = asyncio.run(resume(thread_id="unknown", resume_input=_approval_input(first.pending_interrupt), system_context=context))
    assert second.status == "CONTROLLED_TERMINAL"
    assert second.terminal_outcome.code is TerminalCode.REQUIRES_RECONCILIATION
    assert second.state["current_request"].write_transaction.status.value == "RECONCILIATION_REQUIRED"
    assert facade.mutation_count == 0

    calls_before = provider.calls
    blocked = asyncio.run(invoke("resume again", thread_id="unknown", system_context=context))
    assert blocked.status == "CONTROLLED_TERMINAL"
    assert blocked.terminal_outcome.code is TerminalCode.REQUIRES_RECONCILIATION
    assert provider.calls == calls_before
    assert facade.mutation_count == 0

from deploy_ci_cloud_agentv2 import reconcile


class LyingResumeFacade(InMemoryPlatformFacade):
    def resume_task(self, task_name: str):
        self.mutation_count += 1
        return {"ok": True, "task_name": task_name, "execution_id": "fake"}


def test_action_verifier_rejects_claimed_success_when_direct_effect_is_absent():
    descriptor = GoalDescriptor(1, (ResumeTask("g1", "task_A"),))
    provider = ScriptedProvider([
        SingleToolCall(ToolCall("w", "resume_task", {"task_name": "task_A"}), descriptor),
        FinalCandidate("claimed done", referenced_goal_ids=("g1",)),
    ], repeat_last=True)
    facade = LyingResumeFacade(tasks={"task_A": {"state": "STOPPED", "revision": 1}})
    context = build_system_context(provider, read_facade=facade)
    first = asyncio.run(invoke("resume", thread_id="lying-action", system_context=context))
    second = asyncio.run(resume(thread_id="lying-action", resume_input=_approval_input(first.pending_interrupt), system_context=context))
    assert facade.mutation_count == 1
    tx = second.state["current_request"].write_transaction
    assert tx.status.value == "VERIFICATION_FAILED"
    assert tx.action_verification.status.value == "FAILED"
    assert second.goal_outcomes[0].status.value == "INCONCLUSIVE"
    # Completion means the interaction is structurally resolved; it does not
    # convert an INCONCLUSIVE goal into success or semantically judge prose.
    assert second.status == "COMPLETED"


def test_reconciliation_of_unknown_no_effect_allows_only_new_transaction_and_new_approval():
    descriptor = GoalDescriptor(1, (ResumeTask("g1", "task_A"),))
    write = SingleToolCall(ToolCall("w", "resume_task", {"task_name": "task_A"}), descriptor)
    provider = ScriptedProvider([write], repeat_last=True)
    facade = InMemoryPlatformFacade(tasks={"task_A": {"state": "STOPPED", "revision": 1}})
    facade.set_mutation_failures({"resume_task": [MutationOutcomeUnknown("lost response")]})
    context = build_system_context(provider, read_facade=facade)
    first = asyncio.run(invoke("resume", thread_id="reconcile-retry", system_context=context))
    old_tx = first.pending_interrupt.transaction_id
    unknown = asyncio.run(resume(thread_id="reconcile-retry", resume_input=_approval_input(first.pending_interrupt), system_context=context))
    assert unknown.terminal_outcome.code is TerminalCode.REQUIRES_RECONCILIATION

    reconciled = asyncio.run(reconcile(thread_id="reconcile-retry", system_context=context))
    assert reconciled.effect_confirmed is False
    assert reconciled.replay_allowed is True
    assert reconciled.status == "NO_CURRENT_EFFECT_CONFIRMED_NEW_TRANSACTION_ALLOWED"

    again = asyncio.run(invoke("resume again", thread_id="reconcile-retry", system_context=context))
    assert again.status == "INTERRUPTED"
    assert again.pending_interrupt.transaction_id != old_tx
    assert facade.mutation_count == 0


def test_no_retry_write_remains_globally_blocked_after_unknown_reconciliation():
    descriptor = GoalDescriptor(1, (DeleteTask("g1", "task_A"),))
    provider = ScriptedProvider([
        SingleToolCall(ToolCall("d", "delete_task", {"task_name": "task_A"}), descriptor),
    ], repeat_last=True)
    facade = InMemoryPlatformFacade(tasks={"task_A": {"state": "RUNNING", "revision": 1}})
    facade.set_mutation_failures({"delete_task": [MutationOutcomeUnknown("lost response")]})
    context = build_system_context(provider, read_facade=facade)
    first = asyncio.run(invoke("delete", thread_id="delete-unknown", system_context=context))
    unknown = asyncio.run(resume(thread_id="delete-unknown", resume_input=_approval_input(first.pending_interrupt), system_context=context))
    assert unknown.terminal_outcome.code is TerminalCode.REQUIRES_RECONCILIATION
    result = asyncio.run(reconcile(thread_id="delete-unknown", system_context=context))
    assert result.replay_allowed is False
    assert result.status == "NO_CURRENT_EFFECT_CONFIRMED_NO_RETRY"

    descriptor2 = GoalDescriptor(1, (DeleteTask("g1", "task_A"),))
    provider2 = ScriptedProvider([
        SingleToolCall(ToolCall("d2", "delete_task", {"task_name": "task_A"}), descriptor2),
        FinalCandidate("blocked", referenced_goal_ids=("g1",)),
    ])
    context2 = build_system_context(
        provider2,
        read_facade=facade,
        event_store=context.event_store,
    )
    blocked = asyncio.run(invoke("delete again", thread_id="delete-other-thread", system_context=context2))
    assert blocked.status == "COMPLETED"
    assert blocked.goal_outcomes[0].status.value == "DENIED"
    assert blocked.goal_outcomes[0].reason_code == "POLICY_DENIED_WRITE"
    assert facade.mutation_count == 0


def test_two_restarted_workers_resuming_same_approval_execute_at_most_once(tmp_path):
    db = tmp_path / "concurrent.sqlite3"
    descriptor = GoalDescriptor(1, (ResumeTask("g1", "task_A"),))
    facade = InMemoryPlatformFacade(tasks={"task_A": {"state": "STOPPED", "revision": 1}})
    initial = build_system_context(
        ScriptedProvider([SingleToolCall(ToolCall("w", "resume_task", {"task_name": "task_A"}), descriptor)]),
        read_facade=facade,
        durable_path=db,
    )
    paused = asyncio.run(invoke("resume", thread_id="concurrent", system_context=initial))
    approval = _approval_input(paused.pending_interrupt)

    worker1 = build_system_context(
        ScriptedProvider([FinalCandidate("done", referenced_goal_ids=("g1",))], repeat_last=True),
        read_facade=facade,
        durable_path=db,
    )
    worker2 = build_system_context(
        ScriptedProvider([FinalCandidate("done", referenced_goal_ids=("g1",))], repeat_last=True),
        read_facade=facade,
        durable_path=db,
    )

    async def run_both():
        return await asyncio.gather(
            resume(thread_id="concurrent", resume_input=approval, system_context=worker1),
            resume(thread_id="concurrent", resume_input=approval, system_context=worker2),
            return_exceptions=True,
        )

    results = asyncio.run(run_both())
    assert facade.mutation_count == 1
    completed = [result for result in results if not isinstance(result, BaseException) and result.status == "COMPLETED"]
    assert len(completed) == 1
    assert all(
        not isinstance(result, BaseException) or isinstance(result, ValueError)
        for result in results
    )
    # Whatever branch wins the final durable tail, the checkpoint must remain
    # tail-consistent and may not permit a duplicate mutation.
    verifier = build_system_context(
        ScriptedProvider([FinalCandidate("noop", referenced_goal_ids=("g1",))], repeat_last=True),
        read_facade=facade,
        durable_path=db,
    )
    loaded = verifier.checkpointer.load("concurrent")
    assert loaded["last_event_id"] == verifier.event_store.for_thread("concurrent")[-1].event_id


def test_concurrent_resume_has_one_durable_approval_identity_and_one_claim(tmp_path):
    db = tmp_path / "approval-identity.sqlite3"
    descriptor = GoalDescriptor(1, (ResumeTask("g1", "task_A"),))
    facade = InMemoryPlatformFacade(tasks={"task_A": {"state": "STOPPED", "revision": 1}})
    initial = build_system_context(
        ScriptedProvider([SingleToolCall(ToolCall("w", "resume_task", {"task_name": "task_A"}), descriptor)]),
        read_facade=facade,
        durable_path=db,
    )
    paused = asyncio.run(invoke("resume", thread_id="approval-identity", system_context=initial))
    approval = _approval_input(paused.pending_interrupt)
    worker1 = build_system_context(
        ScriptedProvider([FinalCandidate("done", referenced_goal_ids=("g1",))], repeat_last=True),
        read_facade=facade,
        durable_path=db,
    )
    worker2 = build_system_context(
        ScriptedProvider([FinalCandidate("done", referenced_goal_ids=("g1",))], repeat_last=True),
        read_facade=facade,
        durable_path=db,
    )

    async def run_both():
        return await asyncio.gather(
            resume(thread_id="approval-identity", resume_input=approval, system_context=worker1),
            resume(thread_id="approval-identity", resume_input=approval, system_context=worker2),
            return_exceptions=True,
        )

    asyncio.run(run_both())
    events = initial.event_store.for_thread("approval-identity")
    grants = [e for e in events if e.event_type == "ApprovalGranted"]
    claims = [e for e in events if e.event_type == "ExecutionClaimed"]
    starts = [e for e in events if e.event_type == "MutationStarted"]
    assert len(grants) == 1
    assert len({e.payload["approval_id"] for e in grants}) == 1
    assert len(claims) == 1
    assert len(starts) == 1
    assert facade.mutation_count == 1


def test_live_inflight_worker_is_not_mistaken_for_crashed_mutation(tmp_path):
    db = tmp_path / "live-inflight.sqlite3"
    started = threading.Event()
    release = threading.Event()

    class BlockingFacade(InMemoryPlatformFacade):
        def resume_task(self, task_name: str):
            started.set()
            if not release.wait(timeout=10):
                raise RuntimeError("test mutation was not released")
            return super().resume_task(task_name)

    descriptor = GoalDescriptor(1, (ResumeTask("g1", "task_A"),))
    facade = BlockingFacade(tasks={"task_A": {"state": "STOPPED", "revision": 1}})
    initial = build_system_context(
        ScriptedProvider([SingleToolCall(ToolCall("w", "resume_task", {"task_name": "task_A"}), descriptor)]),
        read_facade=facade,
        durable_path=db,
    )
    paused = asyncio.run(invoke("resume", thread_id="live-inflight", system_context=initial))
    approval = _approval_input(paused.pending_interrupt)
    worker_a = build_system_context(
        ScriptedProvider([FinalCandidate("done", referenced_goal_ids=("g1",))], repeat_last=True),
        read_facade=facade,
        durable_path=db,
    )
    worker_b = build_system_context(
        ScriptedProvider([FinalCandidate("should-not-run", referenced_goal_ids=("g1",))], repeat_last=True),
        read_facade=facade,
        durable_path=db,
    )

    async def exercise():
        winner = asyncio.create_task(
            resume(thread_id="live-inflight", resume_input=approval, system_context=worker_a)
        )
        assert await asyncio.to_thread(started.wait, 5)
        with pytest.raises(ValueError, match="in flight"):
            await resume(
                thread_id="live-inflight", resume_input=approval, system_context=worker_b
            )
        release.set()
        return await winner

    result = asyncio.run(exercise())
    assert result.status == "COMPLETED"
    assert facade.mutation_count == 1


def test_agent_context_never_exposes_approval_or_execution_capabilities():
    from deploy_ci_cloud_agentv2.agent.context import ContextBuilder

    descriptor = GoalDescriptor(1, (ResumeTask("g1", "task_A"),))
    provider = ScriptedProvider([
        SingleToolCall(ToolCall("w", "resume_task", {"task_name": "task_A"}), descriptor),
        FinalCandidate("done", referenced_goal_ids=("g1",)),
    ])
    facade = InMemoryPlatformFacade(tasks={"task_A": {"state": "STOPPED", "revision": 1}})
    context = build_system_context(provider, read_facade=facade)
    paused = asyncio.run(invoke("resume", thread_id="context-capability", system_context=context))
    completed = asyncio.run(
        resume(
            thread_id="context-capability",
            resume_input=_approval_input(paused.pending_interrupt),
            system_context=context,
        )
    )
    current = completed.state["current_request"]
    tx = current.write_transaction
    model_context = ContextBuilder().build(current, completed.state["thread_history"])
    rendered = repr(model_context.model_facing_payload())
    secrets = (
        tx.approval_request_id,
        tx.approval.approval_id,
        tx.execution_claim.claim_id,
        tx.execution_attempt_id,
    )
    assert all(secret and secret not in rendered for secret in secrets)
    projection = model_context.runtime_structured.write_transaction
    assert "approval_request_id" not in projection
    assert "approval_id" not in projection
    assert "execution_claim_id" not in projection
    assert "execution_attempt_id" not in projection


def test_canonical_graph_has_no_second_semantic_authority_and_only_resolution_edges_return_to_agent():
    from deploy_ci_cloud_agentv2.agent.graph import build_graph
    from deploy_ci_cloud_agentv2.agent.runtime import _dependencies
    from deploy_ci_cloud_agentv2.agent.state import LatestStateHolder

    descriptor = GoalDescriptor(1, (ResumeTask("g1", "task_A"),))
    facade = InMemoryPlatformFacade(tasks={"task_A": {"state": "STOPPED", "revision": 1}})
    context = build_system_context(
        ScriptedProvider([SingleToolCall(ToolCall("w", "resume_task", {"task_name": "task_A"}), descriptor)]),
        read_facade=facade,
    )
    paused = asyncio.run(invoke("resume", thread_id="graph-shape", system_context=context))
    holder = LatestStateHolder()
    holder.record(paused.state)
    graph = build_graph(
        _dependencies(context, paused.state["current_request"], holder),
        checkpointer=context.graph_checkpointer,
    ).get_graph()
    nodes = set(graph.nodes)
    forbidden = {
        "planner", "planner_agent", "router", "intent_router", "semantic_router",
        "supervisor", "answer_judge", "strategy_engine", "decision_engine",
        "adaptive_controller",
    }
    assert not (nodes & forbidden)
    assert {
        "agent", "read_executor", "write_guard", "approval", "revalidate_write",
        "execution_claim", "execute_write", "action_verify", "operational_goal_verify",
        "response_completion_gate",
    }.issubset(nodes)
    allowed_back_edges = {
        "agent",
        "read_executor", "write_guard", "approval", "revalidate_write",
        "execute_write", "action_verify", "operational_goal_verify",
        "response_completion_gate",
    }
    for edge in graph.edges:
        if edge.target == "agent":
            # The canonical graph necessarily has START -> agent.  Only
            # edges after the initial entry are subject to the resolution
            # back-edge restriction.
            if edge.source != "__start__":
                assert edge.source in allowed_back_edges
