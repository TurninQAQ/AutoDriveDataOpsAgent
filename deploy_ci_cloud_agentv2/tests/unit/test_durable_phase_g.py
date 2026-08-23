import asyncio
import sqlite3

import pytest

from deploy_ci_cloud_agentv2 import build_system_context, invoke
from deploy_ci_cloud_agentv2.agent.decisions import FinalCandidate, SingleToolCall, ToolCall
from deploy_ci_cloud_agentv2.agent.goals import GoalDescriptor, ResumeTask
from deploy_ci_cloud_agentv2.agent.outcomes import TerminalCode
from deploy_ci_cloud_agentv2.memory import CheckpointIntegrityError
from deploy_ci_cloud_agentv2.memory.sqlite import SQLiteExecutionClaimStore
from deploy_ci_cloud_agentv2.platform.facade import InMemoryPlatformFacade
from deploy_ci_cloud_agentv2.providers import ScriptedProvider
from deploy_ci_cloud_agentv2.safety.approval import ApprovalDecision, ApprovalValidator, ResumeInput


def _paused(tmp_path, thread="durable"):
    db = tmp_path / "runtime.sqlite3"
    descriptor = GoalDescriptor(1, (ResumeTask("g1", "task_A"),))
    facade = InMemoryPlatformFacade(tasks={"task_A": {"state": "STOPPED", "revision": 1}})
    context = build_system_context(
        ScriptedProvider([SingleToolCall(ToolCall("w", "resume_task", {"task_name": "task_A"}), descriptor)]),
        read_facade=facade,
        durable_path=db,
    )
    result = asyncio.run(invoke("resume", thread_id=thread, system_context=context))
    return db, facade, context, result


def test_sqlite_checkpoint_digest_tamper_fails_closed(tmp_path):
    db, _facade, context, paused = _paused(tmp_path, "tamper")
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE checkpoints SET state_json = state_json || ' ' WHERE thread_id='tamper'")
        conn.commit()
    with pytest.raises(CheckpointIntegrityError, match="digest"):
        context.checkpointer.load("tamper")


def test_invoke_detects_durable_event_tail_ahead_of_checkpoint(tmp_path):
    db, facade, context, paused = _paused(tmp_path, "tail")
    current = paused.state["current_request"]
    # Simulate a durable event that landed after the checkpoint but whose state
    # projection did not. Recovery must not silently start a new Agent turn.
    last = context.event_store.for_thread("tail")[-1]
    context.event_store.append(
        event_type="InjectedDurableTail",
        request_id=current.identity.request_id,
        thread_id="tail",
        payload={"test": True},
        provenance=last.provenance,
        causation_id=last.event_id,
    )
    calls = context.provider.calls
    result = asyncio.run(invoke("new request", thread_id="tail", system_context=context))
    assert result.status == "CONTROLLED_TERMINAL"
    assert result.terminal_outcome.code is TerminalCode.CHECKPOINT_CORRUPTION
    assert context.provider.calls == calls
    assert facade.mutation_count == 0


def test_sqlite_execution_claim_is_single_use_across_store_instances(tmp_path):
    db, _facade, context, paused = _paused(tmp_path, "claim-cas")
    current = paused.state["current_request"]
    tx = current.write_transaction
    pending = paused.pending_interrupt
    resume = ResumeInput(
        ApprovalDecision.APPROVE,
        pending.approval_request_id,
        pending.transaction_id,
        pending.fingerprint,
    )
    approval = ApprovalValidator.validate_resume(
        tx, pending, resume, operator_id=context.operator_id, trust_domain=context.trust_domain
    )
    first = SQLiteExecutionClaimStore(db)
    second = SQLiteExecutionClaimStore(db)
    claim = first.claim(tx, approval)
    with pytest.raises(RuntimeError, match="already exists"):
        second.claim(tx, approval)
    first_attempt = first.consume_attempt(claim)
    assert first_attempt.startswith("attempt_")
    with pytest.raises(RuntimeError, match="already consumed"):
        second.consume_attempt(claim)

from deploy_ci_cloud_agentv2 import resume
from deploy_ci_cloud_agentv2.memory.sqlite import (
    SQLiteApprovalRecordStore,
    SQLiteEventStore,
)


class CrashAfterApprovalStore(SQLiteApprovalRecordStore):
    def record_with_event(self, *args, **kwargs):
        result = super().record_with_event(*args, **kwargs)
        raise SystemExit("simulated crash after durable approval")


class CrashAfterClaimStore(SQLiteExecutionClaimStore):
    def claim_with_event(self, *args, **kwargs):
        result = super().claim_with_event(*args, **kwargs)
        raise SystemExit("simulated crash after durable execution claim")


class CrashInsideMutationFacade(InMemoryPlatformFacade):
    def resume_task(self, task_name: str):
        self.mutation_count += 1
        raise SystemExit("simulated process death after MutationStarted")


def _approve(pending):
    return ResumeInput(
        ApprovalDecision.APPROVE,
        pending.approval_request_id,
        pending.transaction_id,
        pending.fingerprint,
    )


def test_restart_recovers_durable_approval_event_and_continues_same_transaction(tmp_path):
    db = tmp_path / "approval-crash.sqlite3"
    descriptor = GoalDescriptor(1, (ResumeTask("g1", "task_A"),))
    facade = InMemoryPlatformFacade(tasks={"task_A": {"state": "STOPPED", "revision": 1}})
    first_context = build_system_context(
        ScriptedProvider([SingleToolCall(ToolCall("w", "resume_task", {"task_name": "task_A"}), descriptor)]),
        read_facade=facade,
        durable_path=db,
        approval_store=CrashAfterApprovalStore(db),
    )
    paused = asyncio.run(invoke("resume", thread_id="approval-crash", system_context=first_context))
    with pytest.raises(SystemExit, match="durable approval"):
        asyncio.run(
            resume(
                thread_id="approval-crash",
                resume_input=_approve(paused.pending_interrupt),
                system_context=first_context,
            )
        )

    restarted = build_system_context(
        ScriptedProvider([FinalCandidate("done", referenced_goal_ids=("g1",))], repeat_last=True),
        read_facade=facade,
        durable_path=db,
    )
    result = asyncio.run(
        resume(
            thread_id="approval-crash",
            resume_input=_approve(paused.pending_interrupt),
            system_context=restarted,
        )
    )
    assert result.status == "COMPLETED"
    assert facade.mutation_count == 1
    approvals = [e for e in restarted.event_store.for_thread("approval-crash") if e.event_type == "ApprovalGranted"]
    assert len(approvals) == 1


def test_restart_recovers_durable_claim_and_executes_only_unstarted_attempt(tmp_path):
    db = tmp_path / "claim-crash.sqlite3"
    descriptor = GoalDescriptor(1, (ResumeTask("g1", "task_A"),))
    facade = InMemoryPlatformFacade(tasks={"task_A": {"state": "STOPPED", "revision": 1}})
    first_context = build_system_context(
        ScriptedProvider([SingleToolCall(ToolCall("w", "resume_task", {"task_name": "task_A"}), descriptor)]),
        read_facade=facade,
        durable_path=db,
        claim_store=CrashAfterClaimStore(db),
    )
    paused = asyncio.run(invoke("resume", thread_id="claim-crash", system_context=first_context))
    with pytest.raises(SystemExit, match="execution claim"):
        asyncio.run(
            resume(
                thread_id="claim-crash",
                resume_input=_approve(paused.pending_interrupt),
                system_context=first_context,
            )
        )
    assert facade.mutation_count == 0

    restarted = build_system_context(
        ScriptedProvider([FinalCandidate("done", referenced_goal_ids=("g1",))], repeat_last=True),
        read_facade=facade,
        durable_path=db,
    )
    result = asyncio.run(
        resume(
            thread_id="claim-crash",
            resume_input=_approve(paused.pending_interrupt),
            system_context=restarted,
        )
    )
    assert result.status == "COMPLETED"
    assert facade.mutation_count == 1
    claims = [e for e in restarted.event_store.for_thread("claim-crash") if e.event_type == "ExecutionClaimed"]
    starts = [e for e in restarted.event_store.for_thread("claim-crash") if e.event_type == "MutationStarted"]
    assert len(claims) == 1
    assert len(starts) == 1


def test_restart_after_mutation_started_never_replays_and_requires_reconciliation(tmp_path):
    db = tmp_path / "mutation-crash.sqlite3"
    descriptor = GoalDescriptor(1, (ResumeTask("g1", "task_A"),))
    facade = CrashInsideMutationFacade(tasks={"task_A": {"state": "STOPPED", "revision": 1}})
    first_context = build_system_context(
        ScriptedProvider([SingleToolCall(ToolCall("w", "resume_task", {"task_name": "task_A"}), descriptor)]),
        read_facade=facade,
        durable_path=db,
    )
    paused = asyncio.run(invoke("resume", thread_id="mutation-crash", system_context=first_context))
    with pytest.raises(SystemExit, match="MutationStarted"):
        asyncio.run(
            resume(
                thread_id="mutation-crash",
                resume_input=_approve(paused.pending_interrupt),
                system_context=first_context,
            )
        )
    assert facade.mutation_count == 1

    restarted = build_system_context(
        ScriptedProvider([FinalCandidate("must-not-run", referenced_goal_ids=("g1",))], repeat_last=True),
        read_facade=facade,
        durable_path=db,
    )
    result = asyncio.run(invoke("new request", thread_id="mutation-crash", system_context=restarted))
    assert result.status == "CONTROLLED_TERMINAL"
    assert result.terminal_outcome.code is TerminalCode.REQUIRES_RECONCILIATION
    assert result.state["current_request"].write_transaction.status.value == "RECONCILIATION_REQUIRED"
    assert facade.mutation_count == 1
    event_types = [e.event_type for e in restarted.event_store.for_thread("mutation-crash")]
    assert event_types.count("MutationStarted") == 1
    assert "MutationResultRecorded" not in event_types
    assert "ReconciliationRequired" in event_types
    assert "WriteReplayBlocked" in event_types


def test_atomic_claim_rolls_back_if_audit_append_fails(tmp_path):
    db, _facade, context, paused = _paused(tmp_path, "claim-rollback")
    tx = paused.state["current_request"].write_transaction
    pending = paused.pending_interrupt
    approval = ApprovalValidator.validate_resume(
        tx, pending, _approve(pending), operator_id=context.operator_id, trust_domain=context.trust_domain
    )
    approval = context.approval_store.record(approval)
    store = SQLiteExecutionClaimStore(db)

    original = context.event_store._append_in_connection
    def fail(*args, **kwargs):
        if kwargs.get("event_type") == "ExecutionClaimed":
            raise RuntimeError("audit failure")
        return original(*args, **kwargs)
    context.event_store._append_in_connection = fail
    with pytest.raises(RuntimeError, match="audit failure"):
        store.claim_with_event(
            context.event_store,
            tx,
            approval,
            request_id=tx.bound_goal_ids[0],
            thread_id="claim-rollback",
            provenance=context.event_store.for_thread("claim-rollback")[-1].provenance,
            causation_id=context.event_store.for_thread("claim-rollback")[-1].event_id,
        )
    assert store.claim_for(tx.transaction_id) is None


def test_atomic_approval_rolls_back_if_audit_append_fails(tmp_path):
    db, _facade, context, paused = _paused(tmp_path, "approval-rollback")
    tx = paused.state["current_request"].write_transaction
    pending = paused.pending_interrupt
    candidate = ApprovalValidator.validate_resume(
        tx, pending, _approve(pending), operator_id=context.operator_id, trust_domain=context.trust_domain
    )
    store = SQLiteApprovalRecordStore(db)
    original = context.event_store._append_in_connection

    def fail(*args, **kwargs):
        if kwargs.get("event_type") == "ApprovalGranted":
            raise RuntimeError("approval audit failure")
        return original(*args, **kwargs)

    context.event_store._append_in_connection = fail
    with pytest.raises(RuntimeError, match="approval audit failure"):
        store.record_with_event(
            context.event_store,
            candidate,
            request_id=paused.state["current_request"].identity.request_id,
            thread_id="approval-rollback",
            provenance=context.event_store.for_thread("approval-rollback")[-1].provenance,
            causation_id=context.event_store.for_thread("approval-rollback")[-1].event_id,
        )
    assert store.get(candidate.approval_request_id) is None


def test_atomic_mutation_start_rolls_back_attempt_if_audit_append_fails(tmp_path):
    db, _facade, context, paused = _paused(tmp_path, "attempt-rollback")
    tx = paused.state["current_request"].write_transaction
    pending = paused.pending_interrupt
    candidate = ApprovalValidator.validate_resume(
        tx, pending, _approve(pending), operator_id=context.operator_id, trust_domain=context.trust_domain
    )
    approval_store = SQLiteApprovalRecordStore(db)
    approval, approval_event = approval_store.record_with_event(
        context.event_store,
        candidate,
        request_id=paused.state["current_request"].identity.request_id,
        thread_id="attempt-rollback",
        provenance=context.event_store.for_thread("attempt-rollback")[-1].provenance,
        causation_id=context.event_store.for_thread("attempt-rollback")[-1].event_id,
    )
    claim_store = SQLiteExecutionClaimStore(db)
    claim, claim_event = claim_store.claim_with_event(
        context.event_store,
        tx,
        approval,
        request_id=paused.state["current_request"].identity.request_id,
        thread_id="attempt-rollback",
        provenance=approval_event.provenance,
        causation_id=approval_event.event_id,
    )
    original = context.event_store._append_in_connection

    def fail(*args, **kwargs):
        if kwargs.get("event_type") == "MutationStarted":
            raise RuntimeError("mutation-start audit failure")
        return original(*args, **kwargs)

    context.event_store._append_in_connection = fail
    with pytest.raises(RuntimeError, match="mutation-start audit failure"):
        claim_store.consume_attempt_with_event(
            context.event_store,
            claim,
            request_id=paused.state["current_request"].identity.request_id,
            thread_id="attempt-rollback",
            tool_name=tx.proposal.tool_name,
            fingerprint=tx.fingerprint,
            provenance=claim_event.provenance,
            causation_id=claim_event.event_id,
        )
    assert claim_store.attempt_for(tx.transaction_id) is None


class CrashAfterMutationResultStore(SQLiteEventStore):
    def append_with_checkpoint(self, checkpointer, state, **kwargs):
        event = super().append_with_checkpoint(checkpointer, state, **kwargs)
        if kwargs.get("event_type") == "MutationResultRecorded":
            raise SystemExit("simulated crash after durable mutation result")
        return event


def test_restart_after_mutation_result_continues_verification_without_reexecution(tmp_path):
    db = tmp_path / "mutation-result-crash.sqlite3"
    descriptor = GoalDescriptor(1, (ResumeTask("g1", "task_A"),))
    facade = InMemoryPlatformFacade(tasks={"task_A": {"state": "STOPPED", "revision": 1}})
    first_context = build_system_context(
        ScriptedProvider([SingleToolCall(ToolCall("w", "resume_task", {"task_name": "task_A"}), descriptor)]),
        read_facade=facade,
        durable_path=db,
        event_store=CrashAfterMutationResultStore(db),
    )
    paused = asyncio.run(invoke("resume", thread_id="mutation-result-crash", system_context=first_context))
    with pytest.raises(SystemExit, match="durable mutation result"):
        asyncio.run(
            resume(
                thread_id="mutation-result-crash",
                resume_input=_approve(paused.pending_interrupt),
                system_context=first_context,
            )
        )
    assert facade.mutation_count == 1

    restarted = build_system_context(
        ScriptedProvider([FinalCandidate("done", referenced_goal_ids=("g1",))], repeat_last=True),
        read_facade=facade,
        durable_path=db,
    )
    result = asyncio.run(invoke("continue prior durable run", thread_id="mutation-result-crash", system_context=restarted))
    assert result.status == "COMPLETED"
    assert facade.mutation_count == 1
    event_types = [e.event_type for e in restarted.event_store.for_thread("mutation-result-crash")]
    assert event_types.count("MutationStarted") == 1
    assert event_types.count("MutationResultRecorded") == 1
    assert "ActionVerificationRecorded" in event_types
