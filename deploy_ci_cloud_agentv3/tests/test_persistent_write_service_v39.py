from __future__ import annotations

import pytest

from deploy_ci_cloud_agentv3.mcp.client import InProcessMCPClient
from deploy_ci_cloud_agentv3.mcp.factory import build_tooling
from deploy_ci_cloud_agentv3.mcp.profiles import RUNTIME_TOOLS
from deploy_ci_cloud_agentv3.persistence.write_execution_store import SQLiteWriteExecutionStore
from deploy_ci_cloud_agentv3.services.write_service import WriteService
from deploy_ci_cloud_agentv3.tests.fakes import FakeFacade
from deploy_ci_cloud_agentv3.tests.test_write_service import pending


@pytest.mark.asyncio
async def test_verified_write_is_not_reexecuted_after_restart(tmp_path):
    facade=FakeFacade(); action=pending(facade); tooling=build_tooling(facade); client=InProcessMCPClient(tooling.registry,RUNTIME_TOOLS)
    path=tmp_path/"state.sqlite"
    first=await WriteService(client,execution_store=SQLiteWriteExecutionStore(path)).execute(action,action.fingerprint)
    assert first.status=="VERIFIED" and len(facade.mutations)==1
    second=await WriteService(client,execution_store=SQLiteWriteExecutionStore(path)).execute(action,action.fingerprint)
    assert second.status=="VERIFIED" and len(facade.mutations)==1


class DropAfterMutationClient:
    def __init__(self):
        self.calls=0; self.priority=1; self.pre={"task_name":"task_a","queue_sha256":"q","task_config_sha256":"c","task_exists":True,"active_task_name":None}
    async def call_tool(self,name,args):
        if name=="capture_write_precondition": return dict(self.pre)
        if name=="set_task_priority":
            self.calls+=1; self.priority=5; raise ConnectionResetError("after dispatch")
        if name=="get_action_verification_snapshot": return {"task_name":"task_a","task_exists":True,"priority":self.priority,"errors":{},"airflow_runs":[],"containers":[],"gpu_reservations":[],"queue":{"location":"queued"}}
        raise AssertionError(name)


@pytest.mark.asyncio
async def test_unknown_outcome_reconciles_after_restart_without_retry(tmp_path):
    from deploy_ci_cloud_agentv3.models.pending_action import PendingAction,compute_pending_action_fingerprint
    c=DropAfterMutationClient(); args={"task_name":"task_a","priority":5}; before={"task_name":"task_a","task_exists":True,"priority":1,"errors":{},"airflow_runs":[],"containers":[],"gpu_reservations":[],"queue":{"location":"queued"}}
    fp=compute_pending_action_fingerprint(proposal_id="p",action="set_task_priority",args=args,artifact=None,precondition=c.pre,action_precondition={})
    action=PendingAction(proposal_id="p",action="set_task_priority",args=args,reason="r",expected_effect="e",before=before,precondition=c.pre,action_precondition={},fingerprint=fp)
    path=tmp_path/"state.sqlite"
    first=await WriteService(c,execution_store=SQLiteWriteExecutionStore(path)).execute(action,fp)
    assert c.calls==1 and first.status=="VERIFIED"  # immediate read reconciliation observes applied effect
    second=await WriteService(c,execution_store=SQLiteWriteExecutionStore(path)).execute(action,fp)
    assert c.calls==1 and second.status=="VERIFIED"


@pytest.mark.asyncio
async def test_new_proposal_same_semantic_content_can_execute_after_external_reset(tmp_path):
    from deploy_ci_cloud_agentv3.models.proposal import ProposalResult
    from deploy_ci_cloud_agentv3.services.pending_action import PendingActionFactory

    facade = FakeFacade()
    tooling = build_tooling(facade)
    client = InProcessMCPClient(tooling.registry, RUNTIME_TOOLS)
    path = tmp_path / "state.sqlite"
    proposal = ProposalResult(
        action="set_task_priority",
        args={"task_name": "task_a", "priority": 5},
        reason="raise priority",
        expected_effect="priority becomes 5",
    )
    factory = PendingActionFactory(client)

    first_action = await factory.build(proposal)
    first = await WriteService(client, execution_store=SQLiteWriteExecutionStore(path)).execute(
        first_action, first_action.fingerprint
    )
    assert first.status == "VERIFIED"
    assert facade.priority == 5
    assert len(facade.mutations) == 1

    # External system returns the platform to the same semantic proposal state.
    facade.priority = 3
    facade.configs["task_a"]["priority"] = 3

    second_action = await factory.build(proposal)
    assert second_action.proposal_id != first_action.proposal_id
    assert second_action.fingerprint != first_action.fingerprint

    second = await WriteService(client, execution_store=SQLiteWriteExecutionStore(path)).execute(
        second_action, second_action.fingerprint
    )
    assert second.status == "VERIFIED"
    assert second.id != first.id
    assert facade.priority == 5
    assert len(facade.mutations) == 2


@pytest.mark.asyncio
async def test_write_audit_events_keep_thread_and_run_scope(tmp_path):
    from deploy_ci_cloud_agentv3.services.audit import AuditStore

    facade = FakeFacade(); action = pending(facade)
    tooling = build_tooling(facade); client = InProcessMCPClient(tooling.registry, RUNTIME_TOOLS)
    db = tmp_path / "audit.sqlite"
    audit = AuditStore(db)
    result = await WriteService(
        client, audit=audit, execution_store=SQLiteWriteExecutionStore(db)
    ).execute(action, action.fingerprint, thread_id="thread-a", run_id="run-a")
    assert result.status == "VERIFIED"

    events = audit.query(run_id="run-a", limit=100)
    event_types = {event["event_type"] for event in events}
    assert {"WriteAttempt", "VerificationResult", "WriteResult"} <= event_types
    assert all(event["thread_id"] == "thread-a" and event["run_id"] == "run-a" for event in events)
