from __future__ import annotations

import asyncio
import json
from pathlib import Path

from platform_agent.actions import WriteActionCoordinator
from platform_agent.approval import ApprovalStore
from platform_agent.models import AgentIntent, AgentPlan, ToolObservation
from platform_agent.policy import AgentPolicyEngine
from platform_agent.verification import ActionVerifier
from platform_core.services.queue_service import QueueService
from platform_core.services.verification_service import ActionVerificationSnapshotService
from platform_mcp.server import ALL_TOOL_NAMES, READ_ONLY_TOOL_NAMES, WRITE_PREP_TOOL_NAMES


def run(coro):
    return asyncio.run(coro)


def base_snapshot(task_name="release_a", *, priority=20, location="active"):
    return {
        "task_name": task_name, "task_exists": True, "config_file_exists": True,
        "dag_file_exists": True, "dag_id": f"batch_pipeline_universal_{task_name}",
        "priority": priority, "task_exclusive": True,
        "available_datasets": ["clip_001", "clip_002"], "selected_datasets": [],
        "queue": {"location": location, "position": 0 if location == "active" else 1,
                  "entry": {"task_name": task_name, "priority": priority} if location != "not_found" else None},
        "containers": [], "gpu_reservations": [], "airflow_dag_exists": True,
        "airflow_runs": [], "errors": {},
    }


class SnapshotClient:
    def __init__(self, snapshots):
        self.snapshots = list(snapshots)
        self.calls = []
    async def execute(self, calls):
        self.calls.extend(calls)
        out=[]
        for call in calls:
            if call.name == "get_action_verification_snapshot":
                data = self.snapshots.pop(0) if len(self.snapshots) > 1 else self.snapshots[0]
            else:
                data = {}
            out.append(ToolObservation(tool_name=call.name, arguments=call.arguments, ok=True, data=data))
        return out


def test_v08_adds_internal_verification_tool_without_exposing_it_to_model():
    assert "get_action_verification_snapshot" in WRITE_PREP_TOOL_NAMES
    assert "get_action_verification_snapshot" in ALL_TOOL_NAMES
    assert "get_action_verification_snapshot" not in READ_ONLY_TOOL_NAMES
    assert len(ALL_TOOL_NAMES) == 17


def test_priority_verification_requires_persisted_config_and_queue_priority():
    result = run(ActionVerifier(SnapshotClient([base_snapshot(priority=5)]), attempts=1, interval_sec=0).verify(
        action="set_task_priority", arguments={"task_name":"release_a","priority":5}, execution_result={"ok":True}))
    assert result.status == "verified"
    failed = run(ActionVerifier(SnapshotClient([base_snapshot(priority=20)]), attempts=1, interval_sec=0).verify(
        action="set_task_priority", arguments={"task_name":"release_a","priority":5}, execution_result={"ok":True}))
    assert failed.status == "failed"
    assert "config_priority_updated" in [x.name for x in failed.checks if not x.passed]


def test_submit_verification_checks_task_dag_queue_and_dagruns():
    snap=base_snapshot("release_generated",priority=10)
    snap["airflow_runs"]=[{"run_id":"r1","dataset_name":"clip_001","state":"queued"},{"run_id":"r2","dataset_name":"clip_002","state":"running"}]
    config={"priority":10,"task_exclusive":True,"datasets":[{"dataset_name":"clip_001"},{"dataset_name":"clip_002"}]}
    result=run(ActionVerifier(SnapshotClient([snap]),attempts=1,interval_sec=0).verify(
        action="submit_task",arguments={"task_prefix":"release","config":config},execution_result={"ok":True,"result":{"task_name":"release_generated","triggered":2}}))
    assert result.status == "verified"
    assert result.task_name == "release_generated"


def test_stop_verification_requires_runtime_resources_reclaimed_and_runs_not_active():
    before=base_snapshot(); before["airflow_runs"]=[{"run_id":"r1","dataset_name":"clip_001","state":"running"}]
    after=base_snapshot(location="not_found"); after["airflow_runs"]=[{"run_id":"r1","dataset_name":"clip_001","state":"failed"}]
    result=run(ActionVerifier(SnapshotClient([after]),attempts=1,interval_sec=0).verify(
        action="stop_task",arguments={"task_name":"release_a","datasets":[]},execution_result={"ok":True},baseline=before))
    assert result.status == "verified"
    leaked=dict(after); leaked["containers"]=[{"id":"c1","running":True}]
    failed=run(ActionVerifier(SnapshotClient([leaked]),attempts=1,interval_sec=0).verify(
        action="stop_task",arguments={"task_name":"release_a","datasets":[]},execution_result={"ok":True},baseline=before))
    assert failed.status == "failed"


def test_resume_verification_compares_baseline_and_requires_new_runs():
    before=base_snapshot(); before["airflow_runs"]=[{"run_id":"old1","dataset_name":"clip_001","state":"failed"}]
    after=base_snapshot(); after["airflow_runs"]=[{"run_id":"new1","dataset_name":"clip_001","state":"queued"},{"run_id":"old1","dataset_name":"clip_001","state":"failed"}]
    result=run(ActionVerifier(SnapshotClient([after]),attempts=1,interval_sec=0).verify(
        action="resume_task",arguments={"task_name":"release_a","datasets":[]},execution_result={"ok":True},baseline=before))
    assert result.status == "verified"
    assert next(x for x in result.checks if x.name=="new_dagruns_created").actual == ["clip_001"]


def test_delete_verification_requires_all_artifacts_and_runtime_state_absent():
    snap=base_snapshot(location="not_found")
    snap.update({"task_exists":False,"config_file_exists":False,"dag_file_exists":False,"priority":None,"airflow_dag_exists":False,"airflow_runs":[]})
    result=run(ActionVerifier(SnapshotClient([snap]),attempts=1,interval_sec=0).verify(
        action="delete_task",arguments={"task_name":"release_a"},execution_result={"ok":True}))
    assert result.status == "verified"
    assert all(x.passed for x in result.checks)


def test_airflow_unavailable_cannot_be_reported_as_success():
    snap=base_snapshot("release_generated",priority=10); snap["errors"]={"airflow":"connection refused"}; snap["airflow_dag_exists"]=None
    config={"priority":10,"task_exclusive":True,"datasets":[{"dataset_name":"clip_001"}]}
    result=run(ActionVerifier(SnapshotClient([snap]),attempts=1,interval_sec=0).verify(
        action="submit_task",arguments={"task_prefix":"release","config":config},execution_result={"result":{"task_name":"release_generated","triggered":1}}))
    assert result.verified is False


def test_verifier_retries_eventual_consistency_until_verified():
    client=SnapshotClient([base_snapshot(priority=20),base_snapshot(priority=5)])
    result=run(ActionVerifier(client,attempts=3,interval_sec=0).verify(
        action="set_task_priority",arguments={"task_name":"release_a","priority":5},execution_result={"ok":True}))
    assert result.status == "verified" and result.attempts == 2


class CoordinatorClient:
    def __init__(self, after):
        self.after=after; self.write_ran=False; self.calls=[]
    async def execute(self,calls):
        self.calls.extend(calls); out=[]
        for call in calls:
            if call.name=="get_write_precondition": data={"queue_sha256":"q","task_name":call.arguments.get("task_name",""),"task_config_sha256":"c","task_exists":True}
            elif call.name=="get_action_verification_snapshot": data=base_snapshot(priority=20) if not self.write_ran else self.after
            elif call.name=="set_task_priority": self.write_ran=True; data={"ok":True,"action":"set_task_priority","result":{"task_name":call.arguments["task_name"]}}
            else: data={}
            out.append(ToolObservation(tool_name=call.name,arguments=call.arguments,ok=True,data=data))
        return out


def prepare_priority(store,client):
    c=WriteActionCoordinator(client,AgentPolicyEngine(),store,verifier=ActionVerifier(client,attempts=1,interval_sec=0))
    p=AgentPlan(intent=AgentIntent.SET_TASK_PRIORITY,task_name="release_a",write_action={"task_name":"release_a","priority":5})
    item=run(c.prepare(state_user_text="priority=5",thread_id="t",plan=p,observations=[]))
    return c,item


def test_coordinator_only_marks_executed_after_verification_passes(tmp_path:Path):
    store=ApprovalStore(tmp_path/"a"); c,item=prepare_priority(store,CoordinatorClient(base_snapshot(priority=5)))
    result=run(c.execute_approval(item.approval_id))
    assert result.status=="executed" and result.verification_result["status"]=="verified"
    assert result.verification_baseline["priority"]==20


def test_write_tool_success_but_verification_failure_is_not_success(tmp_path:Path):
    store=ApprovalStore(tmp_path/"a"); c,item=prepare_priority(store,CoordinatorClient(base_snapshot(priority=20)))
    result=run(c.execute_approval(item.approval_id))
    assert result.status=="verification_failed"
    assert result.execution_result is not None and result.verification_result["status"]=="failed"


def test_snapshot_service_can_verify_deleted_task_without_loading_deleted_yaml(tmp_path:Path):
    q=tmp_path/"queue.lock"; q.write_text(json.dumps({"version":2,"active":None,"queue":[]}),encoding="utf-8")
    class Docker:
        def task_containers(self,task_name,datasets=None): return []
    class GPU:
        def reservations(self,cleanup_dead=True): return []
    class Gateway:
        def get_dag(self,dag_id): raise RuntimeError("GET DAG failed HTTP 404: not found")
    class Airflow:
        gateway=Gateway()
        def runs(self,dag_id,limit=100): raise AssertionError("must not run after 404")
    service=ActionVerificationSnapshotService(task_config_root=tmp_path/"tasks",dags_dir=tmp_path/"dags",queue_service=QueueService(q),docker_gateway=Docker(),gpu_service=GPU(),airflow_service=Airflow())
    snap=service.snapshot("deleted_task")
    assert snap["task_exists"] is False and snap["airflow_dag_exists"] is False and snap["containers"]==[]

def test_agent_settings_reads_verification_retry_config(monkeypatch, tmp_path: Path):
    from platform_agent.settings import AgentSettings
    from platform_core.settings import PlatformSettings

    monkeypatch.setenv("PLATFORM_HOME", str(tmp_path))
    monkeypatch.setenv("AIRFLOW_HOME", str(tmp_path / "airflow"))
    monkeypatch.setenv("PLATFORM_AGENT_VERIFY_ATTEMPTS", "7")
    monkeypatch.setenv("PLATFORM_AGENT_VERIFY_INTERVAL_SEC", "0.25")
    settings = AgentSettings.from_env(PlatformSettings.from_env())
    assert settings.verification_attempts == 7
    assert settings.verification_interval_sec == 0.25
    platform_script = (Path(__file__).resolve().parents[1] / "platform").read_text(encoding="utf-8")
    assert "PLATFORM_AGENT_VERIFY_ATTEMPTS" in platform_script
    assert "PLATFORM_AGENT_VERIFY_INTERVAL_SEC" in platform_script
