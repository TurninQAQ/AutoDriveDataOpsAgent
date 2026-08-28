from __future__ import annotations

from fastapi.testclient import TestClient

from deploy_ci_cloud_agentv3.api.app import create_app
from deploy_ci_cloud_agentv3.api.dependencies import AppServices
from deploy_ci_cloud_agentv3.api.events import EventBroker
from deploy_ci_cloud_agentv3.persistence.audit_store import AuditStore
from deploy_ci_cloud_agentv3.persistence.run_store import RunStore


class FakeRuntime:
    def __init__(self):
        self.mutations=0
        self.pending_by_thread={}
    async def start(self,thread_id,message,*,run_id=None):
        if "priority" in message.lower() or "优先级" in message:
            pending={"action":"set_task_priority","args":{"task_name":"task_A","priority":5},"before":{"priority":1},"reason":"low priority","expected_effect":"priority becomes 5","fingerprint":"fp1"}
            self.pending_by_thread[thread_id]=pending
            return {"__interrupt__":[pending]}
        return {"final_response":{"status":"informational","message":"diagnosis complete"}}
    async def review(self,thread_id,decision):
        kind=decision["decision"]
        if kind=="approve":
            assert decision["fingerprint"]==self.pending_by_thread[thread_id]["fingerprint"]
            self.mutations+=1
            return {"final_response":{"status":"write_verified","message":"verified"}}
        if kind=="edit":
            pending={**self.pending_by_thread[thread_id],"args":decision["args"],"fingerprint":"fp2"}
            self.pending_by_thread[thread_id]=pending
            return {"__interrupt__":[pending]}
        return {"final_response":{"status":"write_not_executed","message":"rejected"}}


def client(tmp_path):
    runtime=FakeRuntime(); db=tmp_path/"state.sqlite"
    services=AppServices(runtime,RunStore(db),AuditStore(db),EventBroker())
    return TestClient(create_app(services)),runtime


def test_read_run_health_and_sse_replay(tmp_path):
    c,runtime=client(tmp_path)
    with c:
        assert c.get("/health").status_code==200
        assert c.get("/ready").json()["status"]=="ready"
        run=c.post("/runs",json={"message":"task_A why failed?"}).json()
        assert run["status"]=="COMPLETED" and runtime.mutations==0
        fetched=c.get(f"/runs/{run['run_id']}").json(); assert fetched["final_response"]["status"]=="informational"
        sse=c.get(f"/runs/{run['run_id']}/events")
        assert "run_created" in sse.text and "final_response" in sse.text


def test_proposal_approve_reject_edit_and_fingerprint_binding(tmp_path):
    c,runtime=client(tmp_path)
    with c:
        run=c.post("/runs",json={"message":"priority too low"}).json()
        assert run["status"]=="WAITING_FOR_REVIEW" and run["pending_action"]["fingerprint"]=="fp1"
        assert c.post(f"/runs/{run['run_id']}/approve",json={"fingerprint":"attacker"}).status_code==409
        edited=c.post(f"/runs/{run['run_id']}/edit",json={"fingerprint":"fp1","args":{"task_name":"task_A","priority":4}}).json()
        assert edited["pending_action"]["fingerprint"]=="fp2"
        done=c.post(f"/runs/{run['run_id']}/approve",json={"fingerprint":"fp2"}).json()
        assert done["status"]=="COMPLETED" and runtime.mutations==1

        run2=c.post("/runs",json={"message":"priority too low"}).json()
        rejected=c.post(f"/runs/{run2['run_id']}/reject",json={"fingerprint":"fp1","reason":"no"}).json()
        assert rejected["status"]=="COMPLETED" and runtime.mutations==1
