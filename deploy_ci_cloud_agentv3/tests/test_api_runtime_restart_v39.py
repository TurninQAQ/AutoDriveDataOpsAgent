from __future__ import annotations

import pytest

pytest.importorskip("mcp")
pytest.importorskip("langgraph")
pytest.importorskip("langgraph.checkpoint.sqlite.aio")

pytestmark = [pytest.mark.integration, pytest.mark.real_langgraph]

import httpx

from deploy_ci_cloud_agentv3.agent.runtime import AgentRuntime
from deploy_ci_cloud_agentv3.api.app import create_app
from deploy_ci_cloud_agentv3.api.dependencies import AppServices
from deploy_ci_cloud_agentv3.api.events import EventBroker
from deploy_ci_cloud_agentv3.persistence.audit_store import AuditStore
from deploy_ci_cloud_agentv3.persistence.checkpoint import CheckpointerFactory
from deploy_ci_cloud_agentv3.persistence.run_store import RunStore
from deploy_ci_cloud_agentv3.providers.base import AssistantMessage, ToolCall
from deploy_ci_cloud_agentv3.providers.scripted import ScriptedProvider
from deploy_ci_cloud_agentv3.tests.fakes import FakeFacade


def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


@pytest.mark.asyncio
async def test_api_proposal_survives_app_and_runtime_restart_then_approves(tmp_path, monkeypatch):
    db = tmp_path / "state.sqlite"
    checkpoint = tmp_path / "checkpoints.sqlite"
    monkeypatch.setenv("AUTODRIVE_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("AUTODRIVE_DB_PATH", str(db))
    facade = FakeFacade()

    async with CheckpointerFactory.open("sqlite", path=checkpoint) as saver:
        runtime = AgentRuntime.local(
            ScriptedProvider([
                AssistantMessage(tool_calls=[ToolCall(
                    id="proposal-api",
                    name="propose_set_task_priority",
                    arguments={"task_name": "task_a", "priority": 5},
                )]),
            ]),
            facade=facade,
            checkpointer=saver,
            audit_path=str(db),
        )
        services = AppServices(runtime, RunStore(db), AuditStore(db), EventBroker())
        app = create_app(services); app.state.services = services
        async with _client(app) as client:
            response = await client.post("/runs", json={"message": "raise task_a priority"})
            assert response.status_code == 200
            waiting = response.json()
            assert waiting["status"] == "WAITING_FOR_REVIEW"
            run_id, thread_id = waiting["run_id"], waiting["thread_id"]
            fingerprint = waiting["pending_action"]["fingerprint"]

    # New checkpointer, runtime, API app and stores: only disk state + platform survive.
    async with CheckpointerFactory.open("sqlite", path=checkpoint) as saver:
        runtime = AgentRuntime.local(
            ScriptedProvider([AssistantMessage(content='{"status":"write_verified","message":"done"}')]),
            facade=facade,
            checkpointer=saver,
            audit_path=str(db),
        )
        services = AppServices(runtime, RunStore(db), AuditStore(db), EventBroker())
        app = create_app(services); app.state.services = services
        async with _client(app) as client:
            approved = await client.post(f"/runs/{run_id}/approve", json={"fingerprint": fingerprint})
            assert approved.status_code == 200
            done = approved.json()
            assert done["status"] == "COMPLETED"
            assert done["thread_id"] == thread_id
            assert done["final_response"]["status"] == "write_verified"
            assert facade.priority == 5 and len(facade.mutations) == 1

            replay = (await client.get(f"/runs/{run_id}/events")).text.lower()
            for event_name in (
                "run_created", "agent_tool_call", "agent_tool_result", "proposalcreated",
                "waiting_for_review", "review_approved", "writeattempt",
                "verificationresult", "writeresult", "final_response",
            ):
                assert event_name in replay
