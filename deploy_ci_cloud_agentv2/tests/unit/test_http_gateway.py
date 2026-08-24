from __future__ import annotations

import json
import sys
import threading
import urllib.request

import pytest
import httpx

from deploy_ci_cloud_agentv2.agent.results import normalize_read_result
from deploy_ci_cloud_agentv2.platform.http_gateway import (
    GatewayDispatcher,
    StdioMCPClient,
    create_server,
)
from deploy_ci_cloud_agentv2.config import PlatformConfig
from deploy_ci_cloud_agentv2.platform.mcp import MCPPlatformFacade
from deploy_ci_cloud_agentv2.platform_backend.client import InProcessPlatformClient, PlatformBackendError
from deploy_ci_cloud_agentv2.agent.results import normalize_read_result
from deploy_ci_cloud_agentv2.platform_backend.core.errors import TaskConfigError


class FakeClient:
    def __init__(self):
        self.calls = []

    def call(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        return {"tool": name, "arguments": dict(arguments)}


def test_dispatch_preserves_request_id_and_forwards_read():
    client = FakeClient()
    response = GatewayDispatcher(client).dispatch(
        {
            "jsonrpc": "2.0",
            "id": "read-1",
            "method": "tools/call",
            "params": {"name": "get_gpu_pool", "arguments": {}},
        }
    )
    assert response["id"] == "read-1"
    assert response["result"]["structuredContent"]["tool"] == "get_gpu_pool"
    assert client.calls == [("get_gpu_pool", {"cleanup_dead": False})]


def test_queue_null_scope_is_adapted_to_stdio_string_default():
    client = FakeClient()
    response = GatewayDispatcher(client).dispatch(
        {
            "jsonrpc": "2.0",
            "id": "queue-1",
            "method": "tools/call",
            "params": {"name": "get_queue_state", "arguments": {"task_name": None}},
        }
    )
    assert "result" in response
    assert client.calls == [("get_queue_state", {"task_name": ""})]


def test_gpu_read_disables_legacy_cleanup_default():
    client = FakeClient()
    response = GatewayDispatcher(client).dispatch(
        {
            "jsonrpc": "2.0",
            "id": "gpu-1",
            "method": "tools/call",
            "params": {"name": "get_gpu_pool", "arguments": {}},
        }
    )
    assert "result" in response
    assert client.calls == [("get_gpu_pool", {"cleanup_dead": False})]


def test_dispatch_exposes_exact_v2_tools_and_rejects_unknown_tool():
    client = FakeClient()
    for name in (
        "get_task_detail",
        "get_gpu_pool",
        "search_knowledge",
        "get_queue_state",
        "diagnose_task",
        "resume_task",
        "submit_task",
        "stop_task",
        "delete_task",
        "set_task_priority",
    ):
        response = GatewayDispatcher(client).dispatch(
            {"jsonrpc": "2.0", "id": name, "method": "tools/call", "params": {"name": name}}
        )
        assert "result" in response
    rejected = GatewayDispatcher(client).dispatch(
        {"jsonrpc": "2.0", "id": "x", "method": "tools/call", "params": {"name": "list_tasks"}}
    )
    assert rejected["error"]["code"] == "TOOL_NOT_FOUND"


def test_dispatch_rejects_malformed_protocol_without_calling_client():
    client = FakeClient()
    cases = [
        {},
        {"jsonrpc": "1.0", "id": 1, "method": "tools/call", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "get_gpu_pool", "arguments": []}},
    ]
    for request in cases:
        response = GatewayDispatcher(client).dispatch(request)
        assert "error" in response
    assert client.calls == []


def test_mock_fault_can_drop_only_write_response_after_dispatch():
    client = FakeClient()
    dispatcher = GatewayDispatcher(client, drop_write_response_after_dispatch=True)
    write_response = dispatcher.dispatch(
        {
            "jsonrpc": "2.0",
            "id": "write-1",
            "method": "tools/call",
            "params": {"name": "set_task_priority", "arguments": {"task_name": "task_A", "priority": 5}},
        }
    )
    read_response = dispatcher.dispatch(
        {
            "jsonrpc": "2.0",
            "id": "read-1",
            "method": "tools/call",
            "params": {"name": "get_task_detail", "arguments": {"task_name": "task_A"}},
        }
    )
    assert write_response["error"]["code"] == "SANDBOX_RESPONSE_DROPPED"
    assert "result" in read_response
    assert client.calls == [
        ("set_task_priority", {"task_name": "task_A", "priority": 5}),
        ("get_task_detail", {"task_name": "task_A"}),
    ]


def test_environment_fault_is_restricted_to_mock_stage(monkeypatch):
    monkeypatch.setenv("AUTODRIVE_TEST_DROP_WRITE_RESPONSE_AFTER_DISPATCH", "1")
    monkeypatch.setenv("PLATFORM_STAGE_RUNTIME", "production")
    client = FakeClient()
    response = GatewayDispatcher(client).dispatch(
        {
            "jsonrpc": "2.0",
            "id": "write-1",
            "method": "tools/call",
            "params": {"name": "set_task_priority", "arguments": {"task_name": "task_A", "priority": 5}},
        }
    )
    assert "result" in response


def test_http_facade_normalizes_canonical_stdio_empty_queue():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": payload["id"],
                "result": {
                    "structuredContent": {
                        "version": 2,
                        "active": None,
                        "queue": [],
                    }
                },
            },
            request=request,
        )

    facade = MCPPlatformFacade(
        PlatformConfig(endpoint="https://platform.test/mcp", max_retries=0),
        transport=httpx.MockTransport(handler),
    )
    payload = facade.get_queue_state()

    assert payload == {"version": 2, "scope": "PLATFORM", "queue": []}
    result = normalize_read_result("get_queue_state", {"task_name": None}, payload)
    assert result.qualifies_for_evidence()


def test_stdio_client_uses_real_line_delimited_mcp_session():
    script = (
        "import json,sys\n"
        "for line in sys.stdin:\n"
        " m=json.loads(line)\n"
        " if m.get('method') == 'initialize':\n"
        "  print(json.dumps({'jsonrpc':'2.0','id':m['id'],'result':{'protocolVersion':'2025-06-18'}}),flush=True)\n"
        " elif m.get('method') == 'tools/call':\n"
        "  print(json.dumps({'jsonrpc':'2.0','id':m['id'],'result':{'structuredContent':{'ok':True,'name':m['params']['name']}}}),flush=True)\n"
    )
    client = StdioMCPClient([sys.executable, "-c", script], timeout_seconds=3)
    assert client.call("get_gpu_pool", {}) == {"ok": True, "name": "get_gpu_pool"}


def test_stdio_missing_task_error_is_narrowly_normalized_to_not_found():
    script = (
        "import json,sys\n"
        "for line in sys.stdin:\n"
        " m=json.loads(line)\n"
        " if m.get('method') == 'initialize':\n"
        "  print(json.dumps({'jsonrpc':'2.0','id':m['id'],'result':{}}),flush=True)\n"
        " elif m.get('method') == 'tools/call':\n"
        "  print(json.dumps({'jsonrpc':'2.0','id':m['id'],'result':{'structuredContent':{'isError':True,'content':[{'type':'text','text':'Error executing tool get_task_detail: Task config not found: /tmp/x'}]}}}),flush=True)\n"
    )
    client = StdioMCPClient([sys.executable, "-c", script], timeout_seconds=3)
    assert client.call("get_task_detail", {"task_name": "missing_task"}) == {
        "status": "NOT_FOUND",
        "task_name": "missing_task",
        "exists": False,
    }


def test_http_facade_normalizes_canonical_stdio_empty_queue():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": payload["id"],
                "result": {
                    "structuredContent": {
                        "version": 2,
                        "active": None,
                        "queue": [],
                    }
                },
            },
            request=request,
        )

    facade = MCPPlatformFacade(
        PlatformConfig(endpoint="https://platform.test/mcp", max_retries=0),
        transport=httpx.MockTransport(handler),
    )
    payload = facade.get_queue_state()

    assert payload == {"version": 2, "scope": "PLATFORM", "queue": []}
    result = normalize_read_result("get_queue_state", {"task_name": None}, payload)
    assert result.qualifies_for_evidence()


class _MissingTaskFacade:
    def get_task_detail(self, task_name):
        raise TaskConfigError("Task config not found: /tmp/tasks/%s/datasets_config.yaml" % task_name)


class _BrokenTaskFacade:
    def get_task_detail(self, task_name):
        raise TaskConfigError("Task config root must be a mapping")


def test_in_process_backend_maps_only_canonical_missing_task_error():
    client = InProcessPlatformClient(_MissingTaskFacade())
    assert client.call("get_task_detail", {"task_name": "missing_task"}) == {
        "status": "NOT_FOUND",
        "task_name": "missing_task",
        "exists": False,
    }


def test_in_process_backend_does_not_promote_generic_task_error_to_not_found():
    client = InProcessPlatformClient(_BrokenTaskFacade())
    with pytest.raises(PlatformBackendError) as error:
        client.call("get_task_detail", {"task_name": "broken_task"})
    assert error.value.code == "PLATFORM_TOOL_ERROR"


def test_in_process_write_requires_runtime_precondition_before_handler():
    with pytest.raises(PlatformBackendError) as error:
        InProcessPlatformClient(object()).call(
            "submit_task",
            {"task_name": "sandbox_task", "config": {}},
        )
    assert error.value.code == "WRITE_PRECONDITION_REQUIRED"


def test_http_write_facade_forwards_detached_precondition():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(
            200,
            json={"jsonrpc": "2.0", "id": payload["id"], "result": {"ok": True}},
            request=request,
        )

    facade = MCPPlatformFacade(
        PlatformConfig(endpoint="https://platform.test/mcp", max_retries=0),
        transport=httpx.MockTransport(handler),
    )
    facade.resume_task(
        "task_A",
        precondition={
            "target": "task_A",
            "tool_name": "resume_task",
            "fingerprint": "fp",
            "entity_version": "1",
            "state": {"status": "SUCCESS"},
        },
    )
    forwarded = requests[0]["params"]["arguments"]
    assert forwarded["task_name"] == "task_A"
    assert forwarded["precondition"]["fingerprint"] == "fp"
    assert forwarded["precondition"]["state"] == {"status": "SUCCESS"}


def test_platform_task_state_mapping_is_deterministic():
    from deploy_ci_cloud_agentv2.platform_backend.mcp.facade import _task_state_from_runs

    assert _task_state_from_runs([]) == "SUBMITTED"
    assert _task_state_from_runs([{"state": "running"}]) == "RUNNING"
    assert _task_state_from_runs([{"state": "success"}]) == "SUCCEEDED"
    assert _task_state_from_runs([{"state": "failed"}]) == "FAILED"


def test_http_health_and_json_rpc_endpoint():
    client = FakeClient()
    server = create_server("127.0.0.1", 0, client)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urllib.request.urlopen(base + "/health", timeout=3) as response:
            assert json.load(response) == {"status": "ok"}
        body = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": "http-1",
                "method": "tools/call",
                "params": {"name": "get_queue_state", "arguments": {}},
            }
        ).encode()
        request = urllib.request.Request(base + "/mcp", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=3) as response:
            payload = json.load(response)
        assert payload["id"] == "http-1"
        assert payload["result"]["structuredContent"]["tool"] == "get_queue_state"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
