from __future__ import annotations

import json
import sys
import threading
import urllib.request

from deploy_ci_cloud_agentv2.platform.http_gateway import (
    GatewayDispatcher,
    StdioMCPClient,
    create_server,
)


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
