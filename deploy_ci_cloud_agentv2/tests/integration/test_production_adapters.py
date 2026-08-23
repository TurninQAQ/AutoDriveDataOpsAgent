"""Production-boundary tests using local deterministic HTTP transports only."""

from __future__ import annotations

import asyncio
import json
import httpx

from deploy_ci_cloud_agentv2 import build_system_context, invoke, resume
from deploy_ci_cloud_agentv2.agent.budgets import RuntimeBudgets
from deploy_ci_cloud_agentv2.config import PlatformConfig, ProviderConfig
from deploy_ci_cloud_agentv2.platform import InMemoryReadFacade, MCPPlatformFacade
from deploy_ci_cloud_agentv2.providers.http_structured import HTTPStructuredProvider


def _provider_config(**overrides):
    values = {
        "name": "fake-qwen",
        "model": "fake-model",
        "endpoint": "https://provider.test/v1/chat/completions",
        "api_key_env": "AUTODRIVE_TEST_PROVIDER_KEY",
        "max_retries": 1,
        "retry_backoff_seconds": 0,
    }
    values.update(overrides)
    return ProviderConfig(**values)


def _response(content: object, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code,
        json={"choices": [{"message": {"content": json.dumps(content)}}]},
        request=httpx.Request("POST", "https://provider.test/v1/chat/completions"),
    )


def test_structured_provider_drives_real_graph_with_local_http_transport(monkeypatch):
    monkeypatch.setenv("AUTODRIVE_TEST_PROVIDER_KEY", "test-only-key")
    responses = [
        {
            "kind": "SINGLE_TOOL_CALL",
            "proposed_goal_descriptor": {
                "descriptor_version": 1,
                "goals": [{"kind": "READ_TASK_STATE", "goal_id": "g1", "target": "task_A"}],
            },
            "call": {"call_id": "read-1", "tool_name": "get_task_detail", "arguments": {"task_name": "task_A"}},
        },
        {"kind": "FINAL_CANDIDATE", "response": "task_A is RUNNING", "referenced_goal_ids": ["g1"]},
    ]
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return _response(responses.pop(0))

    provider = HTTPStructuredProvider(
        _provider_config(max_retries=0),
        transport=httpx.MockTransport(handler),
    )
    facade = InMemoryReadFacade(
        responses={"get_task_detail": {"task_name": "task_A", "state": "RUNNING"}}
    )
    context = build_system_context(provider, read_facade=facade, budgets=RuntimeBudgets(max_agent_steps=5))
    result = asyncio.run(invoke("what is task_A status?", thread_id="production-read", system_context=context))

    assert result.status == "COMPLETED"
    assert len(calls) == 2
    assert "UNTRUSTED_EXTERNAL_DATA" in calls[0]["messages"][0]["content"]
    assert "test-only-key" not in json.dumps(calls)


def test_malformed_provider_output_is_bounded_and_graph_recovers(monkeypatch):
    monkeypatch.setenv("AUTODRIVE_TEST_PROVIDER_KEY", "test-only-key")
    responses: list[httpx.Response] = [
        httpx.Response(200, content=b"not-json", request=httpx.Request("POST", "https://provider.test/v1/chat/completions")),
        _response({
            "kind": "SINGLE_TOOL_CALL",
            "proposed_goal_descriptor": {
                "descriptor_version": 1,
                "goals": [{"kind": "READ_TASK_STATE", "goal_id": "g1", "target": "task_A"}],
            },
            "call": {"call_id": "read-1", "tool_name": "get_task_detail", "arguments": {"task_name": "task_A"}},
        }),
        _response({"kind": "FINAL_CANDIDATE", "response": "recovered", "referenced_goal_ids": ["g1"]}),
    ]

    async def handler(_request: httpx.Request) -> httpx.Response:
        return responses.pop(0)

    provider = HTTPStructuredProvider(
        _provider_config(max_retries=0), transport=httpx.MockTransport(handler)
    )
    facade = InMemoryReadFacade(
        responses={"get_task_detail": {"task_name": "task_A", "state": "RUNNING"}}
    )
    context = build_system_context(provider, read_facade=facade, budgets=RuntimeBudgets(max_agent_steps=6))
    result = asyncio.run(invoke("what is task_A status?", thread_id="provider-recovery", system_context=context))
    assert result.status == "COMPLETED"
    assert result.response == "recovered"
    assert any(event.event_type == "AgentDecisionRejected" for event in context.event_store.all())


def test_provider_http_429_is_bounded_retry(monkeypatch):
    monkeypatch.setenv("AUTODRIVE_TEST_PROVIDER_KEY", "test-only-key")
    calls = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, json={"error": "rate limit"}, request=httpx.Request("POST", "https://provider.test"))
        return _response({"kind": "FINAL_CANDIDATE", "response": "x", "referenced_goal_ids": []})

    provider = HTTPStructuredProvider(_provider_config(), transport=httpx.MockTransport(handler))
    # Parsing is intentionally reached only after the transport retry.
    from deploy_ci_cloud_agentv2.agent.context import (
        AgentContext, OperatingGuidanceContext, RuntimeStructuredContext,
        SemanticObservationContext,
    )
    from deploy_ci_cloud_agentv2.agent.budgets import BudgetState
    from deploy_ci_cloud_agentv2.agent.evidence import EvidenceProjection
    from deploy_ci_cloud_agentv2.agent.identity import RequestIdentity
    from deploy_ci_cloud_agentv2.agent.state import ThreadHistory

    rid = RequestIdentity("thread", "request", "turn")
    context = AgentContext(
        user_input="x", messages=(),
        runtime_structured=RuntimeStructuredContext(rid, None, None, (), EvidenceProjection((), 0, 0, 0), BudgetState(RuntimeBudgets()), None, (), True),
        operating_guidance=OperatingGuidanceContext("v", "h", ()),
        semantic_observations=SemanticObservationContext(()),
        thread_history=(), new_turn=True,
    )
    result = asyncio.run(provider.generate(context))
    assert result.response == "x"
    assert calls == 2


def test_mcp_facade_maps_jsonrpc_and_preserves_runtime_contract_boundary():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        name = payload["params"]["name"]
        if name == "get_task_detail":
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": payload["id"], "result": {"task_name": "task_A", "state": "RUNNING"}}, request=request)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": payload["id"], "result": {"ok": True, "task_name": "task_A"}}, request=request)

    facade = MCPPlatformFacade(
        PlatformConfig(endpoint="https://platform.test/mcp", max_retries=0),
        transport=httpx.MockTransport(handler),
    )
    value = facade.get_task_detail("task_A")
    assert value["task_name"] == "task_A"
    assert requests[0]["method"] == "tools/call"
    assert requests[0]["params"]["name"] == "get_task_detail"


def test_mcp_sandbox_write_requires_approval_and_executes_once(monkeypatch):
    monkeypatch.setenv("AUTODRIVE_TEST_PROVIDER_KEY", "test-only-key")
    provider_responses = [
        {
            "kind": "SINGLE_TOOL_CALL",
            "proposed_goal_descriptor": {
                "descriptor_version": 1,
                "goals": [{"kind": "RESUME_TASK", "goal_id": "g1", "target": "task_A"}],
            },
            "call": {"call_id": "write-1", "tool_name": "resume_task", "arguments": {"task_name": "task_A"}},
        },
        {"kind": "FINAL_CANDIDATE", "response": "task_A resumed", "referenced_goal_ids": ["g1"]},
    ]
    provider_calls = 0

    async def provider_handler(_request: httpx.Request) -> httpx.Response:
        nonlocal provider_calls
        provider_calls += 1
        return _response(provider_responses.pop(0))

    task_state = "STOPPED"
    mutation_count = 0

    def platform_handler(request: httpx.Request) -> httpx.Response:
        nonlocal task_state, mutation_count
        payload = json.loads(request.content)
        name = payload["params"]["name"]
        if name == "get_task_detail":
            value = {"task_name": "task_A", "state": task_state, "exists": True, "entity_version": "1"}
        elif name == "resume_task":
            mutation_count += 1
            task_state = "RUNNING"
            value = {"ok": True, "task_name": "task_A", "state": "RUNNING", "execution_id": "sandbox-1"}
        else:
            value = {"ok": True, "task_name": "task_A"}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": payload["id"], "result": value}, request=request)

    provider = HTTPStructuredProvider(
        _provider_config(max_retries=0), transport=httpx.MockTransport(provider_handler)
    )
    facade = MCPPlatformFacade(
        PlatformConfig(endpoint="https://platform.test/mcp", max_retries=0),
        transport=httpx.MockTransport(platform_handler),
    )
    context = build_system_context(provider, read_facade=facade, budgets=RuntimeBudgets(max_agent_steps=6))
    first = asyncio.run(invoke("resume task_A", thread_id="mcp-write", system_context=context))
    assert first.status == "INTERRUPTED"
    assert mutation_count == 0

    from deploy_ci_cloud_agentv2.safety.approval import ApprovalDecision, ResumeInput

    pending = first.pending_interrupt
    second = asyncio.run(resume(
        thread_id="mcp-write",
        resume_input=ResumeInput(ApprovalDecision.APPROVE, pending.approval_request_id, pending.transaction_id, pending.fingerprint),
        system_context=context,
    ))
    assert second.status == "COMPLETED"
    assert mutation_count == 1
    assert provider_calls == 2
