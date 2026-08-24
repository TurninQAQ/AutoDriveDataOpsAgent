"""Production-boundary tests using local deterministic HTTP transports only."""

from __future__ import annotations

import asyncio
import json
import httpx
import pytest

from deploy_ci_cloud_agentv2 import build_system_context, invoke, reconcile, resume
from deploy_ci_cloud_agentv2.agent.budgets import RuntimeBudgets
from deploy_ci_cloud_agentv2.config import ConfigurationError, PlatformConfig, ProviderConfig
from deploy_ci_cloud_agentv2.platform import InMemoryReadFacade, MCPPlatformFacade
from deploy_ci_cloud_agentv2.providers.http_structured import HTTPStructuredProvider
from deploy_ci_cloud_agentv2.providers import ScriptedProvider
from deploy_ci_cloud_agentv2.providers.errors import ProviderResponseInvalid, ProviderTransportFailure
from deploy_ci_cloud_agentv2.agent.decisions import SingleToolCall, ToolCall
from deploy_ci_cloud_agentv2.agent.goals import GoalDescriptor, ResumeTask
from deploy_ci_cloud_agentv2.safety.approval import ApprovalDecision, ResumeInput
from deploy_ci_cloud_agentv2.tools.write_runtime import MutationOutcomeUnknown


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


def _minimal_provider_context():
    from deploy_ci_cloud_agentv2.agent.context import (
        AgentContext, OperatingGuidanceContext, RuntimeStructuredContext,
        SemanticObservationContext,
    )
    from deploy_ci_cloud_agentv2.agent.budgets import BudgetState
    from deploy_ci_cloud_agentv2.agent.evidence import EvidenceProjection
    from deploy_ci_cloud_agentv2.agent.identity import RequestIdentity
    from deploy_ci_cloud_agentv2.agent.state import ThreadHistory

    rid = RequestIdentity("thread", "request", "turn")
    return AgentContext(
        user_input="x", messages=(),
        runtime_structured=RuntimeStructuredContext(
            rid, None, None, (), EvidenceProjection((), 0, 0, 0),
            BudgetState(RuntimeBudgets()), None, (), True,
        ),
        operating_guidance=OperatingGuidanceContext("v", "h", ()),
        semantic_observations=SemanticObservationContext(()),
        thread_history=(), new_turn=True,
    )


def test_provider_prompt_requires_initial_goal_descriptor():
    provider = HTTPStructuredProvider(_provider_config(max_retries=0))
    request = provider._build_request(_minimal_provider_context())
    system = request.messages[0]["content"]
    user = request.messages[1]["content"]

    assert "proposed_goal_descriptor is REQUIRED on the first tool or final decision" in system
    assert '"required_when": "runtime_structured_context.goal_descriptor is null"' in user
    assert '"descriptor_version": 1' in user
    assert '"goals"' in user
    assert '"call_id": "call_1"' in user
    assert '"INSPECT_GPU": ["goal_id", "kind"]' in user
    assert "never omit call_id or use null" in system
    assert "platform-wide INSPECT_QUEUE" in system


def test_qwen_strict_schema_request_contains_canonical_contract():
    provider = HTTPStructuredProvider(
        _provider_config(
            name="qwen",
            model="qwen3.7-plus-2026-05-26",
            structured_output_mode="json_schema",
            max_retries=0,
        )
    )
    request = provider._build_request(_minimal_provider_context())
    body = provider._request_body(request, mode="json_schema")
    response_format = body["response_format"]

    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "agent_decision"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert schema["anyOf"]
    assert "proposed_goal_descriptor" in schema["anyOf"][0]["required"]
    assert "call" in schema["anyOf"][0]["required"]
    assert body["enable_thinking"] is False
    assert body["temperature"] == 0


def test_legacy_json_object_mode_remains_default_for_old_qwen_model():
    provider = HTTPStructuredProvider(_provider_config(name="qwen", model="qwen-plus-2025-07-28"))
    request = provider._build_request(_minimal_provider_context())
    body = provider._request_body(request, mode="json_object")
    assert provider._structured_output_mode() == "json_object"
    assert body["response_format"] == {
        "type": "json_object"
    }
    assert "enable_thinking" not in body


@pytest.mark.parametrize(("thinking_mode", "expected"), [("enabled", True), ("disabled", False)])
def test_explicit_thinking_policy_is_serialized_for_legacy_mode(thinking_mode, expected):
    provider = HTTPStructuredProvider(
        _provider_config(
            name="qwen",
            model="qwen-plus-2025-07-28",
            thinking_mode=thinking_mode,
        )
    )
    request = provider._build_request(_minimal_provider_context())
    assert provider._request_body(request, mode="json_object")["enable_thinking"] is expected


def test_unsupported_thinking_policy_fails_closed():
    with pytest.raises(ConfigurationError, match="thinking_mode"):
        _provider_config(thinking_mode="sometimes")


def test_json_schema_mode_fails_closed_without_regeneration(monkeypatch):
    monkeypatch.setenv("AUTODRIVE_TEST_PROVIDER_KEY", "test-only-key")
    calls = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _response({
            "kind": "SINGLE_TOOL_CALL",
            "proposed_goal_descriptor": {
                "descriptor_version": 1,
                "goals": [{"kind": "INSPECT_GPU", "goal_id": "g1"}],
            },
            "call": {"call_id": "", "tool_name": "get_gpu_pool", "arguments": {}},
        })

    provider = HTTPStructuredProvider(
        _provider_config(
            name="qwen",
            model="qwen3.7-plus-2026-05-26",
            structured_output_mode="json_schema",
            max_retries=0,
        ),
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ProviderResponseInvalid):
        asyncio.run(provider.generate(_minimal_provider_context()))
    assert calls == 1


def test_json_object_mode_allows_one_bounded_schema_regeneration(monkeypatch):
    monkeypatch.setenv("AUTODRIVE_TEST_PROVIDER_KEY", "test-only-key")
    calls: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return _response({
                "kind": "SINGLE_TOOL_CALL",
                "proposed_goal_descriptor": {
                    "descriptor_version": 1,
                    "goals": [{"kind": "INSPECT_GPU", "goal_id": "g1"}],
                },
                "call": {"call_id": "", "tool_name": "get_gpu_pool", "arguments": {}},
            })
        return _response({
            "kind": "SINGLE_TOOL_CALL",
            "proposed_goal_descriptor": {
                "descriptor_version": 1,
                "goals": [{"kind": "INSPECT_GPU", "goal_id": "g1"}],
            },
            "call": {"call_id": "read-1", "tool_name": "get_gpu_pool", "arguments": {}},
        })

    provider = HTTPStructuredProvider(
        _provider_config(max_retries=0), transport=httpx.MockTransport(handler)
    )
    result = asyncio.run(provider.generate(_minimal_provider_context()))
    assert result.call.call_id == "read-1"
    assert len(calls) == 2
    assert calls[1]["messages"][-1]["role"] == "user"
    assert "failed the V2 AgentDecision schema" in calls[1]["messages"][-1]["content"]


def test_json_object_mode_second_schema_failure_is_bounded(monkeypatch):
    monkeypatch.setenv("AUTODRIVE_TEST_PROVIDER_KEY", "test-only-key")
    calls = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _response({
            "kind": "SINGLE_TOOL_CALL",
            "proposed_goal_descriptor": {
                "descriptor_version": 1,
                "goals": [{"kind": "INSPECT_GPU", "goal_id": "g1"}],
            },
            "call": {"call_id": "", "tool_name": "get_gpu_pool", "arguments": {}},
        })

    provider = HTTPStructuredProvider(
        _provider_config(max_retries=0), transport=httpx.MockTransport(handler)
    )
    with pytest.raises(ProviderResponseInvalid):
        asyncio.run(provider.generate(_minimal_provider_context()))
    assert calls == 2


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
    runtime = json.dumps(calls[0])
    assert "available_tools" in runtime
    assert "get_task_detail" in runtime
    assert "allowed_goal_types" in runtime
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
        return _response({
            "kind": "FINAL_CANDIDATE",
            "proposed_goal_descriptor": {
                "descriptor_version": 1,
                "goals": [{"kind": "INSPECT_GPU", "goal_id": "g1"}],
            },
            "response": "x",
            "referenced_goal_ids": [],
        })

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


@pytest.mark.parametrize("status_code", [429, 500, 502, 503])
def test_provider_retryable_http_failure_is_bounded_and_typed(monkeypatch, status_code):
    monkeypatch.setenv("AUTODRIVE_TEST_PROVIDER_KEY", "test-only-key")
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status_code, content=b"temporary", request=request)

    provider = HTTPStructuredProvider(
        _provider_config(max_retries=2), transport=httpx.MockTransport(handler)
    )
    with pytest.raises(ProviderTransportFailure):
        asyncio.run(provider.generate(_minimal_provider_context()))
    assert calls == 3


@pytest.mark.parametrize("failure", [
    httpx.ConnectError("connection refused"),
    httpx.ReadTimeout("read timeout"),
])
def test_provider_network_failure_exhaustion_never_leaks_transport_exception(monkeypatch, failure):
    monkeypatch.setenv("AUTODRIVE_TEST_PROVIDER_KEY", "test-only-key")
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise type(failure)(str(failure), request=request)

    provider = HTTPStructuredProvider(
        _provider_config(max_retries=1), transport=httpx.MockTransport(handler)
    )
    with pytest.raises(ProviderTransportFailure):
        asyncio.run(provider.generate(_minimal_provider_context()))
    assert calls == 2


@pytest.mark.parametrize("response", [
    httpx.Response(200, content=b"", request=httpx.Request("POST", "https://provider.test")),
    httpx.Response(200, content=b"{partial", request=httpx.Request("POST", "https://provider.test")),
])
def test_provider_malformed_body_is_typed_response_rejection(monkeypatch, response):
    monkeypatch.setenv("AUTODRIVE_TEST_PROVIDER_KEY", "test-only-key")

    async def handler(_request: httpx.Request) -> httpx.Response:
        return response

    provider = HTTPStructuredProvider(
        _provider_config(max_retries=0), transport=httpx.MockTransport(handler)
    )
    with pytest.raises(ProviderResponseInvalid):
        asyncio.run(provider.generate(_minimal_provider_context()))


def test_provider_missing_secret_is_typed_unavailable(monkeypatch):
    monkeypatch.delenv("AUTODRIVE_TEST_PROVIDER_KEY", raising=False)
    provider = HTTPStructuredProvider(_provider_config(max_retries=0))
    with pytest.raises(ProviderTransportFailure):
        asyncio.run(provider.generate(_minimal_provider_context()))


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


def test_mcp_remote_jsonrpc_error_after_effect_is_unknown_and_never_replayed():
    """A remote tool error is not proof that the mutation did not happen."""
    descriptor = GoalDescriptor(1, (ResumeTask("g1", "task_A"),))
    provider = ScriptedProvider(
        [SingleToolCall(ToolCall("write-1", "resume_task", {"task_name": "task_A"}), descriptor)],
        repeat_last=True,
    )
    task_state = "STOPPED"
    mutation_count = 0

    def platform_handler(request: httpx.Request) -> httpx.Response:
        nonlocal task_state, mutation_count
        payload = json.loads(request.content)
        name = payload["params"]["name"]
        if name == "get_task_detail":
            result = {"task_name": "task_A", "state": task_state, "exists": True, "entity_version": "1"}
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": payload["id"], "result": result}, request=request)
        if name == "resume_task":
            mutation_count += 1
            task_state = "RUNNING"
            return httpx.Response(
                200,
                json={"jsonrpc": "2.0", "id": payload["id"], "error": {"code": "RESPONSE_BUILD_FAILED"}},
                request=request,
            )
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": payload["id"], "result": {}}, request=request)

    facade = MCPPlatformFacade(
        PlatformConfig(endpoint="https://platform.test/mcp", max_retries=0),
        transport=httpx.MockTransport(platform_handler),
    )
    context = build_system_context(provider, read_facade=facade)
    first = asyncio.run(invoke("resume task_A", thread_id="mcp-unknown-after-effect", system_context=context))
    pending = first.pending_interrupt
    second = asyncio.run(
        resume(
            thread_id="mcp-unknown-after-effect",
            resume_input=ResumeInput(
                ApprovalDecision.APPROVE,
                pending.approval_request_id,
                pending.transaction_id,
                pending.fingerprint,
            ),
            system_context=context,
        )
    )
    assert second.status == "CONTROLLED_TERMINAL"
    assert second.state["current_request"].write_transaction.mutation_result.outcome.value == "OUTCOME_UNKNOWN"
    assert second.state["current_request"].write_transaction.status.value == "RECONCILIATION_REQUIRED"
    assert mutation_count == 1
    assert any(event.event_type == "WriteReplayBlocked" for event in context.event_store.all())

    reconciled = asyncio.run(reconcile(thread_id="mcp-unknown-after-effect", system_context=context))
    assert reconciled.effect_confirmed is True
    assert reconciled.replay_allowed is False
    assert mutation_count == 1

    # Reusing the original approval/transaction cannot cross the mutation
    # boundary again after reconciliation.
    with pytest.raises(ValueError):
        asyncio.run(
            resume(
                thread_id="mcp-unknown-after-effect",
                resume_input=ResumeInput(
                    ApprovalDecision.APPROVE,
                    pending.approval_request_id,
                    pending.transaction_id,
                    pending.fingerprint,
                ),
                system_context=context,
            )
        )
    assert mutation_count == 1


def test_mcp_connection_drop_after_write_dispatch_is_unknown():
    mutation_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal mutation_count
        mutation_count += 1
        raise httpx.ReadError("connection reset after dispatch", request=request)

    facade = MCPPlatformFacade(
        PlatformConfig(endpoint="https://platform.test/mcp", max_retries=2, retry_backoff_seconds=0),
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(MutationOutcomeUnknown):
        facade.resume_task("task_A")
    assert mutation_count == 1
