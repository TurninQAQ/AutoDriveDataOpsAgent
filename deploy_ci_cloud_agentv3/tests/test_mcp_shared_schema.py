from __future__ import annotations

import inspect

import pytest

pytest.importorskip("mcp")

from deploy_ci_cloud_agentv3.mcp.factory import build_tooling
from deploy_ci_cloud_agentv3.mcp.profiles import AGENT_TOOLS, RUNTIME_TOOLS
from deploy_ci_cloud_agentv3.mcp.server import _tool_handler
from deploy_ci_cloud_agentv3.tests.fakes import FakeFacade


def test_mcp_handler_signature_is_derived_from_shared_pydantic_schema():
    tool = build_tooling(FakeFacade()).registry.get("set_task_priority")
    handler = _tool_handler(tool)
    sig = inspect.signature(handler)
    assert list(sig.parameters) == ["task_name", "priority", "precondition"]
    priority_annotation = repr(sig.parameters["priority"].annotation)
    assert "Ge(ge=0)" in priority_annotation
    assert "Le(le=100)" in priority_annotation
    assert tool.input_schema["properties"]["priority"]["minimum"] == 0
    assert tool.input_schema["properties"]["priority"]["maximum"] == 100


@pytest.mark.parametrize("tool_name", sorted(AGENT_TOOLS | RUNTIME_TOOLS))
def test_every_mcp_tool_handler_uses_valid_keyword_only_signature(tool_name):
    """Every shared schema must be representable as a valid MCP handler."""
    tool = build_tooling(FakeFacade()).registry.get(tool_name)
    signature = inspect.signature(_tool_handler(tool))

    assert list(signature.parameters) == list(tool.args_model.model_fields)
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )


def test_runtime_inherited_required_fields_following_optional_fields_are_valid():
    """Regression for RuntimeResume/Stop schemas with required-after-default fields."""
    tooling = build_tooling(FakeFacade())
    for tool_name in ("resume_task", "stop_task"):
        signature = inspect.signature(_tool_handler(tooling.registry.get(tool_name)))
        assert signature.parameters["datasets"].default is None
        assert signature.parameters["precondition"].default is inspect.Parameter.empty
