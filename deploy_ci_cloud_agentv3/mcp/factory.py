from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from deploy_ci_cloud_agentv3.mcp.profiles import AGENT_TOOLS, RUNTIME_TOOLS
from deploy_ci_cloud_agentv3.mcp.registry import ToolRegistry
from deploy_ci_cloud_agentv3.mcp.tools import register_prepare_tools, register_proposal_tools, register_read_tools, register_runtime_tools
from deploy_ci_cloud_agentv3.services.artifacts import ArtifactStore
from deploy_ci_cloud_agentv3.services.task_preparation import TaskPreparationService


@dataclass
class Tooling:
    registry: ToolRegistry
    artifacts: ArtifactStore
    task_preparation: TaskPreparationService


def build_tooling(facade: Any) -> Tooling:
    registry = ToolRegistry()
    artifacts = ArtifactStore()
    preparation = TaskPreparationService(facade, artifacts)
    register_read_tools(registry, facade)
    register_prepare_tools(registry, preparation)
    register_proposal_tools(registry, artifacts)
    register_runtime_tools(registry, facade, artifacts)
    return Tooling(registry=registry, artifacts=artifacts, task_preparation=preparation)


def assert_profile_boundaries(registry: ToolRegistry) -> None:
    agent_names = {item.name for item in registry.list(AGENT_TOOLS)}
    runtime_names = {item.name for item in registry.list(RUNTIME_TOOLS)}
    forbidden = {"set_task_priority", "resume_task", "stop_task", "delete_task", "submit_task"}
    if agent_names & forbidden:
        raise AssertionError("agent profile exposes real write tools")
    if not forbidden <= runtime_names:
        raise AssertionError("runtime profile is missing real write tools")
