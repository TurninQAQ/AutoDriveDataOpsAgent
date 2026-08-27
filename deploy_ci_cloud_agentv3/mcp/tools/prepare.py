from __future__ import annotations

from deploy_ci_cloud_agentv3.mcp.registry import ToolDefinition, ToolRegistry
from deploy_ci_cloud_agentv3.mcp.schemas import TaskDraftArgs
from deploy_ci_cloud_agentv3.services.task_preparation import TaskPreparationService


def register_prepare_tools(registry: ToolRegistry, service: TaskPreparationService) -> None:
    registry.register(ToolDefinition("prepare_task_spec", "Build and validate a TaskSpec/YAML artifact from explicit draft fields plus deterministic platform defaults. No platform mutation.", "PREPARE", TaskDraftArgs, service.prepare))
