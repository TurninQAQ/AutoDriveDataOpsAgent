from __future__ import annotations

from typing import Any
from deploy_ci_cloud_agentv3.mcp.registry import ToolDefinition, ToolRegistry
from deploy_ci_cloud_agentv3.mcp.schemas import DiagnoseTaskArgs, GetGpuPoolArgs, GetQueueStateArgs, GetTaskDetailArgs, SearchKnowledgeArgs


def register_read_tools(registry: ToolRegistry, facade: Any) -> None:
    registry.register(ToolDefinition("get_task_detail", "Get task configuration, queue state and recent Airflow runs.", "READ", GetTaskDetailArgs, facade.get_task_detail))
    registry.register(ToolDefinition("get_gpu_pool", "Get GPU devices and current reservations.", "READ", GetGpuPoolArgs, facade.get_gpu_pool))
    registry.register(ToolDefinition("get_queue_state", "Get global queue state or one task's queue position.", "READ", GetQueueStateArgs, facade.get_queue_state))
    registry.register(ToolDefinition("diagnose_task", "Collect deterministic diagnosis evidence for a task.", "READ", DiagnoseTaskArgs, facade.diagnose_task))
    registry.register(ToolDefinition("search_knowledge", "Search AutoDrive runbooks and platform knowledge.", "READ", SearchKnowledgeArgs, facade.search_knowledge))
