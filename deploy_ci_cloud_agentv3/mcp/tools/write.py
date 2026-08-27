from __future__ import annotations

from typing import Any
from deploy_ci_cloud_agentv3.mcp.registry import ToolDefinition, ToolRegistry
from deploy_ci_cloud_agentv3.mcp.schemas import CapturePreconditionArgs, RuntimeDeleteTaskArgs, RuntimeResumeTaskArgs, RuntimeSetTaskPriorityArgs, RuntimeStopTaskArgs, RuntimeSubmitTaskArgs, SubmitArtifactArgs, VerificationSnapshotArgs, GetTaskConfigForVerificationArgs


def register_runtime_tools(registry: ToolRegistry, facade: Any, artifacts: Any) -> None:
    registry.register(ToolDefinition("capture_write_precondition", "Capture current platform state fingerprint for deterministic write admission.", "RUNTIME_INTERNAL", CapturePreconditionArgs, facade.get_write_precondition))
    registry.register(ToolDefinition("get_action_verification_snapshot", "Read post-mutation evidence without requiring task YAML to still exist.", "RUNTIME_INTERNAL", VerificationSnapshotArgs, facade.get_action_verification_snapshot))
    registry.register(ToolDefinition("get_prepared_artifact", "Resolve a previously prepared immutable task artifact for runtime review/execution.", "RUNTIME_INTERNAL", SubmitArtifactArgs, artifacts.get))

    def get_task_config_for_verification(task_name: str):
        helper = getattr(facade, "get_task_config_for_verification", None)
        if callable(helper):
            return helper(task_name)
        query = getattr(facade, "task_query_service", None)
        if query is None or not hasattr(query, "load_config"):
            raise RuntimeError("platform facade does not expose deterministic task config readback")
        _path, config = query.load_config(task_name)
        return {"task_name": task_name, "config": config}

    registry.register(ToolDefinition("get_task_config_for_verification", "Runtime-only exact task config readback for post-submit artifact verification.", "RUNTIME_INTERNAL", GetTaskConfigForVerificationArgs, get_task_config_for_verification))
    registry.register(ToolDefinition("set_task_priority", "REAL WRITE: mutate task priority. Runtime-only.", "WRITE", RuntimeSetTaskPriorityArgs, facade.set_task_priority))
    registry.register(ToolDefinition("resume_task", "REAL WRITE: resume task/datasets. Runtime-only.", "WRITE", RuntimeResumeTaskArgs, facade.resume_task))
    registry.register(ToolDefinition("stop_task", "REAL WRITE: stop task/datasets. Runtime-only.", "WRITE", RuntimeStopTaskArgs, facade.stop_task))
    registry.register(ToolDefinition("delete_task", "REAL WRITE: delete task. Runtime-only.", "WRITE", RuntimeDeleteTaskArgs, facade.delete_task))
    registry.register(ToolDefinition("submit_task", "REAL WRITE: submit validated task config. Runtime-only.", "WRITE", RuntimeSubmitTaskArgs, facade.submit_task))
