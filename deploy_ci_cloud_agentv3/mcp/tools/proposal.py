from __future__ import annotations

from deploy_ci_cloud_agentv3.mcp.registry import ToolDefinition, ToolRegistry
from deploy_ci_cloud_agentv3.mcp.schemas import DeleteTaskArgs, ResumeTaskArgs, SetTaskPriorityArgs, StopTaskArgs, SubmitArtifactArgs
from deploy_ci_cloud_agentv3.models.proposal import ProposalResult
from deploy_ci_cloud_agentv3.services.artifacts import ArtifactStore


def register_proposal_tools(registry: ToolRegistry, artifacts: ArtifactStore) -> None:
    def propose_set_task_priority(task_name: str, priority: int) -> ProposalResult:
        return ProposalResult(action="set_task_priority", args={"task_name": task_name, "priority": priority}, reason="Adjust the task priority requested by the current reasoning step.", expected_effect=f"Task {task_name} priority becomes {priority}.")

    def propose_resume_task(task_name: str, datasets: list[str] | None = None) -> ProposalResult:
        return ProposalResult(action="resume_task", args={"task_name": task_name, "datasets": datasets}, reason="Resume the selected task scope after human review.", expected_effect=f"Task {task_name} becomes eligible to continue running.")

    def propose_stop_task(task_name: str, datasets: list[str] | None = None) -> ProposalResult:
        return ProposalResult(action="stop_task", args={"task_name": task_name, "datasets": datasets}, reason="Stop the selected task scope after human review.", expected_effect=f"Task {task_name} stops active execution for the selected scope.")

    def propose_delete_task(task_name: str) -> ProposalResult:
        return ProposalResult(action="delete_task", args={"task_name": task_name}, reason="Delete the task only after explicit human review.", expected_effect=f"Task {task_name} and generated runtime artifacts are removed.")

    def propose_submit_task(artifact_id: str) -> ProposalResult:
        artifact = artifacts.get(artifact_id)
        return ProposalResult(action="submit_task", args={"artifact_id": artifact_id}, reason="Submit the exact prepared and validated YAML artifact after human review.", expected_effect=f"A new task using prefix {artifact.task_prefix} is created from artifact {artifact_id}.")

    registry.register(ToolDefinition("propose_set_task_priority", "Create a zero-side-effect priority-change proposal.", "PROPOSAL", SetTaskPriorityArgs, propose_set_task_priority))
    registry.register(ToolDefinition("propose_resume_task", "Create a zero-side-effect task-resume proposal.", "PROPOSAL", ResumeTaskArgs, propose_resume_task))
    registry.register(ToolDefinition("propose_stop_task", "Create a zero-side-effect task-stop proposal.", "PROPOSAL", StopTaskArgs, propose_stop_task))
    registry.register(ToolDefinition("propose_delete_task", "Create a zero-side-effect task-deletion proposal.", "PROPOSAL", DeleteTaskArgs, propose_delete_task))
    registry.register(ToolDefinition("propose_submit_task", "Create a zero-side-effect proposal bound to an existing prepared artifact.", "PROPOSAL", SubmitArtifactArgs, propose_submit_task))
