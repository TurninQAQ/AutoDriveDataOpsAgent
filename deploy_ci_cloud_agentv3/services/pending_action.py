from __future__ import annotations

import uuid
from typing import Any

from deploy_ci_cloud_agentv3.models.common import sha256_json
from deploy_ci_cloud_agentv3.models.pending_action import PendingAction, compute_pending_action_fingerprint
from deploy_ci_cloud_agentv3.models.proposal import ProposalResult
from deploy_ci_cloud_agentv3.services.artifacts import ArtifactStore


class PendingActionFactory:
    """Runtime authority that enriches a safe Proposal with frozen safety fields."""

    def __init__(self, runtime_mcp: Any, artifacts: ArtifactStore | None = None) -> None:
        self.runtime_mcp = runtime_mcp
        self.artifacts = artifacts

    async def build(self, proposal: ProposalResult) -> PendingAction:
        task_name = str(proposal.args.get("task_name") or "")
        artifact_payload = None
        frozen_args = dict(proposal.args)
        precondition_target = task_name
        before: dict[str, Any] | None = None
        action_precondition: dict[str, Any] = {}

        if proposal.action == "submit_task":
            artifact_id = str(proposal.args["artifact_id"])
            if self.artifacts is not None:
                artifact = self.artifacts.get(artifact_id)
            else:
                from deploy_ci_cloud_agentv3.models.artifact import PreparedArtifact

                artifact = PreparedArtifact.model_validate(
                    await self.runtime_mcp.call_tool("get_prepared_artifact", {"artifact_id": artifact_id})
                )
            artifact_payload = artifact.model_dump(mode="json")
            frozen_args = {
                "artifact_id": artifact.artifact_id,
                "task_prefix": artifact.task_prefix,
                "config": artifact.config,
            }
            precondition_target = ""

        # Legacy resume(datasets=None) resolves failed datasets at mutation time.
        # V3 resolves that dynamic scope exactly once before review and reuses the
        # same observation as the verification baseline. This removes the race
        # between "target resolution" and "before snapshot".
        if proposal.action == "resume_task" and proposal.args.get("datasets") is None:
            resolution_snapshot = await self._snapshot(task_name, None)
            self._require_resume_snapshot(resolution_snapshot, phase="target resolution")
            failed = self._latest_failed_datasets(resolution_snapshot)
            if not failed:
                raise ValueError("resume_task has no currently failed datasets to freeze for review")
            frozen_args["datasets"] = failed
            before = resolution_snapshot
            action_precondition = {
                "kind": "resume_latest_state_failed",
                "datasets": failed,
                "before_sha256": sha256_json(before),
            }

        precondition = await self.runtime_mcp.call_tool(
            "capture_write_precondition", {"task_name": precondition_target}
        )
        if before is None:
            before = await self._before_snapshot(proposal.action, frozen_args)

        # Explicit resume targets are not required to be failed (the backend allows
        # an explicit re-run), but their before snapshot is still safety-critical
        # because post-write verification compares run identity. Bind it to approval.
        if proposal.action == "resume_task" and not action_precondition:
            action_precondition = {
                "kind": "resume_explicit_targets",
                "datasets": list(frozen_args.get("datasets") or []),
                "before_sha256": sha256_json(before),
            }

        fingerprint = compute_pending_action_fingerprint(
            action=proposal.action,
            args=frozen_args,
            artifact=artifact_payload,
            precondition=precondition,
            action_precondition=action_precondition,
        )
        return PendingAction(
            proposal_id=f"proposal_{uuid.uuid4().hex}",
            action=proposal.action,
            args=frozen_args,
            reason=proposal.reason,
            expected_effect=proposal.expected_effect,
            before=before,
            artifact=artifact_payload,
            precondition=precondition,
            action_precondition=action_precondition,
            fingerprint=fingerprint,
        )

    async def rebuild_from_edit(self, current: PendingAction, edited_args: dict[str, Any]) -> PendingAction:
        """Edits create a new proposal/fingerprint; the old approval can never authorize it."""
        proposal_args = dict(edited_args)
        if current.action == "submit_task" and "artifact_id" not in proposal_args:
            proposal_args["artifact_id"] = current.args.get("artifact_id")
        return await self.build(
            ProposalResult(
                action=current.action,
                args=proposal_args,
                reason=current.reason,
                expected_effect=current.expected_effect,
            )
        )

    async def _snapshot(self, task_name: str, datasets: list[str] | None) -> dict[str, Any]:
        return await self.runtime_mcp.call_tool(
            "get_action_verification_snapshot",
            {"task_name": task_name, "datasets": datasets},
        )

    async def _before_snapshot(self, action: str, args: dict[str, Any]) -> dict[str, Any]:
        if action == "submit_task":
            return {"task_prefix": args.get("task_prefix"), "artifact_id": args.get("artifact_id")}
        task_name = str(args.get("task_name") or "")
        if action == "resume_task":
            # Resume verification requires Airflow run identity. A task-detail fallback
            # would turn observation failure into an empty baseline and can create a
            # false success, so resume must fail closed here.
            snapshot = await self._snapshot(task_name, args.get("datasets"))
            self._require_resume_snapshot(snapshot, phase="before snapshot")
            return snapshot
        try:
            return await self._snapshot(task_name, args.get("datasets"))
        except Exception:
            return await self.runtime_mcp.call_tool("get_task_detail", {"task_name": task_name})

    @staticmethod
    def _require_resume_snapshot(snapshot: dict[str, Any], *, phase: str) -> None:
        errors = dict(snapshot.get("errors") or {})
        if errors:
            raise RuntimeError(f"cannot create safe resume PendingAction during {phase}: {errors}")
        if snapshot.get("task_exists") is not True:
            raise RuntimeError(f"cannot create safe resume PendingAction during {phase}: task not found")
        if not isinstance(snapshot.get("airflow_runs"), list):
            raise RuntimeError(f"cannot create safe resume PendingAction during {phase}: Airflow run evidence missing")

    @staticmethod
    def _latest_failed_datasets(snapshot: dict[str, Any]) -> list[str]:
        """Return datasets whose *latest observed* DagRun is failed.

        Snapshot runs are latest-first in the platform adapter. Looking only for any
        historical failed run would incorrectly classify a dataset that has since
        progressed to running/success as resumable.
        """
        latest: dict[str, str] = {}
        ordered: list[str] = []
        for run in snapshot.get("airflow_runs") or []:
            dataset = str(run.get("dataset_name") or "")
            if not dataset or dataset in latest:
                continue
            latest[dataset] = str(run.get("state") or "").lower()
            ordered.append(dataset)
        return [dataset for dataset in ordered if latest.get(dataset) == "failed"]
