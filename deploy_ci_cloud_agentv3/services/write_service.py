from __future__ import annotations

import uuid
from typing import Any

from deploy_ci_cloud_agentv3.models.common import sha256_json
from deploy_ci_cloud_agentv3.models.pending_action import PendingAction
from deploy_ci_cloud_agentv3.models.write_result import WriteResult
from deploy_ci_cloud_agentv3.services.audit import AuditStore
from deploy_ci_cloud_agentv3.services.verification import VerificationService


class WriteService:
    """Deterministic approved-write core.

    Validation -> global precondition -> action-specific revalidation ->
    runtime-derived idempotency -> one mutation attempt -> action-specific read-back
    verification -> audit. A possibly-applied mutation is never blindly retried.
    """

    def __init__(self, runtime_mcp: Any, verification: VerificationService | None = None, audit: AuditStore | None = None) -> None:
        self.runtime_mcp = runtime_mcp
        self.verification = verification or VerificationService(runtime_mcp)
        self.audit = audit or AuditStore()
        self._results_by_key: dict[str, WriteResult] = {}
        self._attempted: set[str] = set()

    async def execute(self, action: PendingAction, approved_fingerprint: str) -> WriteResult:
        actual_fingerprint = self._validate_approval(action, approved_fingerprint)
        self._validate_bound_before_snapshot(action)
        idempotency_key = self._idempotency_key(actual_fingerprint)

        if idempotency_key in self._results_by_key:
            return self._results_by_key[idempotency_key]

        current_precondition = await self.runtime_mcp.call_tool(
            "capture_write_precondition", {"task_name": self._precondition_target(action)}
        )
        if current_precondition != action.precondition:
            return self._precondition_failed(
                action,
                idempotency_key,
                actual_fingerprint,
                after={"current_precondition": current_precondition},
                error="platform state changed after proposal/review",
                expected=action.precondition,
                actual=current_precondition,
            )

        admissible, evidence, reason = await self._revalidate_action_precondition(action)
        if not admissible:
            self.audit.append(
                "ActionPreconditionFailed",
                {
                    "action": action.action,
                    "fingerprint": actual_fingerprint,
                    "action_precondition": action.action_precondition,
                    "evidence": evidence,
                    "reason": reason,
                },
            )
            return self._precondition_failed(
                action,
                idempotency_key,
                actual_fingerprint,
                after={"action_revalidation": evidence},
                error=reason or "action-specific precondition changed after approval",
            )

        if idempotency_key in self._attempted:
            # This approval already crossed the side-effect boundary. Reconcile only.
            return await self._reconcile_unknown(action, idempotency_key, {}, "mutation attempt already consumed")
        self._attempted.add(idempotency_key)

        call_args = self._runtime_arguments(action)
        self.audit.append(
            "WriteAttempt",
            {
                "action": action.action,
                "fingerprint": actual_fingerprint,
                "idempotency_key": idempotency_key,
            },
        )
        try:
            raw = await self.runtime_mcp.call_tool(action.action, call_args)
        except Exception as exc:
            # The request may have reached the server. Never retry mutation here.
            return await self._reconcile_unknown(
                action, idempotency_key, {}, f"{type(exc).__name__}: {exc}"
            )

        raw_dict = raw if isinstance(raw, dict) else {"result": raw}
        try:
            verified, after = await self.verification.verify(
                action.action, action.args, raw_dict, before=action.before
            )
        except Exception as exc:
            return self._remember(
                idempotency_key,
                WriteResult(
                    id=f"write_{uuid.uuid4().hex}",
                    action=action.action,
                    status="VERIFICATION_FAILED",
                    verified=False,
                    before=action.before,
                    after={},
                    raw_result=raw_dict,
                    error=f"verification read failed: {type(exc).__name__}: {exc}",
                ),
            )

        status = "VERIFIED" if verified else "VERIFICATION_FAILED"
        self.audit.append(
            "VerificationResult",
            {
                "action": action.action,
                "fingerprint": actual_fingerprint,
                "verified": verified,
                "after": after,
            },
        )
        return self._remember(
            idempotency_key,
            WriteResult(
                id=f"write_{uuid.uuid4().hex}",
                action=action.action,
                status=status,
                verified=verified,
                before=action.before,
                after=after,
                raw_result=raw_dict,
                error=None if verified else "post-write business state did not match expected effect",
            ),
        )

    async def _revalidate_action_precondition(
        self, action: PendingAction
    ) -> tuple[bool, dict[str, Any], str | None]:
        condition = dict(action.action_precondition or {})
        if action.action != "resume_task":
            return True, {}, None

        targets = [str(item) for item in action.args.get("datasets") or [] if str(item)]
        if not targets:
            return False, {}, "approved resume action has no frozen dataset targets"
        try:
            snapshot = await self.runtime_mcp.call_tool(
                "get_action_verification_snapshot",
                {"task_name": str(action.args.get("task_name") or ""), "datasets": targets},
            )
        except Exception as exc:
            return False, {"error": f"{type(exc).__name__}: {exc}"}, "resume revalidation observation failed"

        errors = dict(snapshot.get("errors") or {})
        if errors:
            return False, snapshot, f"resume revalidation observation failed: {errors}"
        if snapshot.get("task_exists") is not True:
            return False, snapshot, "resume target task no longer exists"

        available = snapshot.get("available_datasets")
        if isinstance(available, list) and available:
            missing = sorted(set(targets) - {str(item) for item in available})
            if missing:
                return False, snapshot, f"resume target datasets no longer exist: {missing}"

        # Only dynamically-resolved resume actions carry the semantic contract
        # "resume datasets that were failed at review time". Explicit target lists
        # are allowed by the backend as intentional re-runs and therefore do not get
        # this additional state restriction.
        if condition.get("kind") != "resume_latest_state_failed":
            return True, snapshot, None

        latest = self._latest_run_by_dataset(snapshot)
        stale: dict[str, Any] = {}
        for dataset in targets:
            run = latest.get(dataset)
            state = str((run or {}).get("state") or "").lower()
            if state != "failed":
                stale[dataset] = run or {"state": "missing"}
        if stale:
            return False, snapshot, f"approved failed-dataset resume condition is stale: {stale}"
        return True, snapshot, None

    async def _reconcile_unknown(
        self, action: PendingAction, idempotency_key: str, raw: dict[str, Any], error: str
    ) -> WriteResult:
        try:
            verified, after = await self.verification.verify(
                action.action, action.args, raw, before=action.before
            )
        except Exception as exc:
            verified, after = False, {"reconcile_error": f"{type(exc).__name__}: {exc}"}
        self.audit.append(
            "UnknownOutcome",
            {"action": action.action, "fingerprint": action.fingerprint, "error": error},
        )
        result = WriteResult(
            id=f"write_{uuid.uuid4().hex}",
            action=action.action,
            status="VERIFIED" if verified else "UNKNOWN_OUTCOME",
            verified=verified,
            before=action.before,
            after=after,
            raw_result=raw,
            error=None if verified else error,
        )
        return self._remember(idempotency_key, result)

    def _precondition_failed(
        self,
        action: PendingAction,
        idempotency_key: str,
        fingerprint: str,
        *,
        after: dict[str, Any],
        error: str,
        expected: dict[str, Any] | None = None,
        actual: dict[str, Any] | None = None,
    ) -> WriteResult:
        payload: dict[str, Any] = {
            "action": action.action,
            "fingerprint": fingerprint,
            "error": error,
        }
        if expected is not None:
            payload["expected"] = expected
        if actual is not None:
            payload["actual"] = actual
        self.audit.append("PreconditionFailed", payload)
        return self._remember(
            idempotency_key,
            WriteResult(
                id=f"write_{uuid.uuid4().hex}",
                action=action.action,
                status="PRECONDITION_FAILED",
                verified=False,
                before=action.before,
                after=after,
                error=error,
            ),
        )

    def _remember(self, idempotency_key: str, result: WriteResult) -> WriteResult:
        self._results_by_key[idempotency_key] = result
        self.audit.append("WriteResult", result.model_dump(mode="json"))
        return result

    @staticmethod
    def _validate_approval(action: PendingAction, approved_fingerprint: str) -> str:
        actual_fingerprint = action.recompute_fingerprint()
        if actual_fingerprint != action.fingerprint:
            raise PermissionError("PendingAction was modified after it was frozen")
        if not approved_fingerprint or approved_fingerprint != actual_fingerprint:
            raise PermissionError("approved fingerprint does not match current frozen PendingAction")
        return actual_fingerprint

    @staticmethod
    def _validate_bound_before_snapshot(action: PendingAction) -> None:
        expected = str((action.action_precondition or {}).get("before_sha256") or "")
        if expected and sha256_json(action.before) != expected:
            raise PermissionError("PendingAction before snapshot was modified after it was frozen")

    @staticmethod
    def _latest_run_by_dataset(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        # Platform snapshots are latest-first. Keep the first run per dataset.
        for run in snapshot.get("airflow_runs") or []:
            dataset = str(run.get("dataset_name") or "")
            if dataset and dataset not in latest:
                latest[dataset] = run
        return latest

    @staticmethod
    def _idempotency_key(actual_fingerprint: str) -> str:
        return f"write_{actual_fingerprint}"

    @staticmethod
    def _precondition_target(action: PendingAction) -> str:
        return "" if action.action == "submit_task" else str(action.args.get("task_name") or "")

    @staticmethod
    def _runtime_arguments(action: PendingAction) -> dict[str, Any]:
        if action.action == "submit_task":
            return {
                "task_prefix": action.args["task_prefix"],
                "config": action.args["config"],
                "precondition": action.precondition,
            }
        args = dict(action.args)
        args["precondition"] = action.precondition
        return args
