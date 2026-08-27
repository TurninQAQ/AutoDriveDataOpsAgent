from __future__ import annotations

from typing import Any

from deploy_ci_cloud_agentv3.models.common import sha256_json


_ACTIVE_AIRFLOW_STATES = {"queued", "scheduled", "running", "up_for_retry", "deferred"}
_RESUME_PROGRESS_STATES = {"queued", "scheduled", "running", "success"}


class VerificationService:
    """Fail-closed action-specific read-back contracts.

    Verification proves the business effect of *this* approved action. In particular,
    resume compares before/after run identity instead of accepting unrelated historical
    success/running runs as evidence.
    """

    def __init__(self, runtime_mcp: Any) -> None:
        self.runtime_mcp = runtime_mcp

    async def verify(
        self,
        action: str,
        args: dict[str, Any],
        raw_result: dict[str, Any],
        *,
        before: dict[str, Any] | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        target = str(args.get("task_name") or "")
        datasets = args.get("datasets")
        if action == "submit_task":
            target = self._submitted_task_name(raw_result)
            if not target:
                return False, {"reason": "generated task identity missing from mutation result"}

        snapshot = await self.runtime_mcp.call_tool(
            "get_action_verification_snapshot", {"task_name": target, "datasets": datasets}
        )
        critical_errors = self._critical_errors(action, snapshot)
        if critical_errors:
            return False, {**snapshot, "verification_errors": critical_errors}

        if action == "set_task_priority":
            ok = snapshot.get("task_exists") is True and snapshot.get("priority") == args.get("priority")
        elif action == "delete_task":
            ok = self._verify_delete(snapshot)
        elif action == "submit_task":
            ok = await self._verify_submit(target, args, snapshot)
        elif action == "stop_task":
            ok = self._verify_stop(snapshot, datasets)
        elif action == "resume_task":
            ok = self._verify_resume(before or {}, snapshot, datasets)
        else:
            ok = False
        return ok, snapshot

    @staticmethod
    def _verify_delete(snapshot: dict[str, Any]) -> bool:
        if snapshot.get("task_exists") is not False:
            return False
        if snapshot.get("config_file_exists") is not False or snapshot.get("dag_file_exists") is not False:
            return False
        if snapshot.get("airflow_dag_exists") is not False:
            return False
        if snapshot.get("containers") or snapshot.get("gpu_reservations"):
            return False
        queue = snapshot.get("queue") or {}
        if not isinstance(queue, dict) or queue.get("location") != "not_found":
            return False
        for run in snapshot.get("airflow_runs") or []:
            if str(run.get("state") or "").lower() in _ACTIVE_AIRFLOW_STATES:
                return False
        return True

    async def _verify_submit(self, target: str, args: dict[str, Any], snapshot: dict[str, Any]) -> bool:
        if snapshot.get("task_exists") is not True:
            return False
        if snapshot.get("config_file_exists") is not True or snapshot.get("dag_file_exists") is not True:
            return False
        if snapshot.get("airflow_dag_exists") is False:
            return False
        try:
            actual = await self.runtime_mcp.call_tool("get_task_config_for_verification", {"task_name": target})
        except Exception:
            return False
        actual_config = actual.get("config") if isinstance(actual, dict) else None
        expected_config = args.get("config")
        return isinstance(actual_config, dict) and sha256_json(actual_config) == sha256_json(expected_config)

    @staticmethod
    def _verify_stop(snapshot: dict[str, Any], datasets: list[str] | None) -> bool:
        selected = {str(item) for item in (datasets or []) if str(item)}
        whole_task = not selected

        # Snapshot collection already filters Docker/GPU evidence by selected datasets
        # when a partial stop was requested.
        if any(bool(c.get("running")) for c in snapshot.get("containers") or []):
            return False
        if snapshot.get("gpu_reservations"):
            return False

        for run in snapshot.get("airflow_runs") or []:
            dataset = str(run.get("dataset_name") or "")
            if selected and dataset not in selected:
                continue
            if str(run.get("state") or "").lower() in _ACTIVE_AIRFLOW_STATES:
                return False

        if whole_task:
            queue = snapshot.get("queue") or {}
            if not isinstance(queue, dict) or queue.get("location") != "not_found":
                return False
        return snapshot.get("task_exists") is True

    @staticmethod
    def _verify_resume(
        before: dict[str, Any], after: dict[str, Any], datasets: list[str] | None
    ) -> bool:
        targets = {str(item) for item in (datasets or []) if str(item)}
        if not targets or after.get("task_exists") is not True:
            return False

        before_ids = {
            str(run.get("run_id") or "")
            for run in before.get("airflow_runs") or []
            if str(run.get("run_id") or "")
        }
        progressed: set[str] = set()
        for run in after.get("airflow_runs") or []:
            run_id = str(run.get("run_id") or "")
            dataset = str(run.get("dataset_name") or "")
            state = str(run.get("state") or "").lower()
            if dataset not in targets or state not in _RESUME_PROGRESS_STATES:
                continue
            # This backend resumes by triggering a new DagRun. Historical runs, even
            # successful/running ones, cannot verify the current approved resume.
            if not run_id or run_id in before_ids:
                continue
            progressed.add(dataset)
        return progressed == targets

    @staticmethod
    def _critical_errors(action: str, snapshot: dict[str, Any]) -> dict[str, Any]:
        errors = dict(snapshot.get("errors") or {})
        if action == "delete_task" and snapshot.get("airflow_dag_exists") is False:
            text = str(errors.get("airflow") or "").lower()
            if any(token in text for token in ("404", "not found", "dagnotfound")):
                errors.pop("airflow", None)
        return errors

    @staticmethod
    def _submitted_task_name(raw_result: dict[str, Any]) -> str:
        result = raw_result.get("result") if isinstance(raw_result, dict) else None
        if isinstance(result, dict):
            nested = result.get("result")
            if isinstance(nested, dict) and nested.get("task_name"):
                return str(nested["task_name"])
            if result.get("task_name"):
                return str(result["task_name"])
        return ""
