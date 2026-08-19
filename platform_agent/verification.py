from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, Field

from .models import ToolCallSpec

ACTIVE_RUN_STATES = {
    "queued", "running", "scheduled", "deferred", "restarting",
    "up_for_retry", "up_for_reschedule",
}


class VerificationCheck(BaseModel):
    name: str
    passed: bool
    expected: Any = None
    actual: Any = None
    evidence: str = ""


class ActionVerificationResult(BaseModel):
    action: str
    task_name: str
    status: str = Field(pattern="^(verified|failed|inconclusive)$")
    attempts: int = 1
    checks: list[VerificationCheck] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    snapshot: dict[str, Any] = Field(default_factory=dict)

    @property
    def verified(self) -> bool:
        return self.status == "verified"


class ActionVerifier:
    """Deterministic Observe-Again/Verify contracts for V0.8 write actions."""

    def __init__(self, tool_client, attempts: int = 5, interval_sec: float = 1.0):
        self.tool_client = tool_client
        self.attempts = max(1, int(attempts))
        self.interval_sec = max(0.0, float(interval_sec))

    @staticmethod
    def _payload_result(execution_result: dict[str, Any]) -> dict[str, Any]:
        result = execution_result.get("result")
        return result if isinstance(result, dict) else {}

    @classmethod
    def target_task_name(cls, action: str, arguments: dict[str, Any], execution_result: dict[str, Any]) -> str:
        if action == "submit_task":
            result = cls._payload_result(execution_result)
            return str(result.get("task_name") or execution_result.get("task_name") or "")
        return str(arguments.get("task_name") or "")

    async def _snapshot(self, task_name: str, datasets: list[str]) -> tuple[dict[str, Any] | None, str | None]:
        result = await self.tool_client.execute([
            ToolCallSpec(
                name="get_action_verification_snapshot",
                arguments={"task_name": task_name, "datasets": datasets, "airflow_limit": 200},
            )
        ])
        if not result:
            return None, "verification snapshot returned no observation"
        obs = result[0]
        if not obs.ok or not isinstance(obs.data, dict):
            return None, obs.error or "verification snapshot failed"
        return obs.data, None

    @staticmethod
    def _priority_value(snapshot: dict[str, Any]) -> int | None:
        raw = snapshot.get("priority")
        if isinstance(raw, dict):
            raw = raw.get("priority")
        try:
            return int(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _queue_location(snapshot: dict[str, Any]) -> str:
        return str((snapshot.get("queue") or {}).get("location") or "")

    @staticmethod
    def _queue_priority(snapshot: dict[str, Any]) -> int | None:
        raw = ((snapshot.get("queue") or {}).get("entry") or {}).get("priority")
        try:
            return int(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _runs(snapshot: dict[str, Any], datasets: list[str] | None = None) -> list[dict[str, Any]]:
        runs = [item for item in snapshot.get("airflow_runs") or [] if isinstance(item, dict)]
        selected = set(datasets or [])
        return [item for item in runs if not selected or str(item.get("dataset_name") or "") in selected]

    @staticmethod
    def _run_ids(snapshot: dict[str, Any], datasets: list[str] | None = None) -> set[str]:
        return {str(item.get("run_id") or "") for item in ActionVerifier._runs(snapshot, datasets) if item.get("run_id")}

    @staticmethod
    def _failed_datasets(snapshot: dict[str, Any]) -> list[str]:
        seen: list[str] = []
        for item in ActionVerifier._runs(snapshot):
            if str(item.get("state") or "").lower() != "failed":
                continue
            name = str(item.get("dataset_name") or "")
            if name and name not in seen:
                seen.append(name)
        return seen

    @staticmethod
    def _airflow_available(snapshot: dict[str, Any]) -> bool:
        errors = snapshot.get("errors") or {}
        return "airflow" not in errors or snapshot.get("airflow_dag_exists") is False

    @staticmethod
    def _check(name: str, passed: bool, expected=None, actual=None, evidence: str = "") -> VerificationCheck:
        return VerificationCheck(name=name, passed=bool(passed), expected=expected, actual=actual, evidence=evidence)

    def _verify_submit(self, arguments, execution_result, snapshot):
        config = arguments.get("config") or {}
        datasets = [str(item.get("dataset_name") or "") for item in config.get("datasets") or [] if isinstance(item, dict)]
        result = self._payload_result(execution_result)
        expected_triggered = int(result.get("triggered") or len(datasets) or 0)
        expected_priority = config.get("priority")
        checks = [
            self._check("task_config_exists", snapshot.get("task_exists") is True, True, snapshot.get("task_exists")),
            self._check("generated_dag_file_exists", snapshot.get("dag_file_exists") is True, True, snapshot.get("dag_file_exists")),
        ]
        if expected_priority is not None:
            checks.append(self._check("priority_persisted", self._priority_value(snapshot) == int(expected_priority), int(expected_priority), self._priority_value(snapshot)))
        if bool(config.get("task_exclusive", True)):
            checks.append(self._check("queue_registered", self._queue_location(snapshot) in {"active", "queued"}, "active|queued", self._queue_location(snapshot)))
        airflow_available = self._airflow_available(snapshot)
        if expected_triggered > 0:
            observed = {str(item.get("dataset_name") or "") for item in self._runs(snapshot)}
            checks.append(self._check("dag_runs_observable", set(datasets).issubset(observed), sorted(datasets), sorted(observed)))
        return checks, airflow_available

    def _verify_priority(self, arguments, snapshot):
        expected = int(arguments["priority"])
        checks = [
            self._check("task_still_exists", snapshot.get("task_exists") is True, True, snapshot.get("task_exists")),
            self._check("config_priority_updated", self._priority_value(snapshot) == expected, expected, self._priority_value(snapshot)),
        ]
        location = self._queue_location(snapshot)
        if location in {"active", "queued"}:
            checks.append(self._check("queue_priority_refreshed", self._queue_priority(snapshot) == expected, expected, self._queue_priority(snapshot)))
        return checks, True

    def _verify_stop(self, arguments, baseline, snapshot):
        datasets = list(arguments.get("datasets") or [])
        checks = [
            self._check("task_config_preserved", snapshot.get("task_exists") is True, True, snapshot.get("task_exists")),
            self._check("containers_reclaimed", len(snapshot.get("containers") or []) == 0, 0, len(snapshot.get("containers") or [])),
            self._check("gpu_reservations_reclaimed", len(snapshot.get("gpu_reservations") or []) == 0, 0, len(snapshot.get("gpu_reservations") or [])),
        ]
        if not datasets:
            checks.append(self._check("queue_removed", self._queue_location(snapshot) == "not_found", "not_found", self._queue_location(snapshot)))
        airflow_available = self._airflow_available(snapshot)
        if airflow_available:
            baseline_active_ids = {
                str(item.get("run_id") or "") for item in self._runs(baseline, datasets)
                if str(item.get("state") or "").lower() in ACTIVE_RUN_STATES and item.get("run_id")
            }
            after = {str(item.get("run_id") or ""): str(item.get("state") or "").lower() for item in self._runs(snapshot, datasets)}
            still_active = [run_id for run_id in baseline_active_ids if after.get(run_id) in ACTIVE_RUN_STATES]
            checks.append(self._check("target_dagruns_not_active", not still_active, [], still_active))
        return checks, airflow_available

    def _verify_resume(self, arguments, baseline, snapshot):
        datasets = list(arguments.get("datasets") or []) or self._failed_datasets(baseline)
        checks = [self._check("task_config_exists", snapshot.get("task_exists") is True, True, snapshot.get("task_exists"))]
        airflow_available = self._airflow_available(snapshot)
        if not datasets:
            checks.append(self._check("resume_noop_no_failed_dataset", True, "no failed datasets", "no failed datasets"))
            return checks, airflow_available
        if airflow_available:
            before_ids = self._run_ids(baseline)
            new_runs = [item for item in self._runs(snapshot, datasets) if str(item.get("run_id") or "") not in before_ids]
            new_datasets = {str(item.get("dataset_name") or "") for item in new_runs}
            checks.append(self._check("new_dagruns_created", set(datasets).issubset(new_datasets), sorted(datasets), sorted(new_datasets)))
        if snapshot.get("task_exclusive") is True:
            checks.append(self._check("resumed_task_scheduled", self._queue_location(snapshot) in {"active", "queued"}, "active|queued", self._queue_location(snapshot)))
        return checks, airflow_available

    def _verify_delete(self, snapshot):
        checks = [
            self._check("task_config_removed", snapshot.get("task_exists") is False, False, snapshot.get("task_exists")),
            self._check("generated_dag_file_removed", snapshot.get("dag_file_exists") is False, False, snapshot.get("dag_file_exists")),
            self._check("queue_removed", self._queue_location(snapshot) == "not_found", "not_found", self._queue_location(snapshot)),
            self._check("containers_reclaimed", len(snapshot.get("containers") or []) == 0, 0, len(snapshot.get("containers") or [])),
            self._check("gpu_reservations_reclaimed", len(snapshot.get("gpu_reservations") or []) == 0, 0, len(snapshot.get("gpu_reservations") or [])),
        ]
        airflow_available = self._airflow_available(snapshot)
        if airflow_available:
            checks.append(self._check("airflow_dag_metadata_removed", snapshot.get("airflow_dag_exists") is False, False, snapshot.get("airflow_dag_exists")))
        return checks, airflow_available

    def _evaluate(self, action, arguments, execution_result, baseline, snapshot, attempts):
        if action == "submit_task":
            checks, complete = self._verify_submit(arguments, execution_result, snapshot)
        elif action == "set_task_priority":
            checks, complete = self._verify_priority(arguments, snapshot)
        elif action == "stop_task":
            checks, complete = self._verify_stop(arguments, baseline, snapshot)
        elif action == "resume_task":
            checks, complete = self._verify_resume(arguments, baseline, snapshot)
        elif action == "delete_task":
            checks, complete = self._verify_delete(snapshot)
        else:
            return ActionVerificationResult(action=action, task_name=self.target_task_name(action, arguments, execution_result), status="failed", attempts=attempts, errors=[f"No verification contract for action: {action}"], snapshot=snapshot)
        errors = [f"{k}: {v}" for k, v in (snapshot.get("errors") or {}).items()]
        failed = [item for item in checks if not item.passed]
        status = "failed" if failed else ("verified" if complete else "inconclusive")
        return ActionVerificationResult(action=action, task_name=self.target_task_name(action, arguments, execution_result), status=status, attempts=attempts, checks=checks, errors=errors, snapshot=snapshot)

    async def verify(self, *, action: str, arguments: dict[str, Any], execution_result: dict[str, Any], baseline: dict[str, Any] | None = None) -> ActionVerificationResult:
        task_name = self.target_task_name(action, arguments, execution_result)
        if not task_name:
            return ActionVerificationResult(action=action, task_name="", status="failed", errors=["Write action did not expose a target task_name for verification"])
        datasets = list(arguments.get("datasets") or [])
        last: ActionVerificationResult | None = None
        for attempt in range(1, self.attempts + 1):
            snapshot, error = await self._snapshot(task_name, datasets)
            if snapshot is None:
                last = ActionVerificationResult(action=action, task_name=task_name, status="inconclusive", attempts=attempt, errors=[error or "verification snapshot unavailable"])
            else:
                last = self._evaluate(action, arguments, execution_result, baseline or {}, snapshot, attempt)
            if last.verified:
                return last
            if attempt < self.attempts and self.interval_sec:
                await asyncio.sleep(self.interval_sec)
        assert last is not None
        return last
