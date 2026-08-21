"""Deterministic post-action verification for user goals.

Action verification answers whether a mutation passed its operation contract.
This module answers the narrower user-goal question for ``resume_task`` using a
fresh read-only snapshot.  It never executes a write Tool and never consults an
LLM self-report.
"""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, Field

from .models import ToolCallSpec, ToolObservation
from .verification import ACTIVE_RUN_STATES, ActionVerificationResult, VerificationCheck


RESUME_ACCEPTED_STATES = frozenset(ACTIVE_RUN_STATES | {"success"})
RESUME_FAILED_STATES = frozenset({"failed"})


class GoalVerificationResult(BaseModel):
    action: str
    task_name: str
    status: str = Field(pattern="^(satisfied|in_progress|failed|inconclusive)$")
    attempts: int = 1
    checks: list[VerificationCheck] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)

    @property
    def satisfied(self) -> bool:
        return self.status == "satisfied"


class GoalVerifier:
    """Verify the user outcome after a successfully verified resume mutation."""

    def __init__(self, tool_client, attempts: int = 1, interval_sec: float = 0.0):
        self.tool_client = tool_client
        self.attempts = max(1, int(attempts))
        self.interval_sec = max(0.0, float(interval_sec))

    @staticmethod
    def _check(name: str, passed: bool, expected=None, actual=None, evidence: str = "") -> VerificationCheck:
        return VerificationCheck(name=name, passed=bool(passed), expected=expected, actual=actual, evidence=evidence)

    @staticmethod
    def _runs(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
        return [item for item in snapshot.get("airflow_runs", []) if isinstance(item, dict)]

    @staticmethod
    def _run_ids(snapshot: dict[str, Any]) -> set[str]:
        return {str(item.get("run_id") or "") for item in GoalVerifier._runs(snapshot) if item.get("run_id")}

    @staticmethod
    def _failed_datasets(snapshot: dict[str, Any]) -> list[str]:
        result: list[str] = []
        for item in GoalVerifier._runs(snapshot):
            if str(item.get("state") or "").lower() != "failed":
                continue
            dataset = str(item.get("dataset_name") or "")
            if dataset and dataset not in result:
                result.append(dataset)
        return result

    async def _snapshot(self, task_name: str, datasets: list[str]) -> tuple[dict[str, Any] | None, str | None]:
        try:
            result = await self.tool_client.execute([
                ToolCallSpec(
                    name="get_action_verification_snapshot",
                    arguments={"task_name": task_name, "datasets": datasets, "airflow_limit": 200},
                )
            ])
        except Exception as exc:
            return None, str(exc)
        if not result:
            return None, "goal verification snapshot returned no observation"
        observation = result[0]
        if not observation.ok or not isinstance(observation.data, dict):
            return None, observation.error or "goal verification snapshot failed"
        return observation.data, None

    def _evaluate(
        self,
        *,
        task_name: str,
        arguments: dict[str, Any],
        baseline: dict[str, Any],
        snapshot: dict[str, Any],
        attempts: int,
    ) -> GoalVerificationResult:
        checks: list[VerificationCheck] = []
        observed_task = str(snapshot.get("task_name") or "").strip()
        identity_ok = observed_task == task_name
        checks.append(self._check("target_identity", identity_ok, task_name, observed_task))
        if not identity_ok:
            return GoalVerificationResult(
                action="resume_task",
                task_name=task_name,
                status="inconclusive",
                attempts=attempts,
                checks=checks,
                errors=["post-action evidence task_name did not exactly match the resume target"],
                evidence={"observed_task": observed_task},
            )

        task_exists = snapshot.get("task_exists")
        checks.append(self._check("task_still_exists", task_exists is True, True, task_exists))
        if task_exists is False:
            return GoalVerificationResult(
                action="resume_task",
                task_name=task_name,
                status="failed",
                attempts=attempts,
                checks=checks,
                errors=["resume target task no longer exists after the mutation"],
                evidence={"task_name": observed_task, "task_exists": task_exists},
            )
        if task_exists is not True:
            return GoalVerificationResult(
                action="resume_task",
                task_name=task_name,
                status="inconclusive",
                attempts=attempts,
                checks=checks,
                errors=["post-action task existence was not deterministically observed"],
                evidence={"task_name": observed_task, "task_exists": task_exists},
            )

        errors = snapshot.get("errors")
        if errors:
            return GoalVerificationResult(
                action="resume_task",
                task_name=task_name,
                status="inconclusive",
                attempts=attempts,
                checks=checks,
                errors=["post-action read evidence reported backend errors"],
                evidence={"task_name": observed_task, "task_exists": task_exists, "errors_present": True},
            )

        if "airflow_runs" not in snapshot:
            return GoalVerificationResult(
                action="resume_task",
                task_name=task_name,
                status="inconclusive",
                attempts=attempts,
                checks=checks,
                errors=["post-action execution evidence was unavailable"],
                evidence={"task_name": observed_task, "task_exists": task_exists},
            )

        selected = [str(item) for item in arguments.get("datasets") or [] if str(item)]
        if not selected:
            selected = self._failed_datasets(baseline)
        before_ids = self._run_ids(baseline)
        new_runs = [
            item
            for item in self._runs(snapshot)
            if str(item.get("run_id") or "")
            and str(item.get("run_id")) not in before_ids
            and (not selected or str(item.get("dataset_name") or "") in selected)
        ]
        new_summary = [
            {
                "run_id": str(item.get("run_id") or ""),
                "dataset_name": str(item.get("dataset_name") or ""),
                "state": str(item.get("state") or "").lower(),
            }
            for item in new_runs
        ]
        checks.append(self._check("new_execution_created", bool(new_runs), True, bool(new_runs), str(new_summary)))
        evidence = {
            "task_name": observed_task,
            "task_exists": task_exists,
            "selected_datasets": selected,
            "new_executions": new_summary,
        }
        if not new_runs:
            return GoalVerificationResult(
                action="resume_task",
                task_name=task_name,
                status="in_progress",
                attempts=attempts,
                checks=checks,
                errors=["resume action was verified, but no new execution was observed"],
                evidence=evidence,
            )

        states = [str(item.get("state") or "").lower() for item in new_runs]
        failed = any(state in RESUME_FAILED_STATES for state in states)
        checks.append(self._check("new_execution_not_failed", not failed, True, states))
        if failed:
            return GoalVerificationResult(
                action="resume_task",
                task_name=task_name,
                status="failed",
                attempts=attempts,
                checks=checks,
                errors=["new resume execution entered failed state"],
                evidence=evidence,
            )

        accepted = all(state in RESUME_ACCEPTED_STATES for state in states)
        checks.append(self._check("new_execution_accepted", accepted, sorted(RESUME_ACCEPTED_STATES), states))
        if accepted:
            return GoalVerificationResult(
                action="resume_task",
                task_name=task_name,
                status="satisfied",
                attempts=attempts,
                checks=checks,
                evidence=evidence,
            )
        return GoalVerificationResult(
            action="resume_task",
            task_name=task_name,
            status="in_progress",
            attempts=attempts,
            checks=checks,
            errors=["new execution exists but its state is not yet an accepted resume state"],
            evidence=evidence,
        )

    async def verify_resume(
        self,
        *,
        arguments: dict[str, Any],
        action_verification: ActionVerificationResult,
        baseline: dict[str, Any] | None = None,
    ) -> GoalVerificationResult:
        task_name = str(arguments.get("task_name") or "").strip()
        if not action_verification.verified:
            return GoalVerificationResult(
                action="resume_task",
                task_name=task_name,
                status="inconclusive",
                errors=["resume Goal Verification requires verified Action Verification"],
            )
        if not task_name:
            return GoalVerificationResult(
                action="resume_task",
                task_name="",
                status="inconclusive",
                errors=["resume Goal Verification requires an exact task target"],
            )

        baseline = baseline or {}
        last: GoalVerificationResult | None = None
        datasets = [str(item) for item in arguments.get("datasets") or [] if str(item)]
        for attempt in range(1, self.attempts + 1):
            snapshot, error = await self._snapshot(task_name, datasets)
            if snapshot is None:
                last = GoalVerificationResult(
                    action="resume_task",
                    task_name=task_name,
                    status="inconclusive",
                    attempts=attempt,
                    errors=[error or "post-action evidence unavailable"],
                )
            else:
                last = self._evaluate(
                    task_name=task_name,
                    arguments=arguments,
                    baseline=baseline,
                    snapshot=snapshot,
                    attempts=attempt,
                )
                if last.status in {"satisfied", "failed"}:
                    return last
            if attempt < self.attempts and self.interval_sec:
                await asyncio.sleep(self.interval_sec)
        assert last is not None
        return last


__all__ = [
    "GoalVerificationResult",
    "GoalVerifier",
    "RESUME_ACCEPTED_STATES",
]
