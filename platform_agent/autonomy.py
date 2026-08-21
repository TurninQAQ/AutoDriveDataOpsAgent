"""Deterministic policy for the V1.7 bounded autonomous resume slice.

This module deliberately has no model/provider dependency.  A model may propose
``resume_task`` through the normal frozen write-action plan, but only this policy
can authorize an automatic mutation.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AutonomyMode(str, Enum):
    AUTO = "AUTO"
    HITL = "HITL"
    DENY = "DENY"


class AutonomyRisk(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    DESTRUCTIVE = "destructive"
    INVALID = "invalid"


class AutonomyCheck(BaseModel):
    name: str
    passed: bool
    expected: Any = None
    actual: Any = None


class AutonomyDecision(BaseModel):
    mode: AutonomyMode
    action: str
    risk_level: AutonomyRisk
    eligible: bool
    policy_version: str = "v1.7.0"
    reasons: list[str] = Field(default_factory=list)
    checks: list[AutonomyCheck] = Field(default_factory=list)
    budget: dict[str, Any] = Field(default_factory=dict)
    frozen_arguments: dict[str, Any] = Field(default_factory=dict)


def _clean_name(value: Any) -> str:
    return str(value or "").strip()


def _dataset_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    result: list[str] = []
    for item in value:
        name = _clean_name(item)
        if name and name not in result:
            result.append(name)
    return result


def _run_order_key(item: dict[str, Any], index: int) -> tuple[int, str, int]:
    """Return a deterministic newest-first ordering key.

    The Airflow read gateway currently returns runs newest first.  When a
    timestamp is preserved by a fixture or future read adapter, use it instead;
    otherwise the service's documented order is retained by using the original
    position as the tie-breaker.
    """

    for field in (
        "run_after",
        "start_date",
        "logical_date",
        "execution_date",
        "created_at",
        "queued_at",
    ):
        value = item.get(field)
        if value:
            return (1, str(value), 0)
    return (0, "", -index)


def latest_dataset_states(airflow_runs: Any) -> dict[str, str]:
    """Get the latest known state for each dataset from a snapshot.

    ``AirflowReadGateway.list_dag_runs`` sorts newest runs first.  Snapshot
    fixtures without timestamps therefore use their first occurrence as latest;
    timestamp-bearing records are ordered explicitly here.  Historical failed
    runs cannot make a dataset currently failed when a newer success exists.
    """

    rows = [item for item in airflow_runs or [] if isinstance(item, dict)]
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, item in enumerate(rows):
        dataset = _clean_name(item.get("dataset_name"))
        if not dataset:
            conf = item.get("conf")
            if isinstance(conf, dict):
                dataset = _clean_name(conf.get("dataset_name"))
        if dataset:
            grouped.setdefault(dataset, []).append((index, item))

    states: dict[str, str] = {}
    for dataset, items in grouped.items():
        # Explicit timestamps sort ascending and the last item is newest.  For
        # the current newest-first service order, the first item is newest.
        timestamped = any(_run_order_key(item, index)[0] for index, item in items)
        if timestamped:
            _, latest = max(items, key=lambda pair: _run_order_key(pair[1], pair[0]))
        else:
            latest = min(items, key=lambda pair: pair[0])[1]
        states[dataset] = _clean_name(latest.get("state")).lower()
    return states


class BoundedAutonomyPolicy:
    """Allow-list policy for one bounded autonomous resume mutation."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        max_actions_per_request: int = 1,
        max_resume_datasets: int = 3,
    ):
        self.enabled = bool(enabled)
        self.max_actions_per_request = max(1, int(max_actions_per_request))
        self.max_resume_datasets = max(1, int(max_resume_datasets))

    @staticmethod
    def _check(name: str, passed: bool, expected: Any = None, actual: Any = None) -> AutonomyCheck:
        return AutonomyCheck(name=name, passed=bool(passed), expected=expected, actual=actual)

    @staticmethod
    def _result(
        *,
        mode: AutonomyMode,
        action: str,
        risk: AutonomyRisk,
        reasons: list[str],
        checks: list[AutonomyCheck],
        budget: dict[str, Any],
        frozen_arguments: dict[str, Any] | None = None,
    ) -> AutonomyDecision:
        return AutonomyDecision(
            mode=mode,
            action=action,
            risk_level=risk,
            eligible=mode == AutonomyMode.AUTO,
            reasons=reasons,
            checks=checks,
            budget=budget,
            frozen_arguments=dict(frozen_arguments or {}),
        )

    def decide(
        self,
        *,
        action: str,
        arguments: dict[str, Any],
        precondition: dict[str, Any] | None,
        baseline: dict[str, Any] | None,
        auto_actions_used: int = 0,
    ) -> AutonomyDecision:
        """Return a deterministic authorization decision.

        The input baseline and precondition are captured before this decision.
        The returned ``frozen_arguments`` is the only dataset scope an AUTO
        executor may pass to the mutation tool.
        """

        args = dict(arguments or {})
        base = dict(baseline or {})
        pre = dict(precondition or {})
        budget = {
            "actions_used": int(auto_actions_used),
            "actions_limit": self.max_actions_per_request,
            "datasets_limit": self.max_resume_datasets,
        }
        checks: list[AutonomyCheck] = []

        if action != "resume_task":
            return self._result(
                mode=AutonomyMode.HITL,
                action=action,
                risk=AutonomyRisk.HIGH if action != "delete_task" else AutonomyRisk.DESTRUCTIVE,
                reasons=["action_not_in_auto_allowlist"],
                checks=[self._check("resume_only_allowlist", False, "resume_task", action)],
                budget=budget,
                frozen_arguments=args,
            )

        if not self.enabled:
            return self._result(
                mode=AutonomyMode.HITL,
                action=action,
                risk=AutonomyRisk.MEDIUM,
                reasons=["autonomy_disabled"],
                checks=[self._check("autonomy_enabled", False, True, False)],
                budget=budget,
                frozen_arguments=args,
            )

        if int(auto_actions_used) >= self.max_actions_per_request:
            return self._result(
                mode=AutonomyMode.HITL,
                action=action,
                risk=AutonomyRisk.MEDIUM,
                reasons=["autonomy_action_budget_exceeded"],
                checks=[self._check("within_action_budget", False, self.max_actions_per_request, auto_actions_used)],
                budget=budget,
                frozen_arguments=args,
            )

        target = _clean_name(args.get("task_name"))
        baseline_target = _clean_name(base.get("task_name"))
        precondition_target = _clean_name(pre.get("task_name"))
        checks.append(self._check("target_present", bool(target), True, target))
        checks.append(self._check("baseline_target_matches", bool(target) and baseline_target == target, target, baseline_target))
        checks.append(self._check("precondition_target_matches", bool(target) and precondition_target == target, target, precondition_target))
        if not target or baseline_target != target or precondition_target != target:
            return self._result(
                mode=AutonomyMode.DENY,
                action=action,
                risk=AutonomyRisk.INVALID,
                reasons=["target_missing_or_provenance_conflict"],
                checks=checks,
                budget=budget,
                frozen_arguments=args,
            )

        task_exists = base.get("task_exists") is True
        config_exists = base.get("config_file_exists") is True
        dag_file_exists = base.get("dag_file_exists") is True
        dag_exists = base.get("airflow_dag_exists") is True
        available = _dataset_list(base.get("available_datasets"))
        errors = base.get("errors") or {}
        runs_present = isinstance(base.get("airflow_runs"), list)
        precondition_valid = bool(
            pre.get("queue_sha256")
            and pre.get("task_config_sha256")
            and pre.get("task_exists") is True
        )
        checks.extend([
            self._check("task_exists", task_exists, True, base.get("task_exists")),
            self._check("config_exists", config_exists, True, base.get("config_file_exists")),
            self._check("dag_file_exists", dag_file_exists, True, base.get("dag_file_exists")),
            self._check("airflow_dag_exists", dag_exists, True, base.get("airflow_dag_exists")),
            self._check("available_datasets_known", bool(available), True, available),
            self._check("airflow_runs_observable", runs_present, True, type(base.get("airflow_runs")).__name__),
            self._check("baseline_errors_empty", not errors, {}, errors),
            self._check("precondition_captured", precondition_valid, True, pre),
        ])
        if not all(check.passed for check in checks):
            return self._result(
                mode=AutonomyMode.DENY,
                action=action,
                risk=AutonomyRisk.INVALID,
                reasons=["critical_current_state_evidence_unavailable"],
                checks=checks,
                budget=budget,
                frozen_arguments=args,
            )

        states = latest_dataset_states(base.get("airflow_runs"))
        currently_failed = sorted(name for name, state in states.items() if state == "failed")
        requested = _dataset_list(args.get("datasets"))
        explicit_datasets = bool(requested)
        selected = requested if explicit_datasets else currently_failed
        checks.append(self._check("currently_failed_datasets_available", bool(currently_failed), True, currently_failed))

        if explicit_datasets:
            unknown = sorted(set(requested) - set(available))
            non_failed = sorted(set(requested) - set(currently_failed))
            checks.extend([
                self._check("requested_datasets_known", not unknown, [], unknown),
                self._check("requested_datasets_currently_failed", not non_failed, requested, sorted(set(requested) & set(currently_failed))),
            ])
            if unknown:
                return self._result(
                    mode=AutonomyMode.DENY,
                    action=action,
                    risk=AutonomyRisk.INVALID,
                    reasons=["unknown_dataset"],
                    checks=checks,
                    budget=budget,
                    frozen_arguments=args,
                )
            if non_failed:
                return self._result(
                    mode=AutonomyMode.HITL,
                    action=action,
                    risk=AutonomyRisk.MEDIUM,
                    reasons=["requested_dataset_not_currently_failed"],
                    checks=checks,
                    budget=budget,
                    frozen_arguments=args,
                )
        elif not selected:
            return self._result(
                mode=AutonomyMode.DENY,
                action=action,
                risk=AutonomyRisk.INVALID,
                reasons=["no_currently_failed_dataset"],
                checks=checks,
                budget=budget,
                frozen_arguments=args,
            )

        unknown_selected = sorted(set(selected) - set(available))
        if unknown_selected:
            return self._result(
                mode=AutonomyMode.DENY,
                action=action,
                risk=AutonomyRisk.INVALID,
                reasons=["unknown_dataset"],
                checks=checks + [self._check("selected_datasets_known", False, available, unknown_selected)],
                budget=budget,
                frozen_arguments=args,
            )
        if len(selected) > self.max_resume_datasets:
            return self._result(
                mode=AutonomyMode.HITL,
                action=action,
                risk=AutonomyRisk.MEDIUM,
                reasons=["autonomy_dataset_budget_exceeded"],
                checks=checks + [self._check("within_dataset_budget", False, self.max_resume_datasets, len(selected))],
                budget=budget,
                frozen_arguments=args,
            )

        active_task = _clean_name(pre.get("active_task_name"))
        exclusive = base.get("task_exclusive") is True
        cross_task = exclusive and bool(active_task) and active_task != target
        checks.extend([
            self._check("no_cross_task_preemption", not cross_task, False, active_task if cross_task else ""),
            self._check("selected_datasets_currently_failed", set(selected).issubset(set(currently_failed)), sorted(selected), currently_failed),
        ])
        if cross_task:
            return self._result(
                mode=AutonomyMode.HITL,
                action=action,
                risk=AutonomyRisk.MEDIUM,
                reasons=["cross_task_preemption_possible"],
                checks=checks,
                budget=budget,
                frozen_arguments=args,
            )

        frozen = dict(args)
        # AUTO never delegates an empty dataset list to the mutation service.
        frozen["task_name"] = target
        frozen["datasets"] = list(selected)
        return self._result(
            mode=AutonomyMode.AUTO,
            action=action,
            risk=AutonomyRisk.LOW,
            reasons=[
                "resume_only",
                "currently_failed_datasets_only",
                "frozen_dataset_scope",
                "no_cross_task_preemption",
                "within_autonomy_budget",
            ],
            checks=checks + [self._check("within_dataset_budget", True, self.max_resume_datasets, len(selected))],
            budget=budget,
            frozen_arguments=frozen,
        )


# Short compatibility name for callers that refer to the V1.7 policy simply as
# ``AutonomyPolicy``.
AutonomyPolicy = BoundedAutonomyPolicy
