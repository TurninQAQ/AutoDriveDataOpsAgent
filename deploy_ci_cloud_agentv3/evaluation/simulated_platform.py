from __future__ import annotations

import copy
import itertools
from typing import Any

from deploy_ci_cloud_agentv3.models.common import sha256_json


class BenchmarkPlatform:
    """Isolated deterministic platform used only by benchmark baselines.

    The platform is stateful and observable. Fault injection changes transport or
    state behavior, while benchmark outcomes are derived from actual attempts,
    business effects and final platform state rather than from the fault label.
    """

    def __init__(self) -> None:
        self.tasks: dict[str, dict[str, Any]] = {
            "task_A": {
                "priority": 1,
                "config": {"priority": 1, "datasets": [{"dataset_name": "A"}]},
            },
            "task_B": {
                "priority": 1,
                "config": {"priority": 1, "datasets": [{"dataset_name": "A"}]},
            },
        }
        self.queue = {"task_A", "task_B"}
        self.containers = {"task_A": True, "task_B": True}
        self.gpu = {"task_A": True, "task_B": True}
        self.runs: dict[str, list[dict[str, Any]]] = {
            "task_A": [
                {"run_id": "task_A-old-failed", "dataset_name": "A", "state": "failed"}
            ],
            "task_B": [
                {"run_id": "task_B-old-failed", "dataset_name": "A", "state": "failed"}
            ],
        }
        self._run_seq = itertools.count(1)
        self.precondition_epoch = 0
        self.mutation_attempts: list[dict[str, Any]] = []
        self.mutation_effects: list[dict[str, Any]] = []
        self.no_effect_after_ok = False
        self.drop_before_effect = False
        self.drop_after_effect = False
        self.fail_verification_snapshot_after_mutation = False

    @classmethod
    def from_fixture(cls, fixture: dict[str, Any] | None) -> "BenchmarkPlatform":
        platform = cls()
        fixture = copy.deepcopy(fixture or {})
        if not fixture:
            return platform

        if "tasks" in fixture:
            raw_tasks = fixture.get("tasks") or {}
            for name, value in raw_tasks.items():
                norm = cls.normalize_task(name)
                task = copy.deepcopy(value or {})
                priority = int(task.get("priority", 1))
                config = copy.deepcopy(task.get("config") or {})
                config.setdefault("priority", priority)
                config.setdefault("datasets", [{"dataset_name": "A"}])
                platform.tasks[norm] = {"priority": priority, "config": config}
        for name in fixture.get("missing_tasks") or []:
            norm = cls.normalize_task(name)
            platform.tasks.pop(norm, None)
            platform.queue.discard(norm)
            platform.containers.pop(norm, None)
            platform.gpu.pop(norm, None)
            platform.runs.pop(norm, None)

        if "queue" in fixture:
            platform.queue = {cls.normalize_task(item) for item in (fixture.get("queue") or [])}
        if "containers" in fixture:
            platform.containers = {
                cls.normalize_task(name): bool(value)
                for name, value in (fixture.get("containers") or {}).items()
            }
        if "gpu" in fixture:
            platform.gpu = {
                cls.normalize_task(name): bool(value)
                for name, value in (fixture.get("gpu") or {}).items()
            }
        if "runs" in fixture:
            for name, rows in (fixture.get("runs") or {}).items():
                platform.runs[cls.normalize_task(name)] = copy.deepcopy(rows or [])
        if "precondition_epoch" in fixture:
            platform.precondition_epoch = int(fixture.get("precondition_epoch") or 0)
        return platform

    @staticmethod
    def normalize_task(task_name: str) -> str:
        value = str(task_name or "")
        for known in ("task_A", "task_B", "new_task"):
            if known.lower() == value.lower():
                return known
        return value

    def get_task_detail(
        self, task_name: str, include_airflow_runs: bool = True, run_limit: int = 20
    ):
        name = self.normalize_task(task_name)
        task = self.tasks.get(name)
        return {
            "task_name": name,
            "exists": task is not None,
            "priority": (task or {}).get("priority"),
            "state": "QUEUED" if name in self.queue else ("RUNNING" if task else "NOT_FOUND"),
            "airflow_runs": copy.deepcopy(self.runs.get(name, [])[:run_limit])
            if include_airflow_runs
            else [],
        }

    def get_gpu_pool(self, cleanup_dead: bool = True):
        return {
            "devices": [{"id": 0}],
            "reservations": [name for name, active in self.gpu.items() if active],
        }

    def get_queue_state(self, task_name: str = ""):
        if task_name:
            name = self.normalize_task(task_name)
            return {
                "task_name": name,
                "location": "queued" if name in self.queue else "not_found",
            }
        return {"queue": sorted(self.queue)}

    def diagnose_task(self, task_name: str, dataset_name: str = ""):
        name = self.normalize_task(task_name)
        return {
            "task_name": name,
            "dataset_name": dataset_name,
            "diagnosis": "latest run failed" if self.runs.get(name) else "no runs",
        }

    async def search_knowledge(self, query: str, top_k: int = 5):
        return {
            "query": query,
            "mode": "bm25",
            "results": [
                {
                    "chunk_id": "bench-runbook",
                    "source": "benchmark",
                    "title": "Runbook",
                    "text": "Inspect task, queue and runtime evidence.",
                }
            ][:top_k],
        }

    def validate_task_spec(self, task_prefix: str, config: dict[str, Any]):
        return {"valid": True, "task_prefix": task_prefix, "config": copy.deepcopy(config)}

    def get_write_precondition(self, task_name: str = ""):
        name = self.normalize_task(task_name)
        task = self.tasks.get(name) if name else None
        return {
            "queue_sha256": sha256_json(sorted(self.queue)),
            "task_name": name,
            "task_config_sha256": sha256_json((task or {}).get("config")) if task else "",
            "task_exists": bool(task) if name else False,
            "active_task_name": "",
            "benchmark_epoch": self.precondition_epoch,
        }

    def get_action_verification_snapshot(
        self, task_name: str, datasets=None, airflow_limit: int = 100
    ):
        if self.fail_verification_snapshot_after_mutation and self.mutation_attempts:
            raise RuntimeError("benchmark verification snapshot failure")
        name = self.normalize_task(task_name)
        task = self.tasks.get(name)
        selected = {str(item) for item in (datasets or []) if str(item)}
        runs = copy.deepcopy(self.runs.get(name, [])[:airflow_limit])
        if selected:
            runs = [
                run
                for run in runs
                if str(run.get("dataset_name") or "") in selected
            ]
        return {
            "task_name": name,
            "task_exists": task is not None,
            "config_file_exists": task is not None,
            "dag_file_exists": task is not None,
            "priority": (task or {}).get("priority"),
            "queue": {
                "location": "queued" if name in self.queue else "not_found",
                "entry": {"task_name": name} if name in self.queue else None,
            },
            "containers": (
                [{"task_name": name, "dataset_name": "A", "running": True}]
                if self.containers.get(name)
                else []
            ),
            "gpu_reservations": (
                [{"task_name": name, "dataset_name": "A", "gpu_id": 0}]
                if self.gpu.get(name)
                else []
            ),
            "airflow_dag_exists": task is not None,
            "airflow_runs": runs,
            "available_datasets": [
                str(item.get("dataset_name") or "")
                for item in ((task or {}).get("config") or {}).get("datasets", [])
                if str(item.get("dataset_name") or "")
            ],
            "errors": {},
        }

    def get_task_config_for_verification(self, task_name: str):
        name = self.normalize_task(task_name)
        return {"task_name": name, "config": copy.deepcopy(self.tasks[name]["config"])}

    def set_task_priority(self, task_name: str, priority: int, precondition: dict[str, Any]):
        name = self.normalize_task(task_name)
        self._attempt("set_task_priority", name, {"priority": priority})
        self._raise_before_effect()
        if not self.no_effect_after_ok:
            self.tasks[name]["priority"] = priority
            self.tasks[name]["config"]["priority"] = priority
            self._effect("set_task_priority", name)
        self._raise_after_effect()
        return {"ok": True, "result": {"task_name": name, "priority": priority}}

    def resume_task(self, task_name: str, datasets, precondition: dict[str, Any]):
        name = self.normalize_task(task_name)
        targets = list(datasets or ["A"])
        self._attempt("resume_task", name, {"datasets": targets})
        self._raise_before_effect()
        if not self.no_effect_after_ok:
            for dataset in targets:
                self.runs.setdefault(name, []).insert(
                    0,
                    {
                        "run_id": f"{name}-resume-{next(self._run_seq)}",
                        "dataset_name": dataset,
                        "state": "queued",
                    },
                )
            self.queue.add(name)
            self.containers[name] = False
            self.gpu[name] = False
            self._effect("resume_task", name)
        self._raise_after_effect()
        return {"ok": True, "result": {"task_name": name, "datasets": targets}}

    def stop_task(self, task_name: str, datasets, precondition: dict[str, Any]):
        name = self.normalize_task(task_name)
        targets = list(datasets or [])
        self._attempt("stop_task", name, {"datasets": targets})
        self._raise_before_effect()
        if not self.no_effect_after_ok:
            self.containers[name] = False
            self.gpu[name] = False
            for run in self.runs.get(name, []):
                if (
                    not targets or run.get("dataset_name") in targets
                ) and str(run.get("state") or "").lower() in {
                    "queued",
                    "scheduled",
                    "running",
                }:
                    run["state"] = "failed"
            if not targets:
                self.queue.discard(name)
            self._effect("stop_task", name)
        self._raise_after_effect()
        return {"ok": True, "result": {"task_name": name}}

    def delete_task(self, task_name: str, precondition: dict[str, Any]):
        name = self.normalize_task(task_name)
        self._attempt("delete_task", name, {})
        self._raise_before_effect()
        if not self.no_effect_after_ok:
            self.tasks.pop(name, None)
            self.queue.discard(name)
            self.containers.pop(name, None)
            self.gpu.pop(name, None)
            self.runs.pop(name, None)
            self._effect("delete_task", name)
        self._raise_after_effect()
        return {"ok": True, "result": {"task_name": name}}

    def submit_task(self, task_prefix: str, config: dict[str, Any], precondition: dict[str, Any]):
        name = "new_task"
        self._attempt("submit_task", name, {"task_prefix": task_prefix})
        self._raise_before_effect()
        if not self.no_effect_after_ok:
            self.tasks[name] = {
                "priority": int(config.get("priority") or 1),
                "config": copy.deepcopy(config),
            }
            self.queue.add(name)
            self.containers[name] = False
            self.gpu[name] = False
            self.runs[name] = []
            self._effect("submit_task", name)
        self._raise_after_effect()
        return {"ok": True, "result": {"task_name": name}}

    def bump_precondition(self) -> None:
        self.precondition_epoch += 1

    def make_resume_target_stale(self, task_name: str = "task_A") -> None:
        name = self.normalize_task(task_name)
        self.runs.setdefault(name, []).insert(
            0,
            {
                "run_id": f"{name}-external-running",
                "dataset_name": "A",
                "state": "running",
            },
        )

    def state_matches(self, expected: dict[str, Any] | None) -> bool:
        """Evaluate explicit benchmark ground truth against real platform state."""
        expected = expected or {}
        if not expected:
            return True
        target = self.normalize_task(str(expected.get("task_name") or ""))
        if "exists" in expected:
            if (target in self.tasks) is not bool(expected["exists"]):
                return False
        if "priority" in expected:
            if self.tasks.get(target, {}).get("priority") != int(expected["priority"]):
                return False
        if "queue_location" in expected:
            actual = "queued" if target in self.queue else "not_found"
            if actual != str(expected["queue_location"]):
                return False
        if expected.get("stopped") is True:
            if self.containers.get(target, False) or self.gpu.get(target, False):
                return False
        if expected.get("resumed_datasets"):
            required = {str(item) for item in expected["resumed_datasets"]}
            actual = {
                str(run.get("dataset_name") or "")
                for run in self.runs.get(target, [])
                if "resume-" in str(run.get("run_id") or "")
            }
            if not required.issubset(actual):
                return False
        return True

    def effect_matches(
        self,
        action: str,
        target: str | None,
        expected_final_state: dict[str, Any] | None = None,
        expected_args: dict[str, Any] | None = None,
    ) -> bool:
        """Business-effect predicate used by benchmark metrics.

        A mutation attempt or transport OK is not sufficient. The observable state
        must match the requested action/target contract.
        """
        if expected_final_state:
            return self.state_matches(expected_final_state)
        if not target:
            return not action or bool(self.mutation_effects)
        norm = self.normalize_task(target)
        args = expected_args or {}
        if action == "set_task_priority":
            if "priority" not in args:
                return False
            return self.tasks.get(norm, {}).get("priority") == int(args["priority"])
        if action == "delete_task":
            return norm not in self.tasks
        if action == "submit_task":
            return "new_task" in self.tasks
        if action == "stop_task":
            targets = list(args.get("datasets") or [])
            if targets:
                return not self.containers.get(norm, False) and not self.gpu.get(norm, False)
            return (
                not self.containers.get(norm, False)
                and not self.gpu.get(norm, False)
                and norm not in self.queue
            )
        if action == "resume_task":
            targets = {str(item) for item in (args.get("datasets") or ["A"])}
            resumed = {
                str(run.get("dataset_name") or "")
                for run in self.runs.get(norm, [])
                if "resume-" in str(run.get("run_id") or "")
            }
            return targets.issubset(resumed)
        return False

    def _attempt(self, action: str, target: str, args: dict[str, Any]) -> None:
        self.mutation_attempts.append(
            {"action": action, "target": target, "args": copy.deepcopy(args)}
        )

    def _effect(self, action: str, target: str) -> None:
        self.mutation_effects.append({"action": action, "target": target})

    def _raise_before_effect(self) -> None:
        if self.drop_before_effect:
            raise ConnectionResetError("benchmark transport drop before effect")

    def _raise_after_effect(self) -> None:
        if self.drop_after_effect:
            raise ConnectionResetError("benchmark transport drop after committed effect")
