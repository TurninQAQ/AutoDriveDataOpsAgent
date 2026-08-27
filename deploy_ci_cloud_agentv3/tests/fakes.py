from __future__ import annotations

from copy import deepcopy
from typing import Any


class FakeFacade:
    def __init__(self) -> None:
        self.mutations: list[tuple[str, dict[str, Any]]] = []
        self.precondition = {"queue_sha256": "q1", "task_name": "task_a", "task_config_sha256": "c1", "task_exists": True, "active_task_name": ""}
        self.priority = 3
        self.exists = True
        self.configs: dict[str, dict[str, Any]] = {"task_a": {"priority": 3, "datasets": []}}

    def get_task_detail(self, task_name: str, include_airflow_runs: bool = True, run_limit: int = 20):
        return {"task_name": task_name, "exists": self.exists, "priority": self.priority, "state": "QUEUED"}

    def get_gpu_pool(self, cleanup_dead: bool = True): return {"devices": [], "reservations": []}
    def get_queue_state(self, task_name: str = ""): return {"task_name": task_name, "position": 1} if task_name else {"queue": []}
    def diagnose_task(self, task_name: str, dataset_name: str = ""): return {"task_name": task_name, "diagnosis": "ok"}
    def search_knowledge(self, query: str, top_k: int = 5): return {"query": query, "results": []}
    def validate_task_spec(self, task_prefix: str, config: dict[str, Any]): return {"valid": True, "task_prefix": task_prefix, "config": deepcopy(config)}

    def get_write_precondition(self, task_name: str = ""):
        value = dict(self.precondition); value["task_name"] = task_name; return value

    def get_action_verification_snapshot(self, task_name: str, datasets=None, airflow_limit: int = 100):
        exists = task_name in self.configs if task_name != "task_a" else self.exists
        return {
            "task_name": task_name, "task_exists": exists,
            "config_file_exists": exists, "dag_file_exists": exists,
            "priority": self.priority if task_name == "task_a" else None,
            "queue": {"location": "queued", "position": 1, "entry": {"task_name": task_name}} if exists else {"location": "not_found", "position": -1, "entry": None},
            "containers": [], "gpu_reservations": [],
            "airflow_dag_exists": True if exists else False,
            "airflow_runs": [{"state": "queued", "dataset_name": "d1"}] if exists else [],
            "errors": {},
        }

    def get_task_config_for_verification(self, task_name: str):
        return {"task_name": task_name, "config": deepcopy(self.configs[task_name])}

    def set_task_priority(self, task_name: str, priority: int, precondition: dict[str, Any]):
        self.mutations.append(("set_task_priority", {"task_name": task_name, "priority": priority})); self.priority = priority
        self.configs.setdefault(task_name, {})["priority"] = priority
        return {"ok": True, "action": "set_task_priority", "result": {"task_name": task_name, "priority": priority}}

    def resume_task(self, task_name: str, datasets, precondition: dict[str, Any]):
        self.mutations.append(("resume_task", {"task_name": task_name, "datasets": datasets})); return {"ok": True}

    def stop_task(self, task_name: str, datasets, precondition: dict[str, Any]):
        self.mutations.append(("stop_task", {"task_name": task_name, "datasets": datasets})); return {"ok": True}

    def delete_task(self, task_name: str, precondition: dict[str, Any]):
        self.mutations.append(("delete_task", {"task_name": task_name})); self.exists = False; self.configs.pop(task_name, None); return {"ok": True}

    def submit_task(self, task_prefix: str, config: dict[str, Any], precondition: dict[str, Any]):
        task_name = f"{task_prefix}_generated"
        self.mutations.append(("submit_task", {"task_prefix": task_prefix, "config": deepcopy(config)}))
        self.configs[task_name] = deepcopy(config)
        return {"ok": True, "result": {"task_name": task_name}}
