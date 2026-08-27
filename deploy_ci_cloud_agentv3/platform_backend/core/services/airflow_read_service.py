from __future__ import annotations

import json
from typing import Any

from ..gateways.airflow_read import AirflowReadGateway


def _run_id(run: dict[str, Any]) -> str:
    return str(run.get("dag_run_id") or run.get("run_id") or "")


def _run_conf(run: dict[str, Any]) -> dict[str, Any]:
    conf = run.get("conf") or {}
    if isinstance(conf, str):
        try:
            conf = json.loads(conf)
        except json.JSONDecodeError:
            conf = {}
    return conf if isinstance(conf, dict) else {}


class AirflowReadService:
    def __init__(self, gateway: AirflowReadGateway):
        self.gateway = gateway

    def health(self) -> dict[str, Any]:
        return self.gateway.health()

    def runs(self, dag_id: str, limit: int = 50) -> list[dict[str, Any]]:
        return self.gateway.list_dag_runs(dag_id, limit=limit)

    def runs_for_dataset(
        self, dag_id: str, dataset_name: str | None, limit: int = 50
    ) -> list[dict[str, Any]]:
        runs = self.runs(dag_id, limit=limit)
        if not dataset_name:
            return runs
        return [run for run in runs if _run_conf(run).get("dataset_name") == dataset_name]

    def latest_run(
        self, dag_id: str, dataset_name: str | None = None
    ) -> dict[str, Any] | None:
        runs = self.runs_for_dataset(dag_id, dataset_name, limit=100)
        return runs[0] if runs else None

    def task_instances(self, dag_id: str, run_id: str) -> list[dict[str, Any]]:
        return self.gateway.list_task_instances(dag_id, run_id)

    def run_evidence(
        self, dag_id: str, dataset_name: str | None = None
    ) -> dict[str, Any]:
        run = self.latest_run(dag_id, dataset_name)
        if not run:
            return {"dag_id": dag_id, "dataset_name": dataset_name, "latest_run": None, "task_instances": []}
        run_id = _run_id(run)
        task_instances = self.task_instances(dag_id, run_id) if run_id else []
        return {
            "dag_id": dag_id,
            "dataset_name": dataset_name,
            "latest_run": run,
            "task_instances": task_instances,
        }

    def task_log(
        self,
        dag_id: str,
        run_id: str,
        task_id: str,
        try_number: int = 1,
        map_index: int = -1,
        tail_lines: int = 200,
    ) -> dict[str, Any]:
        text = self.gateway.get_task_log(
            dag_id, run_id, task_id, try_number=try_number, map_index=map_index
        )
        lines = text.splitlines()
        tail_lines = max(1, min(int(tail_lines), 2000))
        return {
            "dag_id": dag_id,
            "run_id": run_id,
            "task_id": task_id,
            "try_number": int(try_number),
            "map_index": int(map_index),
            "tail_lines": tail_lines,
            "log": "\n".join(lines[-tail_lines:]),
        }
