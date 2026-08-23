from __future__ import annotations

import argparse
import contextlib
import io
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml


class LegacyMutationGateway:
    """Direct-Python compatibility bridge to the existing task runtime.

    Agent/MCP code never executes arbitrary shell commands. Until all legacy mutation
    orchestration is migrated into Platform Core, this gateway imports the existing
    task_manager module and invokes its Python functions with validated arguments.
    This preserves the production queue/preemption/recovery semantics in V0.7.
    """

    def __init__(self, settings, parse_timeout_sec: int | None = None):
        self.settings = settings
        self.parse_timeout_sec = int(parse_timeout_sec or os.environ.get("AIRFLOW_DAG_PARSE_TIMEOUT_SEC", "300"))

    @staticmethod
    def _task_manager():
        # scripts is made a package in V0.7; import lazily because task_manager has
        # environment-sensitive constants and optional Airflow runtime paths.
        from deploy_ci_cloud_agentv2.platform_backend.scripts import task_manager

        return task_manager

    @contextlib.contextmanager
    def _capture_stdout(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            yield stream

    def submit(self, task_prefix: str, config: dict[str, Any]) -> dict[str, Any]:
        tm = self._task_manager()
        with tempfile.TemporaryDirectory(prefix="dataops-agent-submit-") as tmpdir:
            yaml_path = Path(tmpdir) / "task.yaml"
            yaml_path.write_text(
                yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                legacy_task_prefix=None,
                legacy_yaml_path=None,
                task_prefix=task_prefix,
                yaml_path=str(yaml_path),
                dags_dir=str(self.settings.dags_dir),
                task_config_root=str(self.settings.task_config_root),
                parse_timeout_sec=self.parse_timeout_sec,
                # Sandbox bootstrap may validate task creation without
                # launching an Airflow DagRun.  Production keeps the normal
                # trigger behavior unless both switches explicitly opt into
                # this local-only mode.
                no_trigger=(
                    os.environ.get("PLATFORM_STAGE_RUNTIME", "").strip().lower() == "mock"
                    and os.environ.get("AUTODRIVE_PLATFORM_SUBMIT_NO_TRIGGER", "0").strip()
                    in {"1", "true", "yes"}
                ),
                schedule=None,
                scheduler_interval_sec=30,
                scheduler_once=False,
            )
            with self._capture_stdout() as output:
                result = tm.submit(args)
            payload = dict(result or {})
            payload["legacy_output"] = output.getvalue().strip()
            return payload

    def set_priority(self, task_name: str, priority: int) -> dict[str, Any]:
        tm = self._task_manager()
        args = argparse.Namespace(
            task_name=task_name,
            priority=str(priority),
            task_config_root=str(self.settings.task_config_root),
            dags_dir=str(self.settings.dags_dir),
        )
        with self._capture_stdout() as output:
            result = tm.set_task_priority(args)
        return {"task_name": task_name, "action": "priority", "priority": int(priority), "result": result, "legacy_output": output.getvalue().strip()}

    def resume(self, task_name: str, datasets: list[str] | None = None) -> dict[str, Any]:
        tm = self._task_manager()
        args = argparse.Namespace(
            task_name=task_name,
            datasets=list(datasets or []),
            dags_dir=str(self.settings.dags_dir),
            task_config_root=str(self.settings.task_config_root),
            api_base=self.settings.airflow_api_base,
            unpause=True,
        )
        with self._capture_stdout() as output:
            result = tm.resume_task(args)
        return {"task_name": task_name, "action": "resume", "datasets": list(datasets or []), "result": result, "legacy_output": output.getvalue().strip()}

    def stop(self, task_name: str, datasets: list[str] | None = None) -> dict[str, Any]:
        tm = self._task_manager()
        args = argparse.Namespace(
            task_name=task_name,
            datasets=list(datasets or []),
            dags_dir=str(self.settings.dags_dir),
            task_config_root=str(self.settings.task_config_root),
            api_base=self.settings.airflow_api_base,
            yes=True,
            stop_containers=True,
            pause_dag=True,
        )
        with self._capture_stdout() as output:
            result = tm.stop_task(args)
        return {"task_name": task_name, "action": "stop", "datasets": list(datasets or []), "result": result, "legacy_output": output.getvalue().strip()}

    def delete(self, task_name: str) -> dict[str, Any]:
        tm = self._task_manager()
        with self._capture_stdout() as output:
            result = tm.delete_task_by_name(
                task_name,
                apply_changes=True,
                dags_dir=str(self.settings.dags_dir),
                task_config_root=str(self.settings.task_config_root),
                stop_running_containers=True,
                api_base_arg=self.settings.airflow_api_base,
                use_api=True,
                print_summary=False,
            )
        payload = dict(result or {})
        payload["legacy_output"] = output.getvalue().strip()
        return payload
