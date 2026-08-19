"""Platform-specific Airflow startup hooks.

This file is loaded by Airflow from ``$AIRFLOW_HOME/config``.  Keep imports
lazy and side effects tightly scoped: the only patch here redirects the native
Airflow DAG delete API for generated platform task DAGs.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from airflow.exceptions import AirflowException
from airflow.utils.session import NEW_SESSION, provide_session

__all__ = ()

LOG = logging.getLogger(__name__)
PATCH_MARKER = "_deploy_ci_cloud_original_delete_dag"


def _ensure_scripts_path() -> None:
    candidates = []
    scripts_dir = os.environ.get("AIRFLOW_SCRIPTS_DIR")
    if scripts_dir:
        candidates.append(Path(scripts_dir))

    airflow_home = Path(os.environ.get("AIRFLOW_HOME", "")).expanduser()
    if airflow_home:
        candidates.append(airflow_home.parent / "opt_airflow" / "scripts")

    for candidate in candidates:
        if candidate.is_dir():
            candidate_text = str(candidate)
            if candidate_text not in sys.path:
                sys.path.insert(0, candidate_text)
            return


def _load_task_manager():
    _ensure_scripts_path()
    try:
        import task_manager
    except Exception as exc:
        raise AirflowException(f"Platform task delete hook cannot import task_manager: {exc}") from exc
    return task_manager


def _platform_task_delete_class(task_manager, task_name: str) -> str:
    paths = task_manager.task_paths(task_name)
    config_exists = Path(paths["config_file"]).is_file()
    generated_dag_exists = Path(paths["dag_file"]).is_file()
    if config_exists:
        return "platform_task"
    if generated_dag_exists:
        return "inconsistent"
    return "ordinary_dag"


def _patch_delete_dag() -> None:
    if os.environ.get("AIRFLOW_PLATFORM_DELETE_PATCH_DISABLED") == "1":
        LOG.info("Platform DAG delete patch disabled by AIRFLOW_PLATFORM_DELETE_PATCH_DISABLED=1")
        return

    from airflow.api.common import delete_dag as delete_dag_module

    original_delete_dag = getattr(delete_dag_module, PATCH_MARKER, None)
    if original_delete_dag is None:
        original_delete_dag = delete_dag_module.delete_dag
        setattr(delete_dag_module, PATCH_MARKER, original_delete_dag)

    @provide_session
    def platform_delete_dag(
        dag_id: str,
        keep_records_in_log: bool = True,
        session=NEW_SESSION,
    ) -> int:
        if os.environ.get("AIRFLOW_PLATFORM_DELETE_BYPASS") == "1":
            return original_delete_dag(
                dag_id,
                keep_records_in_log=keep_records_in_log,
                session=session,
            )

        task_manager = _load_task_manager()
        if dag_id in task_manager.PROTECTED_PLATFORM_DAG_IDS:
            raise AirflowException(f"Platform shared DAG cannot be deleted from Airflow UI: {dag_id}")

        task_name = task_manager.task_name_from_generated_dag_id(dag_id)
        if not task_name:
            return original_delete_dag(
                dag_id,
                keep_records_in_log=keep_records_in_log,
                session=session,
            )

        delete_class = _platform_task_delete_class(task_manager, task_name)
        if delete_class == "ordinary_dag":
            return original_delete_dag(
                dag_id,
                keep_records_in_log=keep_records_in_log,
                session=session,
            )
        if delete_class == "inconsistent":
            raise AirflowException(
                "Platform task DAG state is inconsistent; refuse UI delete. "
                f"dag_id={dag_id} task_name={task_name}"
            )

        try:
            result = task_manager.delete_task_by_name(
                task_name,
                apply_changes=True,
                stop_running_containers=True,
                use_api=False,
                original_delete_dag=original_delete_dag,
                session=session,
                keep_records_in_log=keep_records_in_log,
                print_summary=False,
            )
        except Exception as exc:
            LOG.exception("Platform task DAG delete failed: dag_id=%s task_name=%s", dag_id, task_name)
            raise AirflowException(
                f"Platform task DAG delete failed: dag_id={dag_id} task_name={task_name}: {exc}"
            ) from exc

        LOG.info(
            "Platform task DAG deleted from Airflow UI: dag_id=%s task_name=%s result=%s",
            dag_id,
            task_name,
            result,
        )
        return int(result.get("dag_metadata_deleted") or 0)

    delete_dag_module.delete_dag = platform_delete_dag
    LOG.info("Installed platform DAG delete patch")


_patch_delete_dag()
