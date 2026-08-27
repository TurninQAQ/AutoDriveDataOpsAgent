from pathlib import Path
import sys
from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.models.param import Param
from airflow.providers.standard.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule


DATA_CENTER_DIR = Path(__file__).resolve().parents[1]
if str(DATA_CENTER_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_CENTER_DIR))

from batch_pipeline_universal import (
    finalize_task_queue_task,
    run_shell_script,
    run_validate,
    verify_pipeline_terminal_state,
)


DAG_ID = "__DAG_ID__"
MAX_ACTIVE_RUNS = __MAX_ACTIVE_RUNS__
PIPELINE_STAGE_GROUPS = __PIPELINE_STAGE_GROUPS__


def flatten_stage_groups(stage_groups):
    return [stage for group in stage_groups for stage in group]


STAGES = flatten_stage_groups(PIPELINE_STAGE_GROUPS)


BASE_PARAMS = {
    "task_name": Param(type="string", default="__TASK_NAME__", description="Submitted task name"),
    "pipeline_stages": Param(type="array", default=PIPELINE_STAGE_GROUPS, description="Pipeline stage groups"),
    "dataset_name": Param(type="string", description="Target dataset name"),
    "dataset_path": Param(type="string", description="Dataset data path"),
    "pool": Param(type="string", default="pool_small", description="Pool name"),
    "tier": Param(type="string", default="small", description="Pool tier"),
    "timeout_min": Param(type="integer", default=60, description="Timeout in minutes"),
    "task_exclusive": Param(
        type="boolean",
        default=True,
        description="Only one submitted task_name may execute stages platform-wide at a time",
    ),
    "task_lock_wait_interval_sec": Param(
        type="integer",
        default=10,
        description="Wait interval while another task_name owns the platform task lock",
    ),
    "preempt_grace_timeout_min": Param(
        type="integer",
        default=60,
        description="Minutes to wait before hard-cleaning a preempted task",
    ),
    "gpu_ids": Param(type="string", default="", description="GPU pool"),
    "gpu_stages": Param(type="string", default="", description="Stages requiring GPU"),
    "exclusive_gpu_stages": Param(
        type=["null", "string"],
        default=None,
        description="GPU stages that require exclusive GPU access; null defaults to gpu_stages",
    ),
    "exclusive_gpu_idle_used_max_mb": Param(
        type="integer",
        default=256,
        description="Maximum used GPU memory allowed before assigning an exclusive GPU stage",
    ),
    "gpu_stage_memory_mb": Param(
        type="object",
        default={},
        description="GPU memory reservation per stage in MiB",
    ),
    "gpu_wait_interval_sec": Param(
        type=["null", "integer"],
        default=None,
        description="GPU wait retry interval",
    ),
    "gpu_reservation_pending_sec": Param(
        type=["null", "integer"],
        default=None,
        description="Compatibility field; active reservations are held for the full stage runtime",
    ),
}

for stage in STAGES:
    BASE_PARAMS[f"image_{stage}"] = Param(
        type="string",
        description=f"Image for stage {stage.upper()}",
    )


with DAG(
    dag_id=DAG_ID,
    schedule=None,
    catchup=False,
    max_active_runs=MAX_ACTIVE_RUNS,
    params=BASE_PARAMS,
    tags=["flywheel-batch", "submitted-task", "__TASK_NAME__"],
    default_args={"retries": 1, "retry_delay": timedelta(seconds=60)},
    start_date=pendulum.datetime(2026, 6, 17, tz="Asia/Shanghai"),
) as dag:
    stage_tasks = {}
    for stage in STAGES:
        run_task = PythonOperator(
            task_id=f"run_{stage}",
            python_callable=run_shell_script,
            op_kwargs={"script_name": f"run_{stage}.sh"},
            pool="default_pool",
        )
        validate_task = PythonOperator(
            task_id=f"validate_{stage}",
            python_callable=run_validate,
            op_kwargs={"task_suffix": stage},
            pool="default_pool",
        )
        run_task >> validate_task
        stage_tasks[stage] = (run_task, validate_task)

    previous_validates = []
    for group in PIPELINE_STAGE_GROUPS:
        current_runs = [stage_tasks[stage][0] for stage in group]
        current_validates = [stage_tasks[stage][1] for stage in group]
        for previous_validate in previous_validates:
            for current_run in current_runs:
                previous_validate >> current_run
        previous_validates = current_validates

    verify_status = PythonOperator(
        task_id="verify_pipeline_status",
        python_callable=verify_pipeline_terminal_state,
        trigger_rule=TriggerRule.ALL_DONE,
        pool="default_pool",
    )
    for previous_validate in previous_validates:
        previous_validate >> verify_status

    finalize_queue = PythonOperator(
        task_id="finalize_task_queue",
        python_callable=finalize_task_queue_task,
        trigger_rule=TriggerRule.ALL_DONE,
        pool="default_pool",
    )
    verify_status >> finalize_queue
