from datetime import datetime, timezone
from airflow.models import DagRun, TaskInstance
from airflow.utils.state import State
from airflow.utils.session import provide_session
import pendulum

DAG_ID = "batch_pipeline_universal"
# start_date=pendulum.datetime(2026, 6, 17, tz="Asia/Shanghai")
# ⚠️ Airflow 3.x 强制使用 UTC-aware 时间对象
# START_DATE = datetime(2026, 6, 25, 0, 0, 0, tzinfo=timezone.utc)
# END_DATE = datetime(2026, 6, 27, 23, 59, 59, tzinfo=timezone.utc)
START_DATE = pendulum.datetime(2026, 6, 25, 0, 0, 0, tz="Asia/Shanghai")
END_DATE = pendulum.datetime(2026, 6, 27, 23, 59, 59, tz="Asia/Shanghai")

@provide_session
def mark_runs_failed(session=None):
    # 【核心修复】Airflow 3.x 必须使用 logical_date，execution_date 已彻底移除
    runs = session.query(DagRun).filter(
        DagRun.dag_id == DAG_ID,
        DagRun.logical_date >= START_DATE,
        DagRun.logical_date <= END_DATE
    ).all()

    if not runs:
        print(f"No DAG runs found for {DAG_ID} between {START_DATE} and {END_DATE}")
        return

    print(f"Found {len(runs)} DAG Runs to process:")
    # terminal_states = [State.SUCCESS, State.FAILED, State.SKIPPED]
    terminal_states = [State.SUCCESS, State.FAILED,]
    total_marked = 0

    for run in runs:
        print(f"  -> {run.run_id} (state: {run.state})")
        
        # Airflow 3.x 中 TaskInstance 通过 run_id 关联，不再依赖 execution_date
        tis = session.query(TaskInstance).filter(
            TaskInstance.dag_id == DAG_ID,
            TaskInstance.run_id == run.run_id,
            TaskInstance.state.notin_(terminal_states)
        ).all()

        print("qzc: ", tis)
        for ti in tis:
            ti.set_state(State.FAILED, session=session)
            total_marked += 1
            print(f"     [{ti.task_id}] -> FAILED")
        print("qzc1: ")
    session.commit()
    print(f"\nDone. Marked {total_marked} task instances as FAILED.")


if __name__ == "__main__":
    mark_runs_failed()

    # scheduler_clip_dags = [
        
    # ]

