"""
batch_clear_failed.py
批量重置多个 DAG 中失败的 TaskInstance (适配定制版 Airflow)
"""
from datetime import datetime, timezone
from airflow.models import DagBag, DagRun, DagModel
from airflow.models.taskinstance import clear_task_instances
from airflow.utils.session import provide_session
from airflow.utils.state import TaskInstanceState

# ================= 配置区 =================
TARGET_DAG_IDS = [
    "scheduler_clip_000_20260617_153655",
    "scheduler_clip_001_20260617_153725",
    "scheduler_clip_002_20260617_153756",
    "scheduler_clip_003_20260617_153828",
    "scheduler_clip_004_20260617_153859",
    "scheduler_clip_005_20260617_153930",
    "scheduler_clip_006_20260617_154001",
    "scheduler_clip_007_20260617_154032",
    "scheduler_clip_008_20260617_154104",
    "scheduler_clip_010_20260617_154206",
    "scheduler_clip_011_20260617_154237",
    "scheduler_clip_012_20260617_154308",
    "scheduler_clip_013_20260617_154340",
    "scheduler_clip_014_20260617_154411",
    "scheduler_clip_015_20260617_154442",
    "scheduler_clip_016_20260617_154513",
    "scheduler_clip_017_20260617_154544",
    "scheduler_clip_018_20260617_154616",
    "scheduler_clip_019_20260617_154647",
    "scheduler_clip_020_20260617_154718",
    "scheduler_clip_021_20260617_154749",
]

# ⚠️ 注意：请确认你的 partition_date 存储的是 UTC 还是本地时间
# 如果是本地时间(如 Asia/Shanghai)，请将 tzinfo 改为对应时区
START_DATE = datetime(2026, 6, 26, 0, 0, 0, tzinfo=timezone.utc)
END_DATE = datetime(2026, 6, 27, 23, 59, 59, tzinfo=timezone.utc)

RECURSIVE = False
# ==========================================


@provide_session
def batch_clear_failed_dags(dag_ids: list[str], start_date: datetime, end_date: datetime, recursive: bool = False, session=None):
    """批量清除指定 DAG 列表中失败的 TaskInstance"""
    # 兼容 DagBag 初始化
    try:
        dagbag = DagBag(read_dags_from_db=True)
    except TypeError:
        dagbag = DagBag()

    total_cleared = 0

    for dag_id in dag_ids:
        print(f"\n{'='*60}")
        print(f"🔍 Processing DAG: {dag_id}")

        if dag_id not in dagbag.dags:
            print(f"   ❌ DAG not found in DagBag, skipping.")
            continue

        dag = dagbag.dags[dag_id]

        # 检查暂停状态
        dag_model = session.query(DagModel).filter(DagModel.dag_id == dag_id).first()
        if dag_model and dag_model.is_paused:
            print(f"   ⚠️  DAG is PAUSED. Cleared tasks won't be scheduled until unpaused.")

        # ✅ 核心修复：使用定制版字段 partition_date 进行时间范围过滤
        dag_runs = (
            session.query(DagRun)
            .filter(
                DagRun.dag_id == dag_id,
                DagRun.partition_date >= start_date,
                DagRun.partition_date <= end_date,
            )
            .all()
        )

        if not dag_runs:
            print(f"   ✅ No DagRuns in range, skipping.")
            continue

        # 收集失败的 TaskInstance
        tis_to_clear = []
        for dr in dag_runs:
            for ti in dr.task_instances:
                if ti.state in (TaskInstanceState.FAILED, TaskInstanceState.UPSTREAM_FAILED):
                    tis_to_clear.append(ti)

        if not tis_to_clear:
            print(f"   ✅ {len(dag_runs)} DagRuns found but no failed tasks, skipping.")
            continue

        # 执行 Clear
        try:
            cleared_count = clear_task_instances(
                tis=tis_to_clear,
                session=session,
                activate_dag_runs=True,
                dag=dag,
                include_parentdag=False,
                include_subdags=recursive,
            )
            session.commit()
            total_cleared += cleared_count
            print(f"   ✅ Cleared {cleared_count} failed tasks across {len(dag_runs)} DagRuns")

        except Exception as e:
            session.rollback()
            print(f"   ❌ Clear failed: {e}")

    print(f"\n{'='*60}")
    print(f"📊 Summary: Total cleared {total_cleared} task instances")


if __name__ == "__main__":
    batch_clear_failed_dags(
        dag_ids=TARGET_DAG_IDS,
        start_date=START_DATE,
        end_date=END_DATE,
        recursive=RECURSIVE,
    )
