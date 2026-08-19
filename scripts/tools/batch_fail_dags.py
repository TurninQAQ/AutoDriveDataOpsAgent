#!/usr/bin/env python3
"""
Batch set DAG runs to failed status in Airflow 3.2.2

Usage:
    python batch_fail_dags.py --dag-ids "dag1,dag2,dag3"
    python batch_fail_dags.py --pattern "scheduler_*"
    python batch_fail_dags.py --all-running
"""

import argparse
import concurrent.futures
import os
import re
import subprocess
import sys
import time
from typing import List, Tuple

AIRFLOW_BIN = os.environ.get(
    "AIRFLOW_BIN", "/home/cidi/miniforge3/envs/airflow/bin/airflow"
)


def run_airflow_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    """Run airflow CLI command and return exit code, stdout, stderr."""
    try:
        command = [AIRFLOW_BIN if cmd[0] == "airflow" else cmd[0], *cmd[1:]]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"
    except Exception as e:
        return 1, "", str(e)


def list_all_dags() -> List[str]:
    """List all DAG IDs."""
    code, stdout, stderr = run_airflow_cmd(["airflow", "dags", "list"])
    if code != 0:
        print(f"Error listing DAGs: {stderr}")
        return []

    dag_ids = []
    for line in stdout.strip().split("\n"):
        if line and not line.startswith("dag_id"):
            parts = line.split("|")
            if parts:
                dag_id = parts[0].strip()
                if dag_id:
                    dag_ids.append(dag_id)
    return dag_ids


def get_running_dag_runs_for_dag(dag_id: str) -> List[Tuple[str, str]]:
    """Get running DAG runs for a specific DAG."""
    code, stdout, stderr = run_airflow_cmd(
        ["airflow", "dags", "list-runs", dag_id]
    )
    if code != 0:
        return []

    runs = []
    for line in stdout.strip().split("\n"):
        if line and not line.startswith("run_id"):
            parts = line.split("|")
            if len(parts) >= 3:
                run_id = parts[0].strip()
                state = parts[2].strip().lower() if len(parts) > 2 else ""
                if run_id and state == "running":
                    runs.append((dag_id, run_id))
    return runs


def get_running_dag_runs() -> List[Tuple[str, str]]:
    """Get all running DAG runs (dag_id, run_id)."""
    all_dags = list_all_dags()
    if not all_dags:
        return []

    print(f"Checking {len(all_dags)} DAGs for running runs...")

    all_runs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(get_running_dag_runs_for_dag, dag_id): dag_id for dag_id in all_dags}

        for future in concurrent.futures.as_completed(futures):
            runs = future.result()
            all_runs.extend(runs)

    return all_runs


def filter_dags_by_pattern(dag_ids: List[str], pattern: str) -> List[str]:
    """Filter DAG IDs by regex pattern."""
    try:
        regex = re.compile(pattern)
        return [dag_id for dag_id in dag_ids if regex.match(dag_id)]
    except re.error as e:
        print(f"Invalid regex pattern: {e}")
        return []


def read_dag_ids_from_file(filepath: str) -> List[str]:
    """Read DAG IDs from a file (one per line)."""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return []

    dag_ids = []
    with open(filepath) as f:
        for line in f:
            dag_id = line.strip()
            if dag_id and not dag_id.startswith("#"):
                dag_ids.append(dag_id)
    return dag_ids


def fail_dag_run(dag_id: str, run_id: str = None) -> Tuple[str, bool]:
    """Set DAG runs to failed through the Airflow ORM for any supported DB backend."""
    from airflow.models.dagrun import DagRun
    from airflow.models.taskinstance import TaskInstance
    from airflow.utils import timezone
    from airflow.utils.session import create_session
    from airflow.utils.state import DagRunState, TaskInstanceState

    try:
        with create_session() as session:
            query = session.query(DagRun).filter(DagRun.dag_id == dag_id)
            if run_id:
                query = query.filter(DagRun.run_id == run_id)
            else:
                query = query.filter(DagRun.state.in_([DagRunState.QUEUED, DagRunState.RUNNING]))

            runs = query.all()
            for dag_run in runs:
                (
                    session.query(TaskInstance)
                    .filter(
                        TaskInstance.dag_id == dag_id,
                        TaskInstance.run_id == dag_run.run_id,
                        TaskInstance.state.in_(
                            [
                                TaskInstanceState.SCHEDULED,
                                TaskInstanceState.QUEUED,
                                TaskInstanceState.RUNNING,
                                TaskInstanceState.UP_FOR_RETRY,
                                TaskInstanceState.UP_FOR_RESCHEDULE,
                                TaskInstanceState.DEFERRED,
                            ]
                        ),
                    )
                    .update(
                        {
                            TaskInstance.state: TaskInstanceState.FAILED,
                            TaskInstance.end_date: timezone.utcnow(),
                        },
                        synchronize_session=False,
                    )
                )
                dag_run.set_state(DagRunState.FAILED)
        return dag_id, True
    except Exception as e:
        print(f"Database update failed for {dag_id}: {e}")
        return dag_id, False


def fail_dag_runs_parallel(dag_runs: List[Tuple[str, str]], max_workers: int = 8) -> Tuple[int, int]:
    """Set DAG runs to failed in parallel."""
    if not dag_runs:
        return 0, 0

    print(f"\nSetting {len(dag_runs)} DAG runs to failed with {max_workers} threads...")
    start = time.time()

    success_count = 0
    fail_count = 0
    failed_dags = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for dag_id, run_id in dag_runs:
            futures[executor.submit(fail_dag_run, dag_id, run_id)] = (dag_id, run_id)

        for future in concurrent.futures.as_completed(futures):
            dag_id, success = future.result()
            if success:
                success_count += 1
            else:
                fail_count += 1
                failed_dags.append(dag_id)

            progress = (success_count + fail_count) / len(dag_runs) * 100
            print(f"\rProgress: {progress:.1f}% ({success_count} success, {fail_count} failed)", end="")
            sys.stdout.flush()

    elapsed = time.time() - start
    print(f"\nSet {success_count} DAG runs to failed in {elapsed:.2f}s")

    if failed_dags:
        print(f"\nFailed to set to failed:")
        for dag_id in failed_dags[:5]:
            print(f"  - {dag_id}")
        if len(failed_dags) > 5:
            print(f"  ... and {len(failed_dags) - 5} more")

    return success_count, fail_count


def confirm_action(dag_runs: List[Tuple[str, str]]) -> bool:
    """Ask user to confirm action."""
    if not dag_runs:
        print("No DAG runs to set to failed.")
        return False

    print(f"\nWill set {len(dag_runs)} DAG runs to failed:")
    for dag_id, run_id in dag_runs[:10]:
        print(f"  - {dag_id} ({run_id})")
    if len(dag_runs) > 10:
        print(f"  ... and {len(dag_runs) - 10} more")

    response = input("\nContinue? [y/N]: ").strip().lower()
    return response == "y"


def main():
    parser = argparse.ArgumentParser(
        description="Batch set DAG runs to failed status in Airflow 3.2.2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Set specific DAG runs to failed
  python batch_fail_dags.py --dag-ids "dag1,dag2,dag3"

  # Set by pattern
  python batch_fail_dags.py --pattern "scheduler_.*"

  # Set all running DAG runs to failed
  python batch_fail_dags.py --all-running

  # From file
  python batch_fail_dags.py --from-file dag_list.txt

  # List running DAG runs first
  python batch_fail_dags.py --list-running

  # Skip confirmation
  python batch_fail_dags.py --dag-ids "dag1" --force

  # Custom parallel workers
  python batch_fail_dags.py --pattern "scheduler_.*" --workers 16
""",
    )
    parser.add_argument(
        "--dag-ids",
        type=str,
        help="Comma-separated list of DAG IDs",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="Read DAG IDs from file (one per line)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        help="Set DAGs matching regex pattern to failed",
    )
    parser.add_argument(
        "--all-running",
        action="store_true",
        help="Set all running DAG runs to failed",
    )
    parser.add_argument(
        "--list-running",
        action="store_true",
        help="List all running DAG runs",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1; increase only with PostgreSQL)",
    )

    args = parser.parse_args()

    if args.list_running:
        runs = get_running_dag_runs()
        print(f"\nRunning DAG runs: {len(runs)}")
        for dag_id, run_id in runs:
            print(f"  {dag_id} ({run_id})")
        return

    dag_runs_to_fail: List[Tuple[str, str]] = []

    if args.all_running:
        dag_runs_to_fail = get_running_dag_runs()

    if args.dag_ids:
        dag_ids = [d.strip() for d in args.dag_ids.split(",") if d.strip()]
        for dag_id in dag_ids:
            dag_runs_to_fail.append((dag_id, None))

    if args.from_file:
        file_dags = read_dag_ids_from_file(args.from_file)
        for dag_id in file_dags:
            dag_runs_to_fail.append((dag_id, None))

    if args.pattern:
        all_dags = list_all_dags()
        matched_dags = filter_dags_by_pattern(all_dags, args.pattern)
        for dag_id in matched_dags:
            dag_runs_to_fail.append((dag_id, None))

    dag_runs_to_fail = list(set(dag_runs_to_fail))

    if not dag_runs_to_fail:
        print("No DAG runs specified to set to failed.")
        return

    if not args.force and not confirm_action(dag_runs_to_fail):
        print("Action cancelled.")
        return

    success_count, fail_count = fail_dag_runs_parallel(dag_runs_to_fail, args.workers)

    print(f"\n=== Summary ===")
    print(f"Total: {len(dag_runs_to_fail)}")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")


if __name__ == "__main__":
    main()
