#!/usr/bin/env python3
"""
Batch trigger DAGs in Airflow 3.2.2

Usage:
    python batch_trigger_dags.py --dag-ids "dag1,dag2,dag3"
    python batch_trigger_dags.py --from-file dag_list.txt
    python batch_trigger_dags.py --pattern "scheduler_*"
"""

import argparse
import concurrent.futures
import os
import re
import subprocess
import sys
import time
from typing import List, Tuple


def run_airflow_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    """Run airflow CLI command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(
            cmd,
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


def trigger_dag_parallel(dag_id: str, conf: str = "") -> Tuple[str, bool, str]:
    """Trigger a single DAG (for parallel execution)."""
    cmd = ["airflow", "dags", "trigger", dag_id]
    if conf:
        cmd.extend(["-c", conf])

    code, stdout, stderr = run_airflow_cmd(cmd)
    if code == 0:
        run_id = ""
        for line in stdout.strip().split("\n"):
            if "run_id" in line.lower():
                run_id = line.split(":")[-1].strip()
                break
        return dag_id, True, run_id
    else:
        return dag_id, False, stderr


def trigger_dags_parallel(dag_ids: List[str], conf: str = "", max_workers: int = 8) -> Tuple[int, int, List[str]]:
    """Trigger DAGs in parallel."""
    if not dag_ids:
        return 0, 0, []

    print(f"\nTriggering {len(dag_ids)} DAGs with {max_workers} threads...")
    start = time.time()

    success_count = 0
    fail_count = 0
    failed_dags = []
    triggered_runs = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for dag_id in dag_ids:
            futures[executor.submit(trigger_dag_parallel, dag_id, conf)] = dag_id

        for future in concurrent.futures.as_completed(futures):
            dag_id, success, result = future.result()
            if success:
                success_count += 1
                triggered_runs.append(f"{dag_id} -> {result}")
            else:
                fail_count += 1
                failed_dags.append(dag_id)

            progress = (success_count + fail_count) / len(dag_ids) * 100
            print(f"\rProgress: {progress:.1f}% ({success_count} success, {fail_count} failed)", end="")
            sys.stdout.flush()

    elapsed = time.time() - start
    print(f"\nTriggered {success_count} DAGs in {elapsed:.2f}s")

    if triggered_runs:
        print(f"\nTriggered runs:")
        for run in triggered_runs[:10]:
            print(f"  {run}")
        if len(triggered_runs) > 10:
            print(f"  ... and {len(triggered_runs) - 10} more")

    if failed_dags:
        print(f"\nFailed to trigger:")
        for dag_id in failed_dags[:5]:
            print(f"  - {dag_id}")
        if len(failed_dags) > 5:
            print(f"  ... and {len(failed_dags) - 5} more")

    return success_count, fail_count, failed_dags


def confirm_trigger(dag_ids: List[str]) -> bool:
    """Ask user to confirm triggering."""
    if not dag_ids:
        print("No DAGs to trigger.")
        return False

    print(f"\nWill trigger {len(dag_ids)} DAG(s):")
    for dag_id in dag_ids[:10]:
        print(f"  - {dag_id}")
    if len(dag_ids) > 10:
        print(f"  ... and {len(dag_ids) - 10} more")

    response = input("\nContinue? [y/N]: ").strip().lower()
    return response == "y"


def main():
    parser = argparse.ArgumentParser(
        description="Batch trigger DAGs in Airflow 3.2.2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Trigger specific DAGs
  python batch_trigger_dags.py --dag-ids "dag1,dag2,dag3"

  # Trigger from file
  python batch_trigger_dags.py --from-file dag_list.txt

  # Trigger by pattern
  python batch_trigger_dags.py --pattern "scheduler_.*"

  # Trigger with config
  python batch_trigger_dags.py --dag-ids "dag1" --conf '{"param1": "value1"}'

  # List all DAGs first
  python batch_trigger_dags.py --list

  # Skip confirmation
  python batch_trigger_dags.py --dag-ids "dag1" --force

  # Custom parallel workers
  python batch_trigger_dags.py --pattern "scheduler_.*" --workers 16
""",
    )
    parser.add_argument(
        "--dag-ids",
        type=str,
        help="Comma-separated list of DAG IDs to trigger",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="Read DAG IDs from file (one per line)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        help="Trigger DAGs matching regex pattern",
    )
    parser.add_argument(
        "--conf",
        type=str,
        default="",
        help="JSON configuration to pass to DAG runs",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all DAGs (no trigger)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)",
    )

    args = parser.parse_args()

    if args.list:
        dag_ids = list_all_dags()
        print(f"\nTotal DAGs: {len(dag_ids)}")
        for dag_id in dag_ids:
            print(f"  {dag_id}")
        return

    dag_ids_to_trigger: List[str] = []

    if args.dag_ids:
        dag_ids_to_trigger = [d.strip() for d in args.dag_ids.split(",") if d.strip()]

    if args.from_file:
        file_dags = read_dag_ids_from_file(args.from_file)
        dag_ids_to_trigger.extend(file_dags)

    if args.pattern:
        all_dags = list_all_dags()
        matched_dags = filter_dags_by_pattern(all_dags, args.pattern)
        dag_ids_to_trigger.extend(matched_dags)

    dag_ids_to_trigger = list(set(dag_ids_to_trigger))

    if not dag_ids_to_trigger:
        print("No DAGs specified for triggering.")
        return

    if not args.force and not confirm_trigger(dag_ids_to_trigger):
        print("Trigger cancelled.")
        return

    success_count, fail_count, failed_dags = trigger_dags_parallel(
        dag_ids_to_trigger, args.conf, args.workers
    )

    print(f"\n=== Summary ===")
    print(f"Total: {len(dag_ids_to_trigger)}")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")


if __name__ == "__main__":
    main()
