#!/usr/bin/env python3
"""
Batch delete DAGs in Airflow 3.2.2 - Optimized version

Usage:
    python batch_delet_dags.py --dag-ids "dag1,dag2,dag3"
    python batch_delet_dags.py --from-file dag_list.txt
    python batch_delet_dags.py --pattern "scheduler_*"
    python batch_delet_dags.py --all-paused
    python batch_delet_dags.py --list  # List all DAGs first
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


def get_paused_dags() -> List[str]:
    """Get all paused DAG IDs."""
    code, stdout, stderr = run_airflow_cmd(["airflow", "dags", "list", "--paused"])
    if code != 0:
        print(f"Error listing paused DAGs: {stderr}")
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


def batch_pause_dags(dag_ids: List[str]) -> None:
    """Pause multiple DAGs efficiently."""
    if not dag_ids:
        return

    print(f"Pausing {len(dag_ids)} DAGs...")
    start = time.time()

    def pause_single(dag_id: str) -> None:
        run_airflow_cmd(["airflow", "dags", "pause", dag_id])

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(pause_single, dag_ids)

    elapsed = time.time() - start
    print(f"Paused {len(dag_ids)} DAGs in {elapsed:.2f}s")


def delete_dag_parallel(dag_id: str) -> Tuple[str, bool]:
    """Delete a single DAG (for parallel execution)."""
    code, stdout, stderr = run_airflow_cmd(["airflow", "dags", "delete", dag_id])
    if code == 0:
        return dag_id, True
    else:
        return dag_id, False


def delete_dags_parallel(dag_ids: List[str], max_workers: int = 8) -> Tuple[int, int]:
    """Delete DAGs in parallel."""
    if not dag_ids:
        return 0, 0

    print(f"\nDeleting {len(dag_ids)} DAGs with {max_workers} threads...")
    start = time.time()

    success_count = 0
    fail_count = 0
    failed_dags = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(delete_dag_parallel, dag_id): dag_id for dag_id in dag_ids}

        for future in concurrent.futures.as_completed(futures):
            dag_id, success = future.result()
            if success:
                success_count += 1
            else:
                fail_count += 1
                failed_dags.append(dag_id)

            progress = (success_count + fail_count) / len(dag_ids) * 100
            print(f"\rProgress: {progress:.1f}% ({success_count} success, {fail_count} failed)", end="")
            sys.stdout.flush()

    elapsed = time.time() - start
    print(f"\nDeleted {success_count} DAGs in {elapsed:.2f}s")

    if failed_dags:
        print(f"\nFailed to delete:")
        for dag_id in failed_dags[:5]:
            print(f"  - {dag_id}")
        if len(failed_dags) > 5:
            print(f"  ... and {len(failed_dags) - 5} more")

    return success_count, fail_count


def confirm_deletion(dag_ids: List[str]) -> bool:
    """Ask user to confirm deletion."""
    if not dag_ids:
        print("No DAGs to delete.")
        return False

    print(f"\nWill delete {len(dag_ids)} DAG(s):")
    for dag_id in dag_ids[:10]:
        print(f"  - {dag_id}")
    if len(dag_ids) > 10:
        print(f"  ... and {len(dag_ids) - 10} more")

    response = input("\nContinue? [y/N]: ").strip().lower()
    return response == "y"


def main():
    parser = argparse.ArgumentParser(
        description="Batch delete DAGs in Airflow 3.2.2 - Optimized version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Delete specific DAGs
  python batch_delet_dags.py --dag-ids "dag1,dag2,dag3"

  # Delete from file
  python batch_delet_dags.py --from-file dag_list.txt

  # Delete by pattern (fastest)
  python batch_delet_dags.py --pattern "scheduler_.*"

  # Delete all paused DAGs
  python batch_delet_dags.py --all-paused

  # List all DAGs first
  python batch_delet_dags.py --list

  # Skip confirmation
  python batch_delet_dags.py --dag-ids "dag1" --force

  # Custom parallel workers
  python batch_delet_dags.py --pattern "scheduler_.*" --workers 16
""",
    )
    parser.add_argument(
        "--dag-ids",
        type=str,
        help="Comma-separated list of DAG IDs to delete",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="Read DAG IDs from file (one per line)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        help="Delete DAGs matching regex pattern",
    )
    parser.add_argument(
        "--all-paused",
        action="store_true",
        help="Delete all paused DAGs",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all DAGs (no deletion)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt",
    )
    parser.add_argument(
        "--pause-first",
        action="store_true",
        default=True,
        help="Pause DAGs before deletion (default: True)",
    )
    parser.add_argument(
        "--no-pause-first",
        action="store_true",
        help="Do not pause DAGs before deletion",
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

    dag_ids_to_delete: List[str] = []

    if args.dag_ids:
        dag_ids_to_delete = [d.strip() for d in args.dag_ids.split(",") if d.strip()]

    if args.from_file:
        file_dags = read_dag_ids_from_file(args.from_file)
        dag_ids_to_delete.extend(file_dags)

    if args.pattern:
        all_dags = list_all_dags()
        matched_dags = filter_dags_by_pattern(all_dags, args.pattern)
        dag_ids_to_delete.extend(matched_dags)

    if args.all_paused:
        paused_dags = get_paused_dags()
        dag_ids_to_delete.extend(paused_dags)

    dag_ids_to_delete = list(set(dag_ids_to_delete))

    if not dag_ids_to_delete:
        print("No DAGs specified for deletion.")
        return

    if not args.force and not confirm_deletion(dag_ids_to_delete):
        print("Deletion cancelled.")
        return

    pause_first = args.pause_first and not args.no_pause_first

    if pause_first:
        # rm -rf /home/cidi/airflow/dags/data_center/* 
        batch_pause_dags(dag_ids_to_delete)


    success_count, fail_count = delete_dags_parallel(dag_ids_to_delete, args.workers)

    print(f"\n=== Summary ===")
    print(f"Total: {len(dag_ids_to_delete)}")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")


if __name__ == "__main__":
    main()
