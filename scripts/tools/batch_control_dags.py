#!/usr/bin/env python3
"""
Batch control DAGs (pause/unpause) in Airflow 3.2.2

Usage:
    python batch_control_dags.py --action pause --dag-ids "dag1,dag2,dag3"
    python batch_control_dags.py --action unpause --pattern "scheduler_*"
    python batch_control_dags.py --action pause --from-file dag_list.txt
"""

import argparse
import concurrent.futures
import os
import re
import fnmatch
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
    code, stdout, stderr = run_airflow_cmd(["airflow", "dags", "list"])
    if code != 0:
        print(f"Error listing DAGs: {stderr}")
        return []

    dag_ids = []
    for line in stdout.strip().split("\n"):
        if line and not line.startswith("dag_id"):
            parts = line.split("|")
            if len(parts) >= 4:
                dag_id = parts[0].strip()
                paused = parts[3].strip().lower()
                if dag_id and paused == "true":
                    dag_ids.append(dag_id)
    return dag_ids


def get_unpaused_dags() -> List[str]:
    """Get all unpaused (active) DAG IDs."""
    code, stdout, stderr = run_airflow_cmd(["airflow", "dags", "list"])
    if code != 0:
        print(f"Error listing DAGs: {stderr}")
        return []

    dag_ids = []
    for line in stdout.strip().split("\n"):
        if line and not line.startswith("dag_id"):
            parts = line.split("|")
            if len(parts) >= 4:
                dag_id = parts[0].strip()
                paused = parts[3].strip().lower()
                if dag_id and paused == "false":
                    dag_ids.append(dag_id)
    return dag_ids


def filter_dags_by_pattern(dag_ids: List[str], pattern: str) -> List[str]:
    """Filter DAG IDs by regex pattern."""
    try:
        # regex = re.compile(pattern)
        return [dag_id for dag_id in dag_ids if fnmatch.fnmatch(dag_id, pattern)]
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


def control_dag_parallel(dag_id: str, action: str) -> Tuple[str, bool]:
    """Pause or unpause a single DAG (for parallel execution)."""
    cmd = ["airflow", "dags", action, dag_id]
    code, stdout, stderr = run_airflow_cmd(cmd)
    return dag_id, code == 0


def control_dags_parallel(dag_ids: List[str], action: str, max_workers: int = 8) -> Tuple[int, int]:
    """Control DAGs in parallel."""
    if not dag_ids:
        return 0, 0

    action_text = "Pausing" if action == "pause" else "Unpausing"
    print(f"\n{action_text} {len(dag_ids)} DAGs with {max_workers} threads...")
    start = time.time()

    success_count = 0
    fail_count = 0
    failed_dags = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(control_dag_parallel, dag_id, action): dag_id for dag_id in dag_ids}

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
    print(f"\n{action_text} {success_count} DAGs in {elapsed:.2f}s")

    if failed_dags:
        print(f"\nFailed to {action}:")
        for dag_id in failed_dags[:5]:
            print(f"  - {dag_id}")
        if len(failed_dags) > 5:
            print(f"  ... and {len(failed_dags) - 5} more")

    return success_count, fail_count


def confirm_action(dag_ids: List[str], action: str) -> bool:
    """Ask user to confirm action."""
    if not dag_ids:
        print("No DAGs to control.")
        return False

    action_text = "pause" if action == "pause" else "unpause"
    print(f"\nWill {action_text} {len(dag_ids)} DAG(s):")
    for dag_id in dag_ids[:10]:
        print(f"  - {dag_id}")
    if len(dag_ids) > 10:
        print(f"  ... and {len(dag_ids) - 10} more")

    response = input("\nContinue? [y/N]: ").strip().lower()
    return response == "y"


def main():
    parser = argparse.ArgumentParser(
        description="Batch control DAGs (pause/unpause) in Airflow 3.2.2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pause specific DAGs
  python batch_control_dags.py --action pause --dag-ids "dag1,dag2,dag3"

  # Unpause by pattern
  python batch_control_dags.py --action unpause --pattern "scheduler_.*"

  # Pause from file
  python batch_control_dags.py --action pause --from-file dag_list.txt

  # Pause all currently unpaused DAGs
  python batch_control_dags.py --action pause --all-unpaused

  # Unpause all currently paused DAGs
  python batch_control_dags.py --action unpause --all-paused

  # List all DAGs first
  python batch_control_dags.py --list

  # List paused DAGs
  python batch_control_dags.py --list-paused

  # Skip confirmation
  python batch_control_dags.py --action pause --dag-ids "dag1" --force

  # Custom parallel workers
  python batch_control_dags.py --action pause --pattern "scheduler_.*" --workers 16
""",
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=["pause", "unpause"],
        required=False,
        help="Action to perform: pause or unpause",
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
        help="Control DAGs matching regex pattern",
    )
    parser.add_argument(
        "--all-paused",
        action="store_true",
        help="Control all currently paused DAGs",
    )
    parser.add_argument(
        "--all-unpaused",
        action="store_true",
        help="Control all currently unpaused DAGs",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all DAGs (no action)",
    )
    parser.add_argument(
        "--list-paused",
        action="store_true",
        help="List paused DAGs (no action)",
    )
    parser.add_argument(
        "--list-unpaused",
        action="store_true",
        help="List unpaused DAGs (no action)",
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

    if args.list_paused:
        dag_ids = get_paused_dags()
        print(f"\nPaused DAGs: {len(dag_ids)}")
        for dag_id in dag_ids:
            print(f"  {dag_id}")
        return

    if args.list_unpaused:
        dag_ids = get_unpaused_dags()
        print(f"\nUnpaused DAGs: {len(dag_ids)}")
        for dag_id in dag_ids:
            print(f"  {dag_id}")
        return

    if not args.action:
        print("Error: --action is required when not using --list, --list-paused, or --list-unpaused")
        parser.print_help()
        return

    dag_ids_to_control: List[str] = []

    if args.dag_ids:
        dag_ids_to_control = [d.strip() for d in args.dag_ids.split(",") if d.strip()]

    if args.from_file:
        file_dags = read_dag_ids_from_file(args.from_file)
        dag_ids_to_control.extend(file_dags)

    if args.pattern:
        all_dags = list_all_dags()
        print(f"Total DAGs: {len(all_dags)}")
        print(all_dags)
        print(args.pattern)
        matched_dags = filter_dags_by_pattern(all_dags, args.pattern)
        print(f"Matched DAGs: {len(matched_dags)}")
        for dag_id in matched_dags:
            print(f"  {dag_id}")
        dag_ids_to_control.extend(matched_dags)

    if args.all_paused:
        paused_dags = get_paused_dags()
        dag_ids_to_control.extend(paused_dags)

    if args.all_unpaused:
        unpaused_dags = get_unpaused_dags()
        dag_ids_to_control.extend(unpaused_dags)

    dag_ids_to_control = list(set(dag_ids_to_control))

    if not dag_ids_to_control:
        print("No DAGs specified for control.")
        return

    if not args.force and not confirm_action(dag_ids_to_control, args.action):
        print("Action cancelled.")
        return

    success_count, fail_count = control_dags_parallel(dag_ids_to_control, args.action, args.workers)

    print(f"\n=== Summary ===")
    print(f"Total: {len(dag_ids_to_control)}")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")


if __name__ == "__main__":
    main()
