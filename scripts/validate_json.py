#!/usr/bin/env python3
import argparse, json, os, sys
from datetime import datetime, timezone

def validate(root_dir, dataset_name, task_suffix, min_date):
    file_path = os.path.join(root_dir, dataset_name, f"results_{task_suffix}.json")

    if not os.path.exists(file_path):
        print(f"[FAIL] File not found: {file_path}", file=sys.stderr)
        sys.exit(1)
    try:
        with open(file_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[FAIL] Invalid JSON in {file_path}: {e}", file=sys.stderr)
        sys.exit(1)
    if data.get("status") != "success":
        print(f"[FAIL] {task_suffix} failed: {data.get('error_message','unknown')}", file=sys.stderr)
        sys.exit(1)
    if data.get("dataset_name") != dataset_name:
        print(f"[FAIL] Dataset mismatch: expected={dataset_name}, got={data.get('dataset_name')}", file=sys.stderr)
        sys.exit(1)
    ts = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
    min_dt = datetime.strptime(min_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if ts < min_dt:
        print(f"[FAIL] Stale JSON: timestamp={ts} < min_date={min_dt}", file=sys.stderr)
        sys.exit(1)

    print(f"[PASS] {task_suffix}/{dataset_name} validated OK")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root-dir", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--task-suffix", required=True)
    p.add_argument("--min-date", required=True)
    args = p.parse_args()
    validate(args.root_dir, args.dataset, args.task_suffix, args.min_date)