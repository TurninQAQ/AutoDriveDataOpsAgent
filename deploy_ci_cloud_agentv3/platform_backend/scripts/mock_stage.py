#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_result(dataset_path: Path, dataset_name: str, stage: str, status: str, error: str = "") -> Path:
    dataset_dir = dataset_path / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    result_path = dataset_dir / f"results_{stage}.json"
    payload = {
        "dataset_path": str(dataset_dir),
        "record_path": str(dataset_path),
        "dataset_name": dataset_name,
        "stage": stage,
        "status": status,
        "error_message": error,
        "timestamp": utc_now_text(),
        "mock": True,
    }
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result_path


def run(stage: str, dataset_path: Path, dataset_name: str, duration_sec: float, result: str) -> int:
    print(
        f"[MOCK] stage={stage} dataset={dataset_name} result={result} duration_sec={duration_sec}",
        flush=True,
    )
    if duration_sec > 0:
        time.sleep(duration_sec)

    if result == "success":
        path = write_result(dataset_path, dataset_name, stage, "success")
        print(f"[MOCK] wrote success result: {path}", flush=True)
        return 0
    if result == "validate_fail":
        path = write_result(
            dataset_path,
            dataset_name,
            stage,
            "failed",
            "mock validation failure",
        )
        print(f"[MOCK] wrote invalid result for validator: {path}", flush=True)
        return 0
    if result == "fail":
        path = write_result(dataset_path, dataset_name, stage, "failed", "mock stage failure")
        print(f"[MOCK] stage failure; result: {path}", file=sys.stderr, flush=True)
        return 1
    if result == "oom":
        path = write_result(
            dataset_path,
            dataset_name,
            stage,
            "failed",
            "CUDA out of memory (simulated)",
        )
        print(
            "CUDA out of memory. Tried to allocate simulated GPU memory.",
            file=sys.stderr,
            flush=True,
        )
        print(f"[MOCK] oom result: {path}", file=sys.stderr, flush=True)
        return 137
    if result == "timeout":
        # A timeout result deliberately remains alive. The caller owns the timeout
        # and termination policy, matching a hung algorithm container/process.
        while True:
            time.sleep(60)
    raise ValueError(f"Unsupported mock result: {result}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic mock algorithm stage")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--dataset-path", default=os.environ.get("DATASET_PATH"))
    parser.add_argument("--dataset-name", default=os.environ.get("DATASET_NAME"))
    parser.add_argument("--duration-sec", type=float, default=0.0)
    parser.add_argument(
        "--result",
        choices=["success", "fail", "timeout", "validate_fail", "oom"],
        default="success",
    )
    args = parser.parse_args()
    if not args.dataset_path or not args.dataset_name:
        parser.error("--dataset-path and --dataset-name are required (or set DATASET_PATH/DATASET_NAME)")
    return run(
        args.stage,
        Path(args.dataset_path),
        args.dataset_name,
        max(0.0, args.duration_sec),
        args.result,
    )


if __name__ == "__main__":
    raise SystemExit(main())
