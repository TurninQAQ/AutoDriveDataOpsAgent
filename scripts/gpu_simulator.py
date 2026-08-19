#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from platform_core.gateways.gpu_runtime import SimulatedGPURuntime, load_simulated_devices_from_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage local simulated GPU hardware state")
    parser.add_argument("--state", required=True, help="Simulator JSON state file")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init")
    init.add_argument("--config", required=True)

    set_mem = sub.add_parser("set-memory")
    set_mem.add_argument("--gpu", required=True)
    set_mem.add_argument("--used-mb", type=int, required=True)

    set_pid = sub.add_parser("set-pid")
    set_pid.add_argument("--pid", type=int, required=True)
    alive = set_pid.add_mutually_exclusive_group(required=True)
    alive.add_argument("--alive", action="store_true")
    alive.add_argument("--dead", action="store_true")

    sub.add_parser("show")
    args = parser.parse_args()

    runtime = SimulatedGPURuntime(args.state)
    if args.command == "init":
        runtime.initialize(load_simulated_devices_from_yaml(args.config), overwrite=True)
    elif args.command == "set-memory":
        runtime.set_external_used_mb(args.gpu, args.used_mb)
    elif args.command == "set-pid":
        runtime.set_process_alive(args.pid, args.alive and not args.dead)

    print(json.dumps(runtime.snapshot(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
