from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from platform_agent.approval import ApprovalStore
from platform_agent.memory import ConversationStore
from platform_agent.model import HeuristicReadOnlyModel
from platform_agent.models import ToolCallSpec, ToolObservation
from platform_agent.verification import ActionVerifier
from platform_agent.workflow import build_agent_runtime
from platform_core.config import normalize_task_priority_config, validate_config
from platform_core.gateways.gpu_runtime import SimulatedGPURuntime
from platform_mcp.server import READ_ONLY_TOOL_NAMES
from platform_observability import TraceRecorder, TraceStore
from platform_planning.service import TaskPlanningService


class E2EStep(BaseModel):
    name: str
    ok: bool
    detail: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class LocalE2EResult(BaseModel):
    ok: bool
    steps: list[E2EStep]
    trace_count: int = 0
    audit_count: int = 0
    artifacts_root: str = ""


class LocalScenarioToolClient:
    """Deterministic external-system substitute for local end-to-end hardening.

    It implements the same Agent Tool contracts as the MCP layer while keeping all
    Agent, planning, approval, precondition and verification code real. It is not a
    replacement for a full Airflow runtime smoke test and is intentionally located
    under platform_hardening rather than platform_core.
    """

    def __init__(self, gpu_runtime: SimulatedGPURuntime):
        self.gpu_runtime = gpu_runtime
        self.tasks: dict[str, dict[str, Any]] = {}
        self.active: str | None = None
        self.queue: list[str] = []
        self.draining: str | None = None
        self.generation = 1
        self.counter = 0
        self.calls: list[ToolCallSpec] = []

    async def describe_tools(self):
        return [{"name": name, "description": name, "input_schema": {}} for name in READ_ONLY_TOOL_NAMES]

    def _hash(self) -> str:
        raw = json.dumps(
            {
                "generation": self.generation,
                "active": self.active,
                "queue": self.queue,
                "draining": self.draining,
                "tasks": {name: {"priority": t["priority"], "exists": t["exists"]} for name, t in self.tasks.items()},
            },
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    def _precondition(self, task_name: str = "") -> dict[str, Any]:
        task = self.tasks.get(task_name) if task_name else None
        return {
            "queue_sha256": self._hash(),
            "task_name": task_name,
            "task_exists": bool(task and task["exists"]),
            "task_config_sha256": str(task.get("config_sha256") or "") if task else "",
        }

    def _assert_precondition(self, expected: dict[str, Any]):
        task_name = str(expected.get("task_name") or "")
        current = self._precondition(task_name)
        for key in ("queue_sha256", "task_exists", "task_config_sha256"):
            if expected.get(key) != current.get(key):
                raise RuntimeError(f"PRECONDITION_FAILED: {key} expected={expected.get(key)!r} actual={current.get(key)!r}")

    def _location(self, task_name: str) -> tuple[str, int | None]:
        if self.draining == task_name:
            return "draining", 0
        if self.active == task_name:
            return "active", 0
        if task_name in self.queue:
            return "queued", self.queue.index(task_name) + 1
        return "not_found", None

    def _entry(self, task_name: str) -> dict[str, Any] | None:
        task = self.tasks.get(task_name)
        if not task or not task["exists"]:
            return None
        return {"task_name": task_name, "priority": task["priority"], "status": self._location(task_name)[0]}

    def _global_queue(self) -> dict[str, Any]:
        return {
            "version": 2,
            "active": self._entry(self.active) if self.active else None,
            "draining": self._entry(self.draining) if self.draining else None,
            "queue": [self._entry(name) for name in self.queue if self._entry(name)],
        }

    def _task_queue(self, task_name: str) -> dict[str, Any]:
        location, position = self._location(task_name)
        return {
            "task_name": task_name,
            "location": location,
            "position": position,
            "entry": self._entry(task_name) if location != "not_found" else None,
            "active": self._entry(self.active) if self.active else None,
        }

    @staticmethod
    def _dataset_names(config: dict[str, Any]) -> list[str]:
        return [str(item.get("dataset_name") or "") for item in config.get("datasets") or [] if isinstance(item, dict)]

    def _snapshot(self, task_name: str, datasets: list[str] | None = None) -> dict[str, Any]:
        task = self.tasks.get(task_name)
        if not task or not task["exists"]:
            return {
                "task_name": task_name, "task_exists": False, "config_file_exists": False,
                "dag_file_exists": False, "dag_id": f"batch_pipeline_universal_{task_name}",
                "priority": None, "task_exclusive": True, "available_datasets": [],
                "selected_datasets": list(datasets or []), "queue": self._task_queue(task_name),
                "containers": [], "gpu_reservations": [], "airflow_dag_exists": False,
                "airflow_runs": [], "errors": {},
            }
        selected = set(datasets or [])
        runs = [dict(r) for r in task["runs"] if not selected or r["dataset_name"] in selected]
        return {
            "task_name": task_name, "task_exists": True, "config_file_exists": True,
            "dag_file_exists": True, "dag_id": task["dag_id"], "priority": task["priority"],
            "task_exclusive": bool(task["config"].get("task_exclusive", True)),
            "available_datasets": self._dataset_names(task["config"]), "selected_datasets": list(datasets or []),
            "queue": self._task_queue(task_name), "containers": list(task.get("containers") or []),
            "gpu_reservations": list(task.get("gpu_reservations") or []), "airflow_dag_exists": True,
            "airflow_runs": runs, "errors": {},
        }

    def _sort_queue(self):
        self.queue = sorted(dict.fromkeys(self.queue), key=lambda name: (int(self.tasks[name]["priority"]), name))

    def _maybe_schedule(self):
        if self.active or self.draining or not self.queue:
            return
        self._sort_queue()
        self.active = self.queue.pop(0)
        self.tasks[self.active]["lifecycle"] = "active"

    def _maybe_request_preemption(self, candidate: str):
        if not self.active or candidate == self.active:
            return
        active_priority = int(self.tasks[self.active]["priority"])
        candidate_priority = int(self.tasks[candidate]["priority"])
        if candidate_priority < active_priority:
            old = self.active
            self.draining = old
            self.active = None
            self.tasks[old]["lifecycle"] = "draining"
            if candidate not in self.queue:
                self.queue.append(candidate)
            self._sort_queue()

    def complete_stage_boundary(self, task_name: str) -> None:
        if self.draining != task_name:
            raise RuntimeError(f"Task is not draining: {task_name}")
        task = self.tasks[task_name]
        task["checkpoint"] = "segment"
        task["lifecycle"] = "preempted"
        self.draining = None
        if task_name not in self.queue:
            self.queue.append(task_name)
        self._sort_queue()
        self._maybe_schedule()
        self.generation += 1

    def complete_task(self, task_name: str) -> None:
        if self.active != task_name:
            raise RuntimeError(f"Task is not active: {task_name}")
        task = self.tasks[task_name]
        for run in task["runs"]:
            if run["state"] in {"queued", "running"}:
                run["state"] = "success"
        task["lifecycle"] = "success"
        self.active = None
        # A previously preempted task gets a recovery run when it becomes active again.
        if self.queue:
            self._sort_queue()
            next_name = self.queue.pop(0)
            self.active = next_name
            nxt = self.tasks[next_name]
            if nxt.get("lifecycle") == "preempted":
                ds = self._dataset_names(nxt["config"])[0]
                nxt["runs"].append({"run_id": f"recovery_{len(nxt['runs'])+1}", "dataset_name": ds, "state": "running", "recovery": True})
            nxt["lifecycle"] = "active"
        self.generation += 1

    def _gpu_pool(self) -> dict[str, Any]:
        devices = []
        for gpu_id in self.gpu_runtime.list_devices():
            info = self.gpu_runtime.get_memory_info(gpu_id)
            devices.append({"gpu_id": gpu_id, "total_mb": info.total_mb, "used_mb": info.used_mb, "free_mb": info.free_mb})
        reservations = []
        for task in self.tasks.values():
            reservations.extend(task.get("gpu_reservations") or [])
        return {"devices": devices, "reservations": reservations}

    def _detail(self, task_name: str) -> dict[str, Any]:
        task = self.tasks.get(task_name)
        if not task or not task["exists"]:
            raise RuntimeError(f"Task not found: {task_name}")
        config = task["config"]
        return {
            "task_name": task_name, "dag_id": task["dag_id"],
            "priority": {"priority": task["priority"], "task_type": config.get("task_type", "")},
            "gpu_stage_memory_mb": dict(config.get("gpu_stage_memory_mb") or {}),
            "pipeline_stages": config.get("pipeline_stages") or [],
            "datasets": config.get("datasets") or [], "airflow_runs": [dict(r) for r in task["runs"]],
            "queue": self._task_queue(task_name),
        }

    async def execute(self, calls: list[ToolCallSpec]) -> list[ToolObservation]:
        self.calls.extend(calls)
        out: list[ToolObservation] = []
        for call in calls:
            try:
                data = self._execute_one(call.name, dict(call.arguments))
                out.append(ToolObservation(tool_name=call.name, arguments=call.arguments, ok=True, data=data))
            except Exception as exc:
                out.append(ToolObservation(tool_name=call.name, arguments=call.arguments, ok=False, error=str(exc)))
        return out

    def _execute_one(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "get_platform_health":
            return {"airflow": {"ok": True, "mode": "local-scenario"}, "gpu": {"ok": True}, "queue": {"ok": True}}
        if name == "list_tasks":
            return {"count": sum(1 for t in self.tasks.values() if t["exists"]), "tasks": [self._detail(n) for n, t in self.tasks.items() if t["exists"]]}
        if name == "get_queue_state":
            return self._task_queue(str(args.get("task_name") or "")) if args.get("task_name") else self._global_queue()
        if name == "get_gpu_pool":
            return self._gpu_pool()
        if name == "get_task_detail":
            return self._detail(str(args["task_name"]))
        if name == "inspect_task_containers":
            task = self.tasks.get(str(args["task_name"])) or {}
            return {"containers": list(task.get("containers") or [])}
        if name == "get_stage_logs":
            task = self.tasks.get(str(args["task_name"])) or {}
            return {"logs": list(task.get("logs") or [])}
        if name == "diagnose_task":
            task_name = str(args["task_name"])
            task = self.tasks.get(task_name)
            if not task:
                raise RuntimeError(f"Task not found: {task_name}")
            runs = task["runs"]
            latest = dict(runs[-1]) if runs else {}
            instances = []
            if latest.get("state") == "failed":
                instances.append({"task_id": "run_segment", "state": "failed"})
            return {
                "queue": self._task_queue(task_name),
                "airflow": {"latest_run": latest, "task_instances": instances},
                "containers": list(task.get("containers") or []),
                "gpu_reservations": list(task.get("gpu_reservations") or []),
                "gpu_devices": self._gpu_pool()["devices"], "errors": [], "evidence_complete": True,
            }
        if name == "validate_task_spec":
            validate_config(dict(args["config"]))
            return {"ok": True, "task_prefix": args["task_prefix"]}
        if name == "get_write_precondition":
            return self._precondition(str(args.get("task_name") or ""))
        if name == "get_action_verification_snapshot":
            return self._snapshot(str(args["task_name"]), list(args.get("datasets") or []))
        if name in {"submit_task", "resume_task", "set_task_priority", "stop_task", "delete_task"}:
            self._assert_precondition(dict(args.get("precondition") or {}))
        if name == "submit_task":
            config = dict(args["config"])
            validate_config(config)
            self.counter += 1
            task_name = f"{args['task_prefix']}_e2e_{self.counter:03d}"
            config_sha = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
            runs = [{"run_id": f"run_{idx+1}", "dataset_name": ds, "state": "running" if idx == 0 else "queued"} for idx, ds in enumerate(self._dataset_names(config))]
            self.tasks[task_name] = {
                "exists": True, "config": config, "config_sha256": config_sha,
                "priority": int(normalize_task_priority_config(config)["priority"]), "dag_id": f"batch_pipeline_universal_{task_name}",
                "runs": runs, "containers": [], "gpu_reservations": [], "logs": [], "lifecycle": "queued", "checkpoint": "",
            }
            if self.active is None and self.draining is None:
                self.active = task_name
                self.tasks[task_name]["lifecycle"] = "active"
            else:
                self.queue.append(task_name)
                self._sort_queue()
                self._maybe_request_preemption(task_name)
            self.generation += 1
            return {"ok": True, "action": name, "result": {"task_name": task_name, "triggered": len(runs)}}
        if name == "set_task_priority":
            task_name = str(args["task_name"])
            task = self.tasks[task_name]
            task["priority"] = int(args["priority"])
            task["config"]["priority"] = int(args["priority"])
            task["config_sha256"] = hashlib.sha256(json.dumps(task["config"], sort_keys=True).encode()).hexdigest()
            if task_name in self.queue:
                self._sort_queue()
                self._maybe_request_preemption(task_name)
            self.generation += 1
            return {"ok": True, "action": name, "result": {"task_name": task_name, "priority": task["priority"]}}
        if name == "stop_task":
            task_name = str(args["task_name"])
            task = self.tasks[task_name]
            for run in task["runs"]:
                if run["state"] in {"running", "queued"}:
                    run["state"] = "failed"
            task["containers"] = []
            task["gpu_reservations"] = []
            self.queue = [n for n in self.queue if n != task_name]
            if self.active == task_name:
                self.active = None
            if self.draining == task_name:
                self.draining = None
            task["lifecycle"] = "stopped"
            self._maybe_schedule()
            self.generation += 1
            return {"ok": True, "action": name, "result": {"task_name": task_name}}
        if name == "resume_task":
            task_name = str(args["task_name"])
            task = self.tasks[task_name]
            selected = list(args.get("datasets") or [])
            if not selected:
                selected = [r["dataset_name"] for r in task["runs"] if r["state"] == "failed"]
            for ds in selected:
                task["runs"].append({"run_id": f"resume_{len(task['runs'])+1}", "dataset_name": ds, "state": "queued"})
            if self.active != task_name and task_name not in self.queue:
                self.queue.append(task_name)
            self._maybe_schedule()
            self.generation += 1
            return {"ok": True, "action": name, "result": {"task_name": task_name, "triggered": len(selected)}}
        if name == "delete_task":
            task_name = str(args["task_name"])
            task = self.tasks[task_name]
            task["exists"] = False
            task["containers"] = []
            task["gpu_reservations"] = []
            self.queue = [n for n in self.queue if n != task_name]
            if self.active == task_name:
                self.active = None
            if self.draining == task_name:
                self.draining = None
            self._maybe_schedule()
            self.generation += 1
            return {"ok": True, "action": name, "result": {"task_name": task_name}}
        raise RuntimeError(f"Unsupported local scenario tool: {name}")


def _run_mock_stage(repo_root: Path, root: Path) -> tuple[bool, str]:
    dataset = "clip_001"
    cmd = [
        sys.executable, str(repo_root / "scripts" / "mock_stage.py"), "--stage", "segment",
        "--dataset-path", str(root), "--dataset-name", dataset, "--duration-sec", "0", "--result", "success",
    ]
    stage = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    if stage.returncode != 0:
        return False, stage.stderr or stage.stdout
    validate = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "validate_json.py"),
            "--root-dir",
            str(root),
            "--dataset",
            dataset,
            "--task-suffix",
            "segment",
            "--min-date",
            datetime.now(timezone.utc).date().isoformat(),
        ],
        capture_output=True, text=True, timeout=10,
    )
    return validate.returncode == 0, (validate.stdout or validate.stderr).strip()


async def _run(root: Path) -> LocalE2EResult:
    repo_root = Path(__file__).resolve().parents[1]
    state = root / "state"
    gpu = SimulatedGPURuntime(state / "gpu_simulator.json", fallback_to_os_processes=False)
    gpu.initialize([
        {"id": "0", "total_memory_mb": 48000, "external_used_mb": 0},
        {"id": "1", "total_memory_mb": 48000, "external_used_mb": 0},
    ])
    client = LocalScenarioToolClient(gpu)
    trace_store = TraceStore(state / "traces", state / "audit" / "audit.jsonl")
    recorder = TraceRecorder(trace_store)
    approval_store = ApprovalStore(state / "approvals", ttl_sec=300)
    agent = build_agent_runtime(
        "sequential", HeuristicReadOnlyModel(), client, ConversationStore(state / "sessions"),
        max_tool_calls=6, knowledge_retriever=None, task_planning_service=TaskPlanningService.from_env(),
        approval_store=approval_store, action_verifier=ActionVerifier(client, attempts=2, interval_sec=0), trace_recorder=recorder,
    )
    steps: list[E2EStep] = []

    mock_ok, mock_detail = _run_mock_stage(repo_root, root / "datasets")
    steps.append(E2EStep(name="mock_stage_validate", ok=mock_ok, detail=mock_detail))

    first = await agent.run("提交一个 reprocess 任务，处理 /tmp/e2e/record_a，最多并发 2 个 Clip", thread_id="e2e")
    first_ok = bool(first.approval_required and first.approval_id and first.task_plan and first.task_plan.get("valid"))
    steps.append(E2EStep(name="submit_plan_and_approval", ok=first_ok, data={"approval_id": first.approval_id or ""}))
    if not first_ok:
        return LocalE2EResult(ok=False, steps=steps, artifacts_root=str(root))
    executed = await agent.approve(first.approval_id)
    first_task = str((executed.execution_result or {}).get("result", {}).get("task_name") or "")
    steps.append(E2EStep(name="submit_execute_verify", ok=executed.status == "executed" and bool(first_task), data={"task_name": first_task, "status": executed.status}))

    status = await agent.run(f"{first_task} 现在是什么状态？", thread_id="e2e")
    steps.append(E2EStep(name="read_after_submit", ok="active" in status.summary.lower(), detail=status.summary))

    gpu.set_external_used_mb("0", 30000)
    gpu.set_external_used_mb("1", 31000)
    diag = await agent.run(f"{first_task} 的 segment 为什么拿不到 GPU？", thread_id="e2e")
    diag_ok = "enough free memory" in (diag.root_cause or "").lower()
    steps.append(E2EStep(name="gpu_diagnosis", ok=diag_ok, detail=diag.root_cause or ""))

    second = await agent.run("提交一个 release 任务，处理 /tmp/e2e/release_a，最多并发 1 个 Clip", thread_id="e2e")
    second_exec = await agent.approve(second.approval_id) if second.approval_id else None
    second_task = str(((second_exec.execution_result if second_exec else {}) or {}).get("result", {}).get("task_name") or "")
    preempt_requested = bool(second_exec and second_exec.status == "executed" and client.draining == first_task and second_task in client.queue)
    steps.append(E2EStep(name="high_priority_submit_soft_preemption", ok=preempt_requested, data={"draining": client.draining or "", "queued": list(client.queue)}))

    if preempt_requested:
        client.complete_stage_boundary(first_task)
        boundary_ok = client.active == second_task and first_task in client.queue and client.tasks[first_task].get("checkpoint") == "segment"
    else:
        boundary_ok = False
    steps.append(E2EStep(name="stage_boundary_switch", ok=boundary_ok, data={"active": client.active or "", "checkpoint": client.tasks.get(first_task, {}).get("checkpoint", "")}))

    if boundary_ok:
        client.complete_task(second_task)
        recovery_runs = [r for r in client.tasks[first_task]["runs"] if r.get("recovery")]
        recovery_ok = client.active == first_task and bool(recovery_runs)
    else:
        recovery_runs = []
        recovery_ok = False
    steps.append(E2EStep(name="recovery_after_high_priority_finish", ok=recovery_ok, data={"active": client.active or "", "recovery_runs": recovery_runs}))

    # Validate a real guarded mutation after the scheduling scenario.
    priority = await agent.run(f"把 {first_task} 的优先级改成 5", thread_id="e2e")
    priority_exec = await agent.approve(priority.approval_id) if priority.approval_id else None
    priority_ok = bool(priority_exec and priority_exec.status == "executed" and client.tasks[first_task]["priority"] == 5)
    steps.append(E2EStep(name="priority_hitl_precondition_verify", ok=priority_ok, data={"status": priority_exec.status if priority_exec else "", "priority": client.tasks[first_task]["priority"]}))

    audits = trace_store.load_audit()
    trace_files = list((state / "traces").glob("*.jsonl"))
    steps.append(E2EStep(name="trace_audit_persisted", ok=bool(audits and trace_files), data={"traces": len(trace_files), "audits": len(audits)}))
    return LocalE2EResult(
        ok=all(step.ok for step in steps), steps=steps,
        trace_count=len(trace_files), audit_count=len(audits), artifacts_root=str(root),
    )


def run_local_e2e(root: str | Path | None = None) -> LocalE2EResult:
    if root is not None:
        path = Path(root)
        path.mkdir(parents=True, exist_ok=True)
        return asyncio.run(_run(path))
    with tempfile.TemporaryDirectory(prefix="dataops_agent_e2e_") as td:
        result = asyncio.run(_run(Path(td)))
        result.artifacts_root = ""
        return result
