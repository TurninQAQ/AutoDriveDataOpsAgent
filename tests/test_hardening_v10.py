from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

from platform_agent.approval import ApprovalStore
from platform_agent.settings import AgentSettings
from platform_core.gateways.gpu_runtime import SimulatedGPURuntime
from platform_core.settings import PlatformSettings
from platform_hardening import run_doctor, run_local_e2e
from platform_observability.models import AuditRecord
from platform_observability.store import TraceStore


def test_trace_retention_prunes_old_and_caps_file_count(tmp_path: Path):
    trace_dir = tmp_path / "traces"
    audit = tmp_path / "audit" / "audit.jsonl"
    store = TraceStore(trace_dir, audit)
    trace_dir.mkdir(parents=True)
    now = time.time()
    files = []
    for idx, age_days in enumerate((30, 10, 2, 1, 0)):
        path = trace_dir / f"trace_{idx}.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        ts = now - age_days * 86400
        os.utime(path, (ts, ts))
        files.append(path)
    result = store.prune_traces(retention_days=14, max_files=2, now=now)
    remaining = sorted(p.name for p in trace_dir.glob("*.jsonl"))
    assert result["trace_files_deleted"] == 3
    assert remaining == ["trace_3.jsonl", "trace_4.jsonl"]


def test_audit_rotation_keeps_bounded_backups(tmp_path: Path):
    audit = tmp_path / "audit" / "audit.jsonl"
    store = TraceStore(tmp_path / "traces", audit)
    audit.parent.mkdir(parents=True)
    audit.write_text("x" * 200, encoding="utf-8")
    result = store.rotate_audit(max_bytes=100, backup_count=2)
    assert result["rotated"] is True
    assert audit.exists() and audit.read_text(encoding="utf-8") == ""
    assert (audit.parent / "audit.jsonl.1").is_file()
    audit.write_text("y" * 200, encoding="utf-8")
    store.rotate_audit(max_bytes=100, backup_count=2)
    assert (audit.parent / "audit.jsonl.2").is_file()



def test_audit_loader_reads_rotated_backups_in_chronological_order(tmp_path: Path):
    audit = tmp_path / "audit" / "audit.jsonl"
    store = TraceStore(tmp_path / "traces", audit)
    for idx in range(3):
        store.append_audit(AuditRecord(
            trace_id=f"t{idx}", kind="agent_request", started_at=float(idx), ended_at=float(idx)+0.1,
            latency_ms=100.0, status="ok", user_request=str(idx), response_summary="x" * 80,
        ))
        if idx < 2:
            store.rotate_audit(max_bytes=1, backup_count=3)
    assert [item.trace_id for item in store.load_audit()] == ["t0", "t1", "t2"]

def test_doctor_reports_dependency_light_ready_with_simulated_gpu(monkeypatch, tmp_path: Path):
    platform_home = tmp_path / "platform"
    airflow_home = platform_home / "airflow"
    sim_state = platform_home / "state" / "gpu.json"
    SimulatedGPURuntime(sim_state, fallback_to_os_processes=False).initialize([
        {"id": "0", "total_memory_mb": 48000, "external_used_mb": 0}
    ])
    monkeypatch.setenv("PLATFORM_HOME", str(platform_home))
    monkeypatch.setenv("AIRFLOW_HOME", str(airflow_home))
    monkeypatch.setenv("PLATFORM_GPU_RUNTIME", "simulated")
    monkeypatch.setenv("PLATFORM_GPU_SIM_STATE", str(sim_state))
    ps = PlatformSettings.from_env()
    aset = AgentSettings.from_env(ps)
    report = run_doctor(ps, aset)
    assert report.ready_dependency_light is True
    assert any(item.name == "gpu_runtime" and item.status == "ok" for item in report.checks)


def test_local_e2e_covers_agent_write_preemption_recovery_and_traces(tmp_path: Path):
    result = run_local_e2e(tmp_path / "e2e")
    assert result.ok is True
    names = {step.name for step in result.steps if step.ok}
    assert {
        "mock_stage_validate",
        "submit_execute_verify",
        "gpu_diagnosis",
        "high_priority_submit_soft_preemption",
        "stage_boundary_switch",
        "recovery_after_high_priority_finish",
        "priority_hitl_precondition_verify",
        "trace_audit_persisted",
    }.issubset(names)
    assert result.trace_count > 0 and result.audit_count > 0


def test_approval_claim_is_single_winner_under_concurrency(tmp_path: Path):
    store = ApprovalStore(tmp_path / "approvals", ttl_sec=300)
    item = store.create(
        thread_id="t", user_request="stop task", tool_name="stop_task",
        arguments={"task_name": "task_a"}, precondition={}, risk_level="high",
        impact_summary="stop",
    )
    outcomes: list[str] = []
    barrier = threading.Barrier(2)

    def claim():
        barrier.wait()
        try:
            store.claim_for_execution(item.approval_id)
            outcomes.append("claimed")
        except RuntimeError as exc:
            outcomes.append(str(exc))

    threads = [threading.Thread(target=claim) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert outcomes.count("claimed") == 1
    assert sum("not pending" in value for value in outcomes) == 1


def test_deploy_script_copies_hardening_package_and_platform_exports_retention():
    root = Path(__file__).resolve().parents[1]
    deploy = (root / "scripts" / "deploy_ci_cloud.sh").read_text(encoding="utf-8")
    platform = (root / "platform").read_text(encoding="utf-8")
    assert "platform_hardening" in deploy
    assert "AIRFLOW_PLATFORM_HARDENING_DIR" in deploy
    for key in (
        "PLATFORM_AGENT_TRACE_RETENTION_DAYS",
        "PLATFORM_AGENT_TRACE_MAX_FILES",
        "PLATFORM_AGENT_AUDIT_MAX_BYTES",
        "PLATFORM_AGENT_AUDIT_BACKUP_COUNT",
    ):
        assert key in platform
