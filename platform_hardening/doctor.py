from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from platform_agent.settings import AgentSettings
from platform_core.gateways.gpu_runtime import create_gpu_runtime_from_env
from platform_core.settings import PlatformSettings


class DoctorCheck(BaseModel):
    name: str
    status: Literal["ok", "warning", "error", "skipped"]
    detail: str = ""
    required_for: list[str] = Field(default_factory=list)


class DoctorReport(BaseModel):
    ready_dependency_light: bool
    ready_full_runtime: bool
    checks: list[DoctorCheck]
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _writable_dir(path: Path) -> tuple[bool, str]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / f".doctor_write_{os.getpid()}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True, str(path)
    except Exception as exc:
        return False, f"{path}: {exc}"


def _module_check(module: str, required_for: str) -> DoctorCheck:
    found = importlib.util.find_spec(module) is not None
    return DoctorCheck(
        name=f"python:{module}",
        status="ok" if found else "warning",
        detail="installed" if found else "not installed",
        required_for=[required_for],
    )


def run_doctor(
    platform_settings: PlatformSettings | None = None,
    agent_settings: AgentSettings | None = None,
    *,
    env: dict[str, str] | None = None,
) -> DoctorReport:
    env = dict(os.environ if env is None else env)
    ps = platform_settings or PlatformSettings.from_env(env)
    # AgentSettings currently reads process env. When a custom env is used by tests,
    # callers can pass the settings explicitly to keep this function side-effect free.
    aset = agent_settings or AgentSettings.from_env(ps)
    checks: list[DoctorCheck] = []

    checks.append(DoctorCheck(name="python", status="ok", detail=sys.version.split()[0], required_for=["dependency-light", "full-runtime"]))
    for path, name in (
        (ps.state_dir, "state_dir"),
        (ps.task_config_root, "task_config_root"),
        (aset.session_dir, "agent_session_dir"),
        (aset.approval_dir, "approval_dir"),
        (aset.trace_dir, "trace_dir"),
        (aset.audit_file.parent, "audit_dir"),
    ):
        ok, detail = _writable_dir(path)
        checks.append(DoctorCheck(name=name, status="ok" if ok else "error", detail=detail, required_for=["dependency-light", "full-runtime"]))

    repo_root = Path(__file__).resolve().parents[1]
    mock_stage = repo_root / "scripts" / "mock_stage.py"
    planning_defaults = repo_root / "config" / "task_planning_defaults.yaml"
    knowledge_dir = aset.knowledge_source_dir
    for name, path in (
        ("mock_stage", mock_stage),
        ("task_planning_defaults", planning_defaults),
        ("knowledge_source", knowledge_dir),
    ):
        checks.append(DoctorCheck(name=name, status="ok" if path.exists() else "error", detail=str(path), required_for=["dependency-light", "full-runtime"]))

    gpu_mode = env.get("PLATFORM_GPU_RUNTIME", os.environ.get("PLATFORM_GPU_RUNTIME", "nvidia")).strip().lower()
    if gpu_mode in {"sim", "simulated", "mock"}:
        try:
            runtime = create_gpu_runtime_from_env(env)
            devices = runtime.list_devices()
            checks.append(DoctorCheck(name="gpu_runtime", status="ok" if devices else "error", detail=f"simulated devices={devices}", required_for=["dependency-light", "full-runtime"]))
        except Exception as exc:
            checks.append(DoctorCheck(name="gpu_runtime", status="error", detail=f"simulated runtime failed: {exc}", required_for=["dependency-light", "full-runtime"]))
    else:
        nvidia = shutil.which("nvidia-smi")
        checks.append(DoctorCheck(name="gpu_runtime", status="ok" if nvidia else "warning", detail=nvidia or "nvidia-smi not found; use PLATFORM_GPU_RUNTIME=simulated for local development", required_for=["full-runtime"]))

    docker_bin = shutil.which("docker")
    checks.append(DoctorCheck(name="docker", status="ok" if docker_bin else "warning", detail=docker_bin or "docker binary not found", required_for=["full-runtime"]))

    checks.extend([
        _module_check("airflow", "full-runtime"),
        _module_check("mcp", "full-runtime"),
        _module_check("langgraph", "full-runtime"),
    ])
    provider = aset.provider
    if provider == "auto":
        if env.get("DASHSCOPE_API_KEY") and env.get("DASHSCOPE_OPENAI_BASE_URL"):
            provider = "qwen"
        elif env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            provider = "gemini"
        else:
            provider = "openai"
    if provider in {"qwen", "dashscope", "aliyun", "alibaba"}:
        checks.append(_module_check("openai", "full-runtime"))
        key_present = bool(env.get("DASHSCOPE_API_KEY") or os.environ.get("DASHSCOPE_API_KEY"))
        base_present = bool(env.get("DASHSCOPE_OPENAI_BASE_URL") or os.environ.get("DASHSCOPE_OPENAI_BASE_URL"))
        checks.append(DoctorCheck(name="dashscope_api_key", status="ok" if key_present else "warning", detail="configured" if key_present else "DASHSCOPE_API_KEY not set", required_for=["full-runtime"]))
        checks.append(DoctorCheck(name="dashscope_openai_base_url", status="ok" if base_present else "warning", detail="configured" if base_present else "DASHSCOPE_OPENAI_BASE_URL not set", required_for=["full-runtime"]))
    elif provider in {"gemini", "google", "google-genai", "google_genai"}:
        checks.append(_module_check("google.genai", "full-runtime"))
        key_present = bool(env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
        checks.append(DoctorCheck(name="gemini_api_key", status="ok" if key_present else "warning", detail="configured" if key_present else "GEMINI_API_KEY/GOOGLE_API_KEY not set", required_for=["full-runtime"]))
    else:
        checks.append(_module_check("langchain_openai", "full-runtime"))
    if aset.knowledge_embedding_provider in {"qwen", "dashscope", "aliyun", "alibaba"}:
        checks.append(_module_check("requests", "full-runtime"))
        key_present = bool(env.get("DASHSCOPE_API_KEY") or os.environ.get("DASHSCOPE_API_KEY"))
        base_present = bool(env.get("DASHSCOPE_API_BASE_URL") or os.environ.get("DASHSCOPE_API_BASE_URL"))
        checks.append(DoctorCheck(name="dashscope_embedding_key", status="ok" if key_present else "warning", detail="configured" if key_present else "DASHSCOPE_API_KEY not set", required_for=["full-runtime"]))
        checks.append(DoctorCheck(name="dashscope_embedding_base_url", status="ok" if base_present else "warning", detail="configured" if base_present else "DASHSCOPE_API_BASE_URL not set", required_for=["full-runtime"]))
    elif aset.knowledge_embedding_provider in {"gemini", "google", "google-genai", "google_genai"}:
        checks.append(_module_check("google.genai", "full-runtime"))
        key_present = bool(env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
        checks.append(DoctorCheck(name="gemini_embedding_key", status="ok" if key_present else "warning", detail="configured" if key_present else "GEMINI_API_KEY/GOOGLE_API_KEY not set", required_for=["full-runtime"]))

    # Dependency-light path intentionally does not require these optional modules.
    dep_errors = [c for c in checks if c.status == "error" and "dependency-light" in c.required_for]
    full_missing = [c for c in checks if c.status in {"warning", "error"} and "full-runtime" in c.required_for]
    errors = [f"{c.name}: {c.detail}" for c in checks if c.status == "error"]
    warnings = [f"{c.name}: {c.detail}" for c in checks if c.status == "warning"]
    return DoctorReport(
        ready_dependency_light=not dep_errors,
        ready_full_runtime=not full_missing,
        checks=checks,
        errors=errors,
        warnings=warnings,
    )
