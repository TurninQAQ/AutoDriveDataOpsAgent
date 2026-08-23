"""Production host assembly and deterministic health/readiness checks.

The host is deliberately thin: it constructs the frozen Runtime graph and
exposes only invoke/resume/reconcile to callers.  It never calls a Tool
directly and never makes a semantic decision.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from .config import RuntimeConfig, ensure_runtime_layout
from .agent.runtime import SystemContext, build_system_context
from .platform.mcp import MCPPlatformFacade
from .providers.qwen import QwenProvider


def build_production_context(config: RuntimeConfig | None = None) -> SystemContext:
    """Build one production context using configured Provider and MCP facade."""
    selected = config or RuntimeConfig.from_env()
    ensure_runtime_layout(selected)
    facade = MCPPlatformFacade(selected.platform)
    provider = QwenProvider(selected.provider)
    return build_system_context(
        provider=provider,
        read_facade=facade,
        durable_path=selected.persistence.sqlite_path,
        principles_path=selected.principles_path,
        environment=selected.environment,
        operator_id=selected.operator_id,
        trust_domain=selected.trust_domain,
        budgets=selected.budgets,
    )


def health(config: RuntimeConfig | None = None) -> dict[str, Any]:
    """Return liveness information without contacting external services."""
    selected = config or RuntimeConfig.from_env()
    return {
        "status": "ok",
        "service": "autodrive-dataops-agent-v2",
        "environment": selected.environment,
        "runtime_root": str(selected.persistence.runtime_root),
    }


def readiness(config: RuntimeConfig | None = None) -> dict[str, Any]:
    """Check local deterministic prerequisites; never performs a WRITE."""
    selected = config or RuntimeConfig.from_env()
    ensure_runtime_layout(selected)
    context = build_production_context(selected)
    writable = _sqlite_writable(selected.persistence.sqlite_path)
    catalog_hash = context.tool_registry.catalog_hash()
    checks = {
        "config": True,
        "tool_catalog_sealed": context.tool_registry.is_sealed,
        "tool_catalog_hash_stable": catalog_hash == context.tool_catalog_hash,
        "provider_configured": bool(selected.provider.endpoint and selected.provider.model and selected.provider.api_key_env),
        "platform_configured": bool(selected.platform.endpoint),
        "sqlite_writable": writable,
        "runtime_version": context.runtime_version,
    }
    return {
        "status": "ready" if all(value is True for key, value in checks.items() if key != "runtime_version") else "not_ready",
        "checks": checks,
    }


def _sqlite_writable(path: Path) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS _autodrive_readiness_probe (id INTEGER PRIMARY KEY)")
            connection.execute("DROP TABLE _autodrive_readiness_probe")
        return True
    except (OSError, sqlite3.Error):
        return False


def pending_approval(*, thread_id: str, context: SystemContext) -> dict[str, Any] | None:
    """Return a safe pending approval projection from Runtime checkpoint state."""
    state = context.checkpointer.load(thread_id)
    if state is None:
        return None
    current = state["current_request"]
    pending = current.pending_interrupt
    if pending is None:
        return None
    return {
        "thread_id": thread_id,
        "request_id": current.identity.request_id,
        "approval_request_id": pending.approval_request_id,
        "transaction_id": pending.transaction_id,
        "fingerprint": pending.fingerprint,
        "tool_name": pending.tool_name,
        "bound_goal_ids": tuple(pending.bound_goal_ids),
        "risk": pending.risk,
    }
