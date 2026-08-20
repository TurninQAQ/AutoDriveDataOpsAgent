"""Canonical read-only Tool Catalog shared by production adapters and eval fixtures."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


_CATALOG: tuple[dict[str, Any], ...] = (
    {
        "name": "get_platform_health",
        "description": "Inspect current platform component health and resource availability.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "list_tasks",
        "description": "List current generated business tasks with priority and queue information.",
        "input_schema": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "default": 100}},
            "required": [],
        },
    },
    {
        "name": "get_task_detail",
        "description": "Inspect the current config, queue status and recent DagRuns for one named business task.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_name": {"type": "string", "description": "Concrete business task identity."},
                "include_airflow_runs": {"type": "boolean", "default": True},
                "run_limit": {"type": "integer", "default": 20},
            },
            "required": ["task_name"],
        },
    },
    {
        "name": "get_queue_state",
        "description": "Inspect the current global priority queue or a named task's queue position.",
        "input_schema": {
            "type": "object",
            "properties": {"task_name": {"type": "string", "default": ""}},
            "required": [],
        },
    },
    {
        "name": "get_gpu_pool",
        "description": (
            "Inspect current/live GPU runtime state, including device memory, availability and active "
            "reservations. Use for what GPU resources or reservations exist now, or current GPU allocation "
            "problems. Do not use it to explain platform concepts, architecture or reservation rules."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"cleanup_dead": {"type": "boolean", "default": True}},
            "required": [],
        },
    },
    {
        "name": "inspect_task_containers",
        "description": "Inspect current Docker containers belonging to a concrete task and optional datasets.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_name": {"type": "string", "description": "Concrete business task identity."},
                "datasets": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["task_name"],
        },
    },
    {
        "name": "get_stage_logs",
        "description": "Retrieve current or recent logs for a named task's failed, running or selected Stage.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_name": {"type": "string", "description": "Concrete business task identity."},
                "dataset_name": {"type": "string", "default": ""},
                "stage": {"type": "string", "default": ""},
                "tail_lines": {"type": "integer", "default": 200},
            },
            "required": ["task_name"],
        },
    },
    {
        "name": "diagnose_task",
        "description": "Aggregate current queue, Airflow, Docker and GPU evidence for one concrete task without LLM inference.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_name": {"type": "string", "description": "Concrete business task identity."},
                "dataset_name": {"type": "string", "default": ""},
            },
            "required": ["task_name"],
        },
    },
    {
        "name": "search_knowledge",
        "description": (
            "Search static platform documentation and runbooks for definitions, architecture, mechanisms, "
            "policies and operating rules such as GPU Reservation, soft preemption and recovery. Use for "
            "what-is, how-it-works and platform-rule questions. It does not return current runtime state."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Static platform knowledge search query."},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 100, "default": 5},
            },
            "required": ["query"],
        },
    },
)

CANONICAL_READ_ONLY_TOOL_CATALOG = tuple(item["name"] for item in _CATALOG)


def build_read_only_tool_catalog(*, knowledge_enabled: bool = True) -> list[dict[str, Any]]:
    """Return fresh production-equivalent read-only Tool metadata."""

    return [
        deepcopy(item)
        for item in _CATALOG
        if knowledge_enabled or item["name"] != "search_knowledge"
    ]


__all__ = ["CANONICAL_READ_ONLY_TOOL_CATALOG", "build_read_only_tool_catalog"]
