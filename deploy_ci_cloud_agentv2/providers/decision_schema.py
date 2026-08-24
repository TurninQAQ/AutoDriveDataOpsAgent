"""Canonical JSON Schema for the untrusted AgentDecision proposal.

This schema is a transport contract only.  The dataclass parser and the
Runtime-owned DecisionIngress validator remain authoritative after a model
response is received.
"""

from __future__ import annotations

from typing import Any


READ_TOOLS = (
    "get_task_detail",
    "get_gpu_pool",
    "search_knowledge",
    "get_queue_state",
    "diagnose_task",
)
WRITE_TOOLS = (
    "resume_task",
    "submit_task",
    "stop_task",
    "delete_task",
    "set_task_priority",
)


def agent_decision_json_schema(*, require_goal_descriptor: bool) -> dict[str, Any]:
    """Return the exact V2 proposal schema for an OpenAI-compatible API."""
    descriptor = _goal_descriptor_schema()
    if not require_goal_descriptor:
        descriptor = {"anyOf": [descriptor, {"type": "null"}]}

    single = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "kind": {"const": "SINGLE_TOOL_CALL"},
            "proposed_goal_descriptor": descriptor,
            "call": _tool_call_schema((*READ_TOOLS, *WRITE_TOOLS)),
        },
        "required": ["kind", "proposed_goal_descriptor", "call"],
    }
    batch = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "kind": {"const": "READ_TOOL_BATCH"},
            "proposed_goal_descriptor": descriptor,
            "calls": {"type": "array", "minItems": 1, "maxItems": 3,
                      "items": _tool_call_schema(READ_TOOLS)},
        },
        "required": ["kind", "proposed_goal_descriptor", "calls"],
    }
    final = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "kind": {"const": "FINAL_CANDIDATE"},
            "proposed_goal_descriptor": descriptor,
            "response": {"type": "string", "minLength": 1},
            "referenced_goal_ids": {"type": "array", "items": {"type": "string", "minLength": 1}},
        },
        "required": ["kind", "proposed_goal_descriptor", "response", "referenced_goal_ids"],
    }
    return {"type": "object", "anyOf": [single, batch, final]}


def _goal_descriptor_schema() -> dict[str, Any]:
    goal_variants = [
        _goal("READ_TASK_STATE", {"target": _text()}),
        _goal("INSPECT_GPU"),
        _goal("INSPECT_QUEUE", {"target": {"anyOf": [_text(), {"type": "null"}]}}),
        _goal("EXPLAIN_KNOWLEDGE", {"topic": _text()}),
        _goal("DIAGNOSE_TASK", {"target": _text()}),
        _goal("RESUME_TASK", {"target": _text()}),
        _goal("STOP_TASK", {"target": _text()}),
        _goal("DELETE_TASK", {"target": _text()}),
        _goal("SET_TASK_PRIORITY", {"target": _text(), "priority": {"type": "integer"}}),
        _goal("SUBMIT_TASK", {"target": _text(), "config": {"type": "object"}}),
    ]
    return {"type": "object", "additionalProperties": False,
            "properties": {"descriptor_version": {"type": "integer", "minimum": 1},
                           "goals": {"type": "array", "minItems": 1, "items": {"anyOf": goal_variants}}},
            "required": ["descriptor_version", "goals"]}


def _goal(kind: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    properties = {"goal_id": _text(), "kind": {"const": kind}}
    if extra:
        properties.update(extra)
    return {"type": "object", "additionalProperties": False,
            "properties": properties, "required": list(properties)}


def _tool_call_schema(tool_names: tuple[str, ...]) -> dict[str, Any]:
    return {"anyOf": [{"type": "object", "additionalProperties": False,
                       "properties": {"call_id": _text(), "tool_name": {"const": name},
                                      "arguments": _arguments_schema(name)},
                       "required": ["call_id", "tool_name", "arguments"]}
            for name in tool_names]}


def _arguments_schema(tool_name: str) -> dict[str, Any]:
    if tool_name == "get_gpu_pool":
        properties, required = {}, []
    elif tool_name == "search_knowledge":
        properties, required = {"query": _text(), "top_k": {"type": "integer", "minimum": 1}}, ["query", "top_k"]
    elif tool_name == "get_queue_state":
        properties, required = {"task_name": {"anyOf": [_text(), {"type": "null"}]}}, ["task_name"]
    elif tool_name in {"get_task_detail", "diagnose_task", "resume_task", "stop_task", "delete_task"}:
        properties, required = {"task_name": _text()}, ["task_name"]
    elif tool_name == "set_task_priority":
        properties, required = {"task_name": _text(), "priority": {"type": "integer"}}, ["task_name", "priority"]
    elif tool_name == "submit_task":
        properties, required = {"task_name": _text(), "config": {"type": "object"}}, ["task_name", "config"]
    else:  # pragma: no cover
        raise ValueError(f"unknown canonical tool: {tool_name}")
    return {"type": "object", "additionalProperties": False,
            "properties": properties, "required": required}


def _text() -> dict[str, Any]:
    return {"type": "string", "minLength": 1}
