"""Deterministic V2.0 READ/WRITE tool catalog and injected facade handlers."""

from __future__ import annotations

from ..platform.facade import ReadFacade
from .metadata import Idempotency, RiskLevel, ToolKind, ToolSpec
from .registry import ToolRegistry


def build_read_registry(facade: ReadFacade) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="get_task_detail",
            kind=ToolKind.READ,
            risk=RiskLevel.LOW,
            schema={
                "type": "object",
                "properties": {"task_name": {"type": "string", "nullable": True}},
                "required": ["task_name"],
            },
            parallel_safe=True,
            idempotency=Idempotency.SAFE_RETRY,
        ),
        facade.get_task_detail,
    )
    registry.register(
        ToolSpec(
            name="get_gpu_pool",
            kind=ToolKind.READ,
            risk=RiskLevel.LOW,
            schema={
                "type": "object",
                "properties": {},
                "required": [],
            },
            parallel_safe=True,
            idempotency=Idempotency.SAFE_RETRY,
        ),
        facade.get_gpu_pool,
    )
    registry.register(
        ToolSpec(
            name="search_knowledge",
            kind=ToolKind.READ,
            risk=RiskLevel.LOW,
            schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer"},
                },
                "required": ["query"],
            },
            parallel_safe=True,
            idempotency=Idempotency.SAFE_RETRY,
        ),
        facade.search_knowledge,
    )
    registry.register(
        ToolSpec(
            name="get_queue_state",
            kind=ToolKind.READ,
            risk=RiskLevel.LOW,
            schema={
                "type": "object",
                "properties": {"task_name": {"type": "string", "nullable": True}},
                "required": [],
            },
            parallel_safe=True,
            idempotency=Idempotency.SAFE_RETRY,
        ),
        facade.get_queue_state,
    )
    registry.register(
        ToolSpec(
            name="diagnose_task",
            kind=ToolKind.READ,
            risk=RiskLevel.LOW,
            schema={
                "type": "object",
                "properties": {"task_name": {"type": "string"}},
                "required": ["task_name"],
            },
            parallel_safe=False,
            idempotency=Idempotency.SAFE_RETRY,
        ),
        facade.diagnose_task,
    )
    registry.seal()
    return registry


def build_full_registry(facade) -> ToolRegistry:
    """Build the complete V2 READ+WRITE catalog; every WRITE still requires approval."""
    registry = ToolRegistry()
    # Re-register the five READ tools locally to keep one sealed catalog hash.
    read = build_read_registry(facade)
    for spec in read.catalog():
        registry.register(spec, read.handler(spec.name))
    write_specs = (
        ToolSpec(
            name="resume_task", kind=ToolKind.WRITE, risk=RiskLevel.MEDIUM,
            schema={"type":"object","properties":{"task_name":{"type":"string"}},"required":["task_name"]},
            requires_precondition=True, verification="ACTION_AND_GOAL", verification_reads=("get_task_detail",),
            idempotency=Idempotency.RECONCILE_BEFORE_RETRY,
        ),
        ToolSpec(
            name="submit_task", kind=ToolKind.WRITE, risk=RiskLevel.HIGH,
            schema={"type":"object","properties":{"task_name":{"type":"string"},"config":{"type":"object"}},"required":["task_name"]},
            requires_precondition=True, verification="ACTION_AND_GOAL", verification_reads=("get_task_detail",),
            idempotency=Idempotency.RECONCILE_BEFORE_RETRY,
        ),
        ToolSpec(
            name="stop_task", kind=ToolKind.WRITE, risk=RiskLevel.HIGH,
            schema={"type":"object","properties":{"task_name":{"type":"string"}},"required":["task_name"]},
            requires_precondition=True, verification="ACTION_AND_GOAL", verification_reads=("get_task_detail",),
            idempotency=Idempotency.RECONCILE_BEFORE_RETRY,
        ),
        ToolSpec(
            name="delete_task", kind=ToolKind.WRITE, risk=RiskLevel.HIGH,
            schema={"type":"object","properties":{"task_name":{"type":"string"}},"required":["task_name"]},
            requires_precondition=True, verification="ACTION", verification_reads=("get_task_detail",),
            idempotency=Idempotency.NO_RETRY,
        ),
        ToolSpec(
            name="set_task_priority", kind=ToolKind.WRITE, risk=RiskLevel.HIGH,
            schema={"type":"object","properties":{"task_name":{"type":"string"},"priority":{"type":"integer"}},"required":["task_name","priority"]},
            requires_precondition=True, verification="ACTION_AND_GOAL", verification_reads=("get_task_detail",),
            idempotency=Idempotency.RECONCILE_BEFORE_RETRY,
        ),
    )
    for spec in write_specs:
        registry.register(spec, getattr(facade, spec.name))
    registry.seal()
    return registry
