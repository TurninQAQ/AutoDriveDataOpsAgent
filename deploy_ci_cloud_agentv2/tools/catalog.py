"""Phase B READ tool catalog and injected facade handlers."""

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
                "properties": {"task_name": {"type": "string"}},
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
                "properties": {"task_name": {"type": "string"}},
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
    return registry
