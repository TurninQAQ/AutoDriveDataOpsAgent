"""Small V2-local READ facade.

Production hosts can inject an adapter around their MCP/read transport.  Phase B
keeps this boundary local and dependency-free so the loop and tests do not need
the V1 runtime or an external platform.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, Protocol

from ..tools.runtime import ReadFailure


class ReadFacade(Protocol):
    def get_task_detail(self, task_name: str) -> Any: ...

    def get_gpu_pool(self) -> Any: ...

    def search_knowledge(self, query: str, top_k: int = 5) -> Any: ...

    def get_queue_state(self, task_name: str | None = None) -> Any: ...

    def diagnose_task(self, task_name: str) -> Any: ...


class InMemoryReadFacade:
    """Deterministic fixture facade and a useful offline host implementation."""

    def __init__(
        self,
        responses: Mapping[str, Any] | None = None,
        failures: Mapping[str, list[BaseException | None]] | None = None,
    ) -> None:
        self.responses = dict(responses or {})
        self.failures = {name: list(items) for name, items in (failures or {}).items()}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def _result(self, tool_name: str, arguments: dict[str, Any], default: Any) -> Any:
        self.calls.append((tool_name, dict(arguments)))
        pending = self.failures.get(tool_name, [])
        if pending:
            failure = pending.pop(0)
            if failure is not None:
                if isinstance(failure, ReadFailure):
                    raise failure
                raise failure
        return copy.deepcopy(self.responses.get(tool_name, default))

    def get_task_detail(self, task_name: str) -> Any:
        default = {"status": "NO_DATA", "task_name": task_name, "state": None}
        return self._result("get_task_detail", {"task_name": task_name}, default)

    def get_gpu_pool(self) -> Any:
        return self._result(
            "get_gpu_pool", {}, {"status": "NO_DATA", "devices": [], "reservations": []}
        )

    def search_knowledge(self, query: str, top_k: int = 5) -> Any:
        default = {
            "status": "EMPTY_RESULT",
            "query": query,
            "top_k": top_k,
            "results": [],
        }
        return self._result(
            "search_knowledge", {"query": query, "top_k": top_k}, default
        )

    def get_queue_state(self, task_name: str | None = None) -> Any:
        default = (
            {"status": "NO_DATA", "scope": "TASK", "task_name": task_name, "queue": []}
            if task_name is not None
            else {"status": "NO_DATA", "scope": "PLATFORM", "queue": []}
        )
        return self._result("get_queue_state", {"task_name": task_name}, default)

    def diagnose_task(self, task_name: str) -> Any:
        default = {"status": "NO_DATA", "task_name": task_name, "diagnosis": None}
        return self._result("diagnose_task", {"task_name": task_name}, default)
