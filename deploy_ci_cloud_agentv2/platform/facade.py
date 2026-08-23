"""Small V2-local READ facade.

Production hosts can inject an adapter around their MCP/read/write transport. V2.0
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
        # Absence is represented by omission.  ``state=None`` would be a
        # present-invalid field under the strict response contract.
        default = {"status": "NO_DATA", "task_name": task_name}
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
        # A missing diagnosis is absent data, not a nullable diagnosis field.
        default = {"status": "NO_DATA", "task_name": task_name}
        return self._result("diagnose_task", {"task_name": task_name}, default)


class WriteFacade(Protocol):
    def resume_task(self, task_name: str, *, precondition: Mapping[str, Any] | None = None) -> Any: ...
    def submit_task(self, task_name: str, config: Mapping[str, Any], *, precondition: Mapping[str, Any] | None = None) -> Any: ...
    def stop_task(self, task_name: str, *, precondition: Mapping[str, Any] | None = None) -> Any: ...
    def delete_task(self, task_name: str, *, precondition: Mapping[str, Any] | None = None) -> Any: ...
    def set_task_priority(self, task_name: str, priority: int, *, precondition: Mapping[str, Any] | None = None) -> Any: ...


class InMemoryPlatformFacade(InMemoryReadFacade):
    """Mutable deterministic platform fixture for the full V2 lifecycle."""

    def __init__(self, tasks: Mapping[str, Mapping[str, Any]] | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.tasks = {str(name): dict(value) for name, value in (tasks or {}).items()}
        self.mutation_count = 0
        self.mutation_failures: dict[str, list[BaseException | None]] = {}

    def set_mutation_failures(self, failures: Mapping[str, list[BaseException | None]]) -> None:
        self.mutation_failures = {name: list(items) for name, items in failures.items()}

    def _mutation_failure(self, tool_name: str) -> None:
        pending = self.mutation_failures.get(tool_name, [])
        if pending:
            failure = pending.pop(0)
            if failure is not None:
                raise failure

    def get_task_detail(self, task_name: str) -> Any:
        if task_name in self.tasks:
            task = copy.deepcopy(self.tasks[task_name])
            task.setdefault("task_name", task_name)
            task.setdefault("state", "RUNNING")
            task.setdefault("exists", True)
            task.setdefault("entity_version", str(task.get("revision", 1)))
            self.calls.append(("get_task_detail", {"task_name": task_name}))
            return task
        self.calls.append(("get_task_detail", {"task_name": task_name}))
        return {"status": "NOT_FOUND", "task_name": task_name, "exists": False}

    def resume_task(self, task_name: str, *, precondition: Mapping[str, Any] | None = None) -> Any:
        self._mutation_failure("resume_task")
        if task_name not in self.tasks:
            return {"ok": False, "error_code": "NOT_FOUND", "task_name": task_name}
        self.mutation_count += 1
        task = self.tasks[task_name]
        task["state"] = "RUNNING"
        task["revision"] = int(task.get("revision", 1)) + 1
        return {"ok": True, "task_name": task_name, "state": "RUNNING", "execution_id": f"exec-{self.mutation_count}"}

    def stop_task(self, task_name: str, *, precondition: Mapping[str, Any] | None = None) -> Any:
        self._mutation_failure("stop_task")
        if task_name not in self.tasks:
            return {"ok": False, "error_code": "NOT_FOUND", "task_name": task_name}
        self.mutation_count += 1
        task = self.tasks[task_name]
        task["state"] = "STOPPED"
        task["revision"] = int(task.get("revision", 1)) + 1
        return {"ok": True, "task_name": task_name, "state": "STOPPED"}

    def delete_task(self, task_name: str, *, precondition: Mapping[str, Any] | None = None) -> Any:
        self._mutation_failure("delete_task")
        if task_name not in self.tasks:
            return {"ok": False, "error_code": "NOT_FOUND", "task_name": task_name}
        self.mutation_count += 1
        del self.tasks[task_name]
        return {"ok": True, "task_name": task_name, "deleted": True}

    def set_task_priority(self, task_name: str, priority: int, *, precondition: Mapping[str, Any] | None = None) -> Any:
        self._mutation_failure("set_task_priority")
        if task_name not in self.tasks:
            return {"ok": False, "error_code": "NOT_FOUND", "task_name": task_name}
        self.mutation_count += 1
        task = self.tasks[task_name]
        task["priority"] = priority
        task["revision"] = int(task.get("revision", 1)) + 1
        return {"ok": True, "task_name": task_name, "priority": priority}

    def submit_task(self, task_name: str, config: Mapping[str, Any], *, precondition: Mapping[str, Any] | None = None) -> Any:
        self._mutation_failure("submit_task")
        if task_name in self.tasks:
            return {"ok": False, "error_code": "ALREADY_EXISTS", "task_name": task_name}
        self.mutation_count += 1
        self.tasks[task_name] = {
            "task_name": task_name,
            "state": "SUBMITTED",
            "exists": True,
            "revision": 1,
            **dict(config),
        }
        return {"ok": True, "task_name": task_name, "state": "SUBMITTED"}
