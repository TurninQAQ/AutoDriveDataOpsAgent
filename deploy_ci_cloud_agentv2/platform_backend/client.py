"""Adapter from the V2 HTTP gateway tool surface to platform services."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .core.errors import TaskConfigError


class PlatformBackendError(RuntimeError):
    """Safe downstream error carrying a gateway-visible deterministic code."""

    def __init__(self, code: str, message: str = "platform tool call failed") -> None:
        super().__init__(message)
        self.code = code
        self.safe_message = message


def _missing_task(task_name: str, error: BaseException) -> bool:
    """Recognize only the canonical missing-config discriminator.

    Generic ``TaskConfigError`` values can represent malformed configuration,
    invalid datasets, or other platform failures.  Only the exact stable
    prefix emitted by ``load_task_config`` is promoted to NOT_FOUND.
    """

    return bool(
        task_name
        and str(error).startswith("Task config not found:")
    )


class InProcessPlatformClient:
    """Call the V2-owned platform facade without a second semantic layer."""

    def __init__(self, facade: Any) -> None:
        self.facade = facade

    def call(self, tool_name: str, arguments: Mapping[str, Any]) -> Any:
        args = dict(arguments)
        try:
            if tool_name == "get_task_detail":
                task_name = str(args.get("task_name", ""))
                try:
                    return self.facade.get_task_detail(task_name)
                except TaskConfigError as exc:
                    if _missing_task(task_name, exc):
                        return {"status": "NOT_FOUND", "task_name": task_name, "exists": False}
                    raise PlatformBackendError("PLATFORM_TOOL_ERROR") from exc
            if tool_name == "get_gpu_pool":
                return self.facade.get_gpu_pool(cleanup_dead=False)
            if tool_name == "search_knowledge":
                return self.facade.search_knowledge(
                    str(args.get("query", "")), int(args.get("top_k", 5))
                )
            if tool_name == "get_queue_state":
                return self.facade.get_queue_state(args.get("task_name"))
            if tool_name == "diagnose_task":
                task_name = str(args.get("task_name", ""))
                try:
                    return self.facade.diagnose_task(task_name)
                except TaskConfigError as exc:
                    if _missing_task(task_name, exc):
                        return {"status": "NOT_FOUND", "task_name": task_name}
                    raise PlatformBackendError("PLATFORM_TOOL_ERROR") from exc
            if tool_name in {"resume_task", "stop_task", "delete_task", "set_task_priority", "submit_task"}:
                # The V2 write contract requires an approval-bound precondition.
                # The direct HTTP tool surface must not synthesize or omit it.
                if "precondition" not in args:
                    raise PlatformBackendError("WRITE_PRECONDITION_REQUIRED")
                method = getattr(self.facade, tool_name, None)
                if method is None:
                    raise PlatformBackendError("TOOL_NOT_FOUND")
                return method(**args)
        except PlatformBackendError:
            raise
        except (TypeError, ValueError, KeyError) as exc:
            raise PlatformBackendError("INVALID_PARAMS") from exc
        except Exception as exc:
            raise PlatformBackendError("PLATFORM_TOOL_ERROR") from exc
        raise PlatformBackendError("TOOL_NOT_FOUND")

