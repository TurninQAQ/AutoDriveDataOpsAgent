"""Adapter from the V2 HTTP gateway tool surface to platform services."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from deploy_ci_cloud_agentv2.safety.precondition import PreconditionReader
from deploy_ci_cloud_agentv2.safety.write_transaction import FrozenToolCall

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
                return _normalize_queue_state(self.facade.get_queue_state(args.get("task_name")))
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
                precondition = args.pop("precondition")
                task_name = str(args.get("task_name") or "")
                if not task_name or not isinstance(precondition, Mapping):
                    raise PlatformBackendError("INVALID_PARAMS")
                self._assert_v2_precondition(tool_name, args, task_name, precondition)
                # The V2 Runtime fingerprint is the authority crossing this
                # boundary. The platform service then captures its own local
                # precondition immediately before the business mutation, giving
                # the backend a second deterministic TOCTOU check without any
                # semantic decision layer.
                backend_precondition = self.facade.get_write_precondition(task_name)
                method = getattr(self.facade, tool_name, None)
                if method is None:
                    raise PlatformBackendError("TOOL_NOT_FOUND")
                if tool_name == "submit_task":
                    args["task_prefix"] = args.pop("task_name")
                    args["precondition"] = backend_precondition
                elif tool_name in {"resume_task", "stop_task"}:
                    args["datasets"] = None
                    args["precondition"] = backend_precondition
                else:
                    args["precondition"] = backend_precondition
                return method(**args)
        except PlatformBackendError:
            raise
        except (TypeError, ValueError, KeyError) as exc:
            raise PlatformBackendError("INVALID_PARAMS") from exc
        except Exception as exc:
            raise PlatformBackendError("PLATFORM_TOOL_ERROR") from exc
        raise PlatformBackendError("TOOL_NOT_FOUND")

    def _assert_v2_precondition(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        task_name: str,
        expected: Mapping[str, Any],
    ) -> None:
        """Recompute the exact V2 fingerprint immediately before dispatch."""

        if expected.get("target") != task_name or expected.get("tool_name") != tool_name:
            raise PlatformBackendError("PRECONDITION_FAILED")
        fingerprint = expected.get("fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise PlatformBackendError("INVALID_PARAMS")
        try:
            call = FrozenToolCall(
                call_id="gateway-precondition",
                tool_name=tool_name,
                arguments=dict(arguments),
            )
            current = PreconditionReader(self.facade).capture(call)
        except Exception as exc:
            raise PlatformBackendError("PRECONDITION_FAILED") from exc
        if current.fingerprint != fingerprint:
            raise PlatformBackendError("PRECONDITION_FAILED")


def _normalize_queue_state(value: Any) -> Any:
    """Map the platform queue-file shape to the strict V2 READ contract.

    The platform queue store represents an empty global queue as
    ``active=None`` and an active item as an object under ``active``.  The V2
    result contract represents global queue state as a platform-scoped queue
    collection; its ``active`` field is reserved for numeric aggregate counts.
    Keep this conversion at the platform boundary so strict result parsing
    remains fail-closed for malformed external payloads.
    """
    if not isinstance(value, Mapping):
        return value
    if "task_name" in value or "position" in value:
        return value
    active = value.get("active")
    queue = value.get("queue")
    if active is not None and not isinstance(active, Mapping):
        return value
    if queue is not None and not isinstance(queue, list):
        return value

    entries: list[dict[str, Any]] = []
    if isinstance(active, Mapping):
        entry = dict(active)
        entry["position"] = 0
        entry["state"] = str(entry.get("state") or entry.get("status") or "ACTIVE").upper()
        entries.append(entry)
    for index, item in enumerate(queue or [], start=1):
        if not isinstance(item, Mapping):
            return value
        entry = dict(item)
        entry.setdefault("position", index)
        if "state" not in entry and "status" in entry:
            entry["state"] = entry["status"]
        if isinstance(entry.get("state"), str):
            entry["state"] = entry["state"].upper()
        entries.append(entry)

    normalized = dict(value)
    normalized.pop("active", None)
    normalized["scope"] = "PLATFORM"
    normalized["queue"] = entries
    return normalized
