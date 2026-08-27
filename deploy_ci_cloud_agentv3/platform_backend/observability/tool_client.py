from __future__ import annotations

import time

# This module is retained as a transport-observability helper only.  Keep the
# WRITE classification stays local to the platform backend instead of importing
# legacy semantic/runtime package from the previous project.
WRITE_TOOL_NAMES = frozenset(
    {
        "resume_task",
        "submit_task",
        "stop_task",
        "delete_task",
        "set_task_priority",
    }
)

from .recorder import TraceRecorder, current_trace_id


class ObservedToolClient:
    """Transparent ToolClient wrapper that traces every MCP call, including internal guarded-write tools."""

    def __init__(self, delegate, recorder: TraceRecorder):
        self.delegate = delegate
        self.recorder = recorder

    async def describe_tools(self):
        return await self.delegate.describe_tools()

    async def execute(self, calls):
        started = time.perf_counter()
        try:
            results = await self.delegate.execute(calls)
        except Exception as exc:
            self.recorder.record_current(
                "tool",
                "tool_batch",
                status="error",
                duration_ms=(time.perf_counter() - started) * 1000,
                data={"calls": [getattr(call, "model_dump", lambda **_: str(call))(mode="json") for call in calls], "error": str(exc)},
            )
            raise
        by_index = list(results)
        elapsed_ms = (time.perf_counter() - started) * 1000
        for index, call in enumerate(calls):
            obs = by_index[index] if index < len(by_index) else None
            is_mutation = call.name in WRITE_TOOL_NAMES
            stage = "mutation" if is_mutation else "tool"
            status = "ok" if obs is not None and getattr(obs, "ok", False) else "error"
            data = {
                "tool": call.name,
                "arguments": getattr(call, "arguments", {}),
                "ok": bool(obs is not None and getattr(obs, "ok", False)),
                "result": getattr(obs, "data", None) if obs is not None else None,
                "error": getattr(obs, "error", None) if obs is not None else "missing observation",
            }
            self.recorder.record_current(stage, call.name, status=status, duration_ms=elapsed_ms, data=data)
        return results
