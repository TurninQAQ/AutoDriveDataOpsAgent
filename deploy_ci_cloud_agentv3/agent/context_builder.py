from __future__ import annotations

from typing import Any


class ContextBuilder:
    """Small context projection that never splits assistant-tool message groups."""

    def __init__(self, system_prompt: str, max_messages: int = 24) -> None:
        self.system_prompt = system_prompt
        self.max_messages = max_messages

    def build(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self._trim_complete_tool_groups(list(state.get("messages") or [])))
        if state.get("pending_action"):
            messages.append({"role": "system", "content": "A write proposal is pending human review. Do not claim it executed."})
        result = state.get("last_write_result")
        if result:
            messages.append({"role": "system", "content": f"Last deterministic write result: {result}"})
        return messages

    def _trim_complete_tool_groups(self, source: list[dict[str, Any]]) -> list[dict[str, Any]]:
        groups: list[list[dict[str, Any]]] = []
        i = 0
        while i < len(source):
            msg = source[i]
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                ids = {str(c.get("id") or "") for c in msg.get("tool_calls") or []}
                group = [msg]
                i += 1
                while i < len(source) and source[i].get("role") == "tool":
                    tool_id = str(source[i].get("tool_call_id") or "")
                    if ids and tool_id not in ids:
                        break
                    group.append(source[i])
                    i += 1
                groups.append(group)
                continue
            if msg.get("role") != "tool":
                groups.append([msg])
            i += 1

        selected: list[list[dict[str, Any]]] = []
        count = 0
        for group in reversed(groups):
            if selected and count + len(group) > self.max_messages:
                break
            selected.append(group)
            count += len(group)
            if count >= self.max_messages:
                break
        return [item for group in reversed(selected) for item in group]
