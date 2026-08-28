from __future__ import annotations

import json
import re
from typing import Any

from deploy_ci_cloud_agentv3.evaluation.models import BenchmarkCase
from deploy_ci_cloud_agentv3.providers.base import AssistantMessage, ToolCall


class BenchmarkScriptedProvider:
    """Deterministic provider policy; it selects tools but never fabricates outcomes."""

    def __init__(self, case: BenchmarkCase) -> None:
        self.case = case
        self.calls = 0
        self.tool_calls = 0

    async def invoke(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> AssistantMessage:
        self.calls += 1
        tool_messages = [item for item in messages if item.get("role") == "tool"]
        emitted_tool_names = []
        for item in messages:
            if item.get("role") == "assistant":
                for call in item.get("tool_calls") or []:
                    emitted_tool_names.append(str((call.get("function") or {}).get("name") or ""))

        if self.case.category == "READ":
            if not emitted_tool_names:
                calls = [self._read_call(name, index) for index, name in enumerate(self.case.expected_tools)]
                self.tool_calls += len(calls); return AssistantMessage(tool_calls=calls)
            return self._final("informational")

        if self.case.category == "MIXED":
            target = self._target()
            if "get_task_detail" not in emitted_tool_names:
                self.tool_calls += 1; return AssistantMessage(tool_calls=[ToolCall(id=f"{self.case.case_id}-read", name="get_task_detail", arguments={"task_name": target})])
            if not any(name.startswith("propose_") for name in emitted_tool_names):
                self.tool_calls += 1; return AssistantMessage(tool_calls=[ToolCall(id=f"{self.case.case_id}-proposal", name="propose_set_task_priority", arguments={"task_name": target, "priority": 5})])
            return self._final("write_verified")

        action = self._action()
        if action == "submit_task":
            if "prepare_task_spec" not in emitted_tool_names:
                self.tool_calls += 1
                return AssistantMessage(tool_calls=[ToolCall(id=f"{self.case.case_id}-prepare", name="prepare_task_spec", arguments={"task_prefix": "new", "dataset_path": "/data/benchmark"})])
            if "propose_submit_task" not in emitted_tool_names:
                artifact_id = self._artifact_id(tool_messages)
                self.tool_calls += 1
                return AssistantMessage(tool_calls=[ToolCall(id=f"{self.case.case_id}-proposal", name="propose_submit_task", arguments={"artifact_id": artifact_id})])
            return self._final("write_verified")

        if not any(name.startswith("propose_") for name in emitted_tool_names):
            proposal_name, arguments = self._proposal(action)
            self.tool_calls += 1
            return AssistantMessage(tool_calls=[ToolCall(id=f"{self.case.case_id}-proposal", name=proposal_name, arguments=arguments)])
        return self._final("write_verified")

    def _read_call(self, name: str, index: int) -> ToolCall:
        target = self._target()
        args: dict[str, Any]
        if name in {"get_task_detail", "diagnose_task"}: args = {"task_name": target}
        elif name == "get_queue_state": args = {"task_name": target}
        elif name == "search_knowledge": args = {"query": self.case.user_input, "top_k": 3}
        else: args = {}
        return ToolCall(id=f"{self.case.case_id}-read-{index}", name=name, arguments=args)

    def _proposal(self, action: str) -> tuple[str, dict[str, Any]]:
        target = self._target()
        if action == "set_task_priority":
            match = re.search(r"priority\s+(?:to\s+)?(\d+)", self.case.user_input, re.I)
            priority = int(match.group(1)) if match else 5
            return "propose_set_task_priority", {"task_name": target, "priority": priority}
        if action == "resume_task":
            datasets = ["A"] if "dataset a" in self.case.user_input.lower() else None
            return "propose_resume_task", {"task_name": target, "datasets": datasets}
        if action == "stop_task":
            datasets = ["A"] if "dataset a" in self.case.user_input.lower() else None
            return "propose_stop_task", {"task_name": target, "datasets": datasets}
        if action == "delete_task": return "propose_delete_task", {"task_name": target}
        raise ValueError(action)

    def _action(self) -> str:
        if self.case.expected_action:
            return self.case.expected_action
        if self.case.ground_truth.get("action"):
            return str(self.case.ground_truth["action"])
        text = self.case.user_input.lower()
        if "resume" in text: return "resume_task"
        if "stop" in text: return "stop_task"
        if "delete" in text: return "delete_task"
        if "submit" in text: return "submit_task"
        return "set_task_priority"

    def _target(self) -> str:
        if self.case.expected_target and self.case.expected_target != "new_task":
            return self.case.expected_target
        if self.case.ground_truth.get("target"):
            return str(self.case.ground_truth["target"])
        return "task_B" if "task_b" in self.case.user_input.lower() else "task_A"

    @staticmethod
    def _artifact_id(tool_messages: list[dict[str, Any]]) -> str:
        for item in reversed(tool_messages):
            try: payload = json.loads(str(item.get("content") or "{}"))
            except Exception: continue
            data = payload.get("data") if isinstance(payload, dict) else None
            if isinstance(data, dict) and data.get("artifact_id"): return str(data["artifact_id"])
        raise RuntimeError("prepared artifact id missing from tool result")

    @staticmethod
    def _final(status: str) -> AssistantMessage:
        return AssistantMessage(content=json.dumps({"status": status, "message": "benchmark scripted final"}))
