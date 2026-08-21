from __future__ import annotations

import re

from platform_mcp.server import READ_ONLY_TOOL_NAMES, WRITE_TOOL_NAMES


# Mutation patterns. Local TaskSpec/YAML planning is intentionally not included.
WRITE_PATTERNS = (
    r"\bsubmit\b",
    r"\btrigger\b",
    r"\bstart\s+(?:the\s+)?task\b",
    r"\brun\s+(?:the\s+)?task\b",
    r"\bstop\b",
    r"\bkill\b",
    r"\bdelete\b",
    r"\bremove\b",
    r"\bresume\b",
    r"\brestart\b",
    r"\bset\s+priority\b",
    r"\bchange\s+priority\b",
    r"提交",
    r"触发.*任务",
    r"启动.*任务",
    r"执行.*任务",
    r"停止",
    r"杀掉",
    r"终止",
    r"删除",
    r"恢复.*任务",
    r"重启",
    r"修改.*优先级",
    r"调整.*优先级",
    r"优先级.*改",
    r"让.*先跑",
)

PLANNING_PATTERNS = (
    r"\bcreate\s+(?:a\s+)?task\b",
    r"\bgenerate\s+(?:a\s+)?task\b",
    r"\btask\s+(?:config|yaml|plan)\b",
    r"创建.*任务",
    r"新建.*任务",
    r"生成.*(?:任务|yaml|配置)",
    r"任务.*(?:配置|yaml|规划)",
    r"规划.*任务",
)


class AgentPolicyEngine:
    supports_writes = True
    RISK = {
        "submit_task": "high",
        "resume_task": "medium",
        "set_task_priority": "high",
        "stop_task": "high",
        "delete_task": "destructive",
    }

    def __init__(
        self,
        allowed_read_tools: tuple[str, ...] = READ_ONLY_TOOL_NAMES,
        allowed_write_tools: tuple[str, ...] = WRITE_TOOL_NAMES,
        max_tool_calls: int = 6,
    ):
        self.allowed_read_tools = frozenset(allowed_read_tools)
        self.allowed_write_tools = frozenset(allowed_write_tools)
        # Compatibility alias used by older tests.
        self.allowed_tools = self.allowed_read_tools
        self.max_tool_calls = max(1, int(max_tool_calls))

    def is_write_request(self, text: str) -> bool:
        normalized = text.strip().lower()
        return any(re.search(pattern, normalized, re.IGNORECASE) for pattern in WRITE_PATTERNS)

    def is_task_planning_request(self, text: str) -> bool:
        normalized = text.strip().lower()
        if self.is_write_request(normalized):
            return False
        return any(re.search(pattern, normalized, re.IGNORECASE) for pattern in PLANNING_PATTERNS)

    def validate_tool_name(self, name: str) -> None:
        # Normal model-planned calls remain read-only. A write tool is only
        # executed by the guarded WriteActionCoordinator after HITL or the
        # separate deterministic V1.7 autonomy policy authorizes resume_task.
        if name not in self.allowed_read_tools:
            raise PermissionError(f"Tool is not allowed before HITL approval: {name}")

    def validate_write_tool(self, name: str) -> None:
        if name not in self.allowed_write_tools:
            raise PermissionError(f"Write tool is not approved by Agent policy: {name}")

    def validate_tool_count(self, count: int) -> None:
        if count > self.max_tool_calls:
            raise PermissionError(
                f"Agent plan requested {count} read tool calls; limit={self.max_tool_calls}"
            )

    def risk_for_tool(self, name: str) -> str:
        self.validate_write_tool(name)
        return self.RISK.get(name, "high")

    def requires_approval(self, name: str) -> bool:
        self.validate_write_tool(name)
        return True


class ReadOnlyPolicy(AgentPolicyEngine):
    supports_writes = False
    """Backward-compatible V0.4-V0.6 policy name.

    Existing tests/imports keep working. V0.8 runtime uses AgentPolicyEngine but
    normal model tool execution is still read-only before approval.
    """
