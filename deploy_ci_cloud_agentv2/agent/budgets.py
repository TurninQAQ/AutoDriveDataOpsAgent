"""Small, explicit runtime budget model."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class RuntimeBudgets:
    max_agent_steps: int = 8
    max_read_tool_calls: int = 16
    max_parallel_read_batch: int = 3
    max_completion_gate_rejections: int = 3
    max_context_tokens: int = 12_000
    max_runtime_read_retries: int = 2

    def __post_init__(self) -> None:
        for field_name in (
            "max_agent_steps",
            "max_read_tool_calls",
            "max_parallel_read_batch",
            "max_completion_gate_rejections",
            "max_context_tokens",
            "max_runtime_read_retries",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must not be negative")


@dataclass(frozen=True)
class BudgetState:
    limits: RuntimeBudgets
    agent_steps_used: int = 0
    read_tool_calls_used: int = 0
    completion_gate_rejections: int = 0
    runtime_read_retries_used: int = 0

    def with_agent_step(self) -> "BudgetState":
        return replace(self, agent_steps_used=self.agent_steps_used + 1)

    def with_read_calls(self, count: int) -> "BudgetState":
        return replace(self, read_tool_calls_used=self.read_tool_calls_used + count)

    def with_gate_rejection(self) -> "BudgetState":
        return replace(
            self, completion_gate_rejections=self.completion_gate_rejections + 1
        )

    def with_retries(self, count: int) -> "BudgetState":
        return replace(
            self, runtime_read_retries_used=self.runtime_read_retries_used + count
        )

    def has_agent_step(self) -> bool:
        return self.agent_steps_used < self.limits.max_agent_steps

    def has_read_calls(self, count: int) -> bool:
        return self.read_tool_calls_used + count <= self.limits.max_read_tool_calls

