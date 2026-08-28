from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class BenchmarkCase(BaseModel):
    case_id: str
    category: Literal["READ", "WRITE", "MIXED", "FAULT"]
    user_input: str
    initial_platform_fixture: dict[str, Any] = Field(default_factory=dict)
    expected_tools: list[str] = Field(default_factory=list)

    # Two different concepts are intentionally separated:
    # - requested_mutation: the user's task asks for a side effect.
    # - expected_safe_mutation: under this case/fault, a safe system may cross the
    #   mutation boundary. For stale approval/tamper this is False; for an API-OK
    #   but no-effect fault it is True even though the business effect is absent.
    requested_mutation: bool = False
    expected_safe_mutation: bool = False

    expected_action: str | None = None
    expected_target: str | None = None
    expected_final_state: dict[str, Any] = Field(default_factory=dict)
    expected_safe_status: str = "informational"
    fault_injection: str | None = None
    ground_truth: dict[str, Any] = Field(default_factory=dict)


class BenchmarkOutcome(BaseModel):
    case_id: str
    baseline: str
    task_success: bool
    false_success: bool
    unsafe_write: bool
    wrong_target: bool
    business_effect_matches: bool
    tool_selection_correct: bool
    verification_success: bool
    llm_calls: int
    tool_calls: int
    latency_ms: float
    final_status: str = ""
    mutation_attempt_count: int = 0
    mutation_count: int = 0
    mutation_targets: list[str] = Field(default_factory=list)
    tool_trace: list[str] = Field(default_factory=list)
