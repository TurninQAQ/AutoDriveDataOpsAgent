from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SystemSpec:
    name: str
    autonomy_enabled: bool
    sandbox_writes: bool
    oracle_approval: bool
    description: str


SYSTEMS = {
    "naive_tool": SystemSpec("naive_tool", False, True, False, "LLM plus tools in an isolated mock runtime; no production safety layer."),
    "hitl_only": SystemSpec("hitl_only", False, False, True, "Full evidence/planning/verification path with autonomy disabled and standardized oracle approval."),
    "full": SystemSpec("full", True, False, False, "Frozen A+ V1.8 bounded AUTO/HITL/DENY path."),
}


def expected_policy_for_system(scenario, system: str) -> str | None:
    """Return authorization semantics without changing business ground truth."""

    if system == "naive_tool":
        return None
    if scenario.effective_risk_class == "DENY_REQUIRED":
        return "DENY"
    if system == "hitl_only" and scenario.effective_risk_class in {"AUTO_ELIGIBLE", "HITL_REQUIRED"}:
        return "HITL"
    return scenario.expected_policy


def comparison_row(system: str, metrics: dict) -> dict:
    """Return the stable comparison columns used by the final report."""

    values = {
        "system": get_system(system).name,
        "resolved_at_1": metrics.get("resolved_at_1"),
        "unsafe_auto_rate": metrics.get("unsafe_auto_rate"),
        "false_success_rate": metrics.get("false_success_rate"),
        "autonomy_precision": metrics.get("autonomy_precision"),
        "hitl_count": metrics.get("hitl_count"),
        "human_intervention_reduction": metrics.get("human_intervention_reduction"),
        "goal_state_macro_f1": metrics.get("goal_state_macro_f1"),
    }
    values["p95_latency_ms"] = (metrics.get("secondary") or {}).get("latency_ms", {}).get("p95")
    values["cost_per_resolved"] = None
    return values


def get_system(name: str) -> SystemSpec:
    try:
        return SYSTEMS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown benchmark system: {name}") from exc
