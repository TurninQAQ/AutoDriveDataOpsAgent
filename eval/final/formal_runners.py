"""Concrete, deterministic runners for harness dry-runs.

These runners use a scripted model stub and an isolated fixture.  They are
not formal model results and never call a provider.  Their purpose is to
prove that scenario -> fixture -> system-specific execution -> raw facts is
actually executable before model quota is spent.
"""

from __future__ import annotations

from typing import Any, Mapping

from .baselines import expected_policy_for_system
from .fixture_registry import Fixture, resolve_fixture


class ScriptedModel:
    """Deterministic stand-in for the model boundary used by dry-runs."""

    def propose(self, scenario: Any, fixture: Fixture) -> dict[str, Any]:
        target = scenario.expected_target
        if fixture.provenance_conflict and target:
            observed_target = "other_task"
        else:
            observed_target = target or fixture.task_name
        return {
            "intent": scenario.expected_intent,
            "target": observed_target,
            "tool_calls": list(scenario.required_tools),
            "input_tokens": 32 + len(scenario.prompt),
            "output_tokens": 12,
        }


class BaseFormalRunner:
    system = "full"

    def __init__(self, model_stub: ScriptedModel | None = None) -> None:
        self.model_stub = model_stub or ScriptedModel()

    def _policy(self, scenario: Any) -> str | None:
        return expected_policy_for_system(scenario, self.system)

    def _scope(self, scenario: Any, fixture: Fixture) -> list[str]:
        return list(scenario.expected_datasets or fixture.currently_failed_datasets)

    def _execution_facts(self, scenario: Any, fixture: Fixture, facts: dict[str, Any], *, policy: str | None) -> None:
        outcome = scenario.effective_outcome_type
        if outcome in {"safe_auto_execution", "hitl_execution"}:
            facts.update(
                {
                    "frozen_datasets": self._scope(scenario, fixture),
                    "mutation_count": 1,
                    "action_verification": {"status": "VERIFIED"},
                    "goal_verification": {"status": (fixture.post_goal or scenario.expected_goal or "SATISFIED")},
                }
            )
        elif outcome == "safe_refusal":
            facts.update(
                {
                    "approval_created": False,
                    "mutation_count": 0,
                    "goal_verification": {"status": scenario.expected_goal or "INCONCLUSIVE"},
                }
            )
        elif scenario.expected_goal:
            facts["goal_verification"] = {"status": scenario.expected_goal}
        if policy == "HITL":
            facts.update({"approval_required": True, "authorization_mode": "hitl", "oracle_approval": True, "mutation_count_before_approval": 0})
        elif policy == "AUTO":
            facts.update({"approval_required": False, "authorization_mode": "auto"})
        elif policy == "DENY":
            facts.update({"approval_required": False, "authorization_mode": "deny"})

    def __call__(self, scenario: Any, repetition: int, model: str) -> Mapping[str, Any]:
        fixture = resolve_fixture(scenario.fixture)
        facts = self.model_stub.propose(scenario, fixture)
        policy = self._policy(scenario)
        facts.update(
            {
                "policy_mode": policy,
                "authorization_mode": (policy or "none").lower(),
                "fixture_name": fixture.name,
                "task_exists": fixture.task_exists,
                "critical_evidence_available": fixture.critical_evidence_available,
                "provenance_conflict": fixture.provenance_conflict,
                "repetition_observed": repetition,
                "status": "OK",
                "adaptive_write": 0,
                "direct_write": False,
                "direct_model_write": False,
                "sandbox_only": False,
            }
        )
        self._execution_facts(scenario, fixture, facts, policy=policy)
        if scenario.expected_datasets and "frozen_datasets" not in facts:
            facts["frozen_datasets"] = list(scenario.expected_datasets)
        return facts


class ScriptedFullRunner(BaseFormalRunner):
    system = "full"


class ScriptedHitlRunner(BaseFormalRunner):
    system = "hitl_only"


class ScriptedNaiveRunner(BaseFormalRunner):
    system = "naive_tool"

    def __call__(self, scenario: Any, repetition: int, model: str) -> Mapping[str, Any]:
        facts = dict(super().__call__(scenario, repetition, model))
        facts["policy_mode"] = "NAIVE_PROPOSAL"
        facts["authorization_mode"] = "naive"
        facts["direct_model_write"] = scenario.effective_outcome_type in {"safe_auto_execution", "hitl_execution", "safe_refusal"}
        facts["sandbox_only"] = True
        if scenario.effective_outcome_type == "safe_refusal":
            # Naive tools may attempt an invalid write in the isolated sandbox;
            # the evaluator, not this runner, decides that it was unresolved.
            facts["mutation_count"] = 1
            facts["approval_created"] = False
        return facts


 # Backward-compatible aliases for existing harness imports.  The explicit
 # Scripted* names prevent these quota-free dry runners from being mistaken
 # for formal live-model execution.
FullFormalRunner = ScriptedFullRunner
HitlOnlyFormalRunner = ScriptedHitlRunner
NaiveToolFormalRunner = ScriptedNaiveRunner


def formal_runner_for(system: str) -> BaseFormalRunner:
    runners = {"full": ScriptedFullRunner, "hitl_only": ScriptedHitlRunner, "naive_tool": ScriptedNaiveRunner}
    try:
        return runners[system]()
    except KeyError as exc:
        raise ValueError(f"Unknown formal runner system: {system}") from exc


def dry_run_systems() -> tuple[str, ...]:
    return ("full", "hitl_only", "naive_tool")
