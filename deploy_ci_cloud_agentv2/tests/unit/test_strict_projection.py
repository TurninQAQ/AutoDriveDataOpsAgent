from pathlib import Path

import pytest

from deploy_ci_cloud_agentv2.agent.budgets import BudgetState, RuntimeBudgets
from deploy_ci_cloud_agentv2.agent.context import ContextBudgetExceeded, ContextBuilder
from deploy_ci_cloud_agentv2.agent.contracts import CompletionContractCompiler
from deploy_ci_cloud_agentv2.agent.evidence import EvidenceState
from deploy_ci_cloud_agentv2.agent.goals import GoalDescriptor, ReadTaskState
from deploy_ci_cloud_agentv2.agent.outcomes import GoalOutcome
from deploy_ci_cloud_agentv2.agent.principles import load_operating_principles
from deploy_ci_cloud_agentv2.agent.state import CurrentRequestContext, ThreadHistory
from deploy_ci_cloud_agentv2.tests.helpers import identity


def _current(tmp_path: Path, *, max_context_tokens=2_000, feedback=()):
    principles = tmp_path / "principles.md"
    principles.write_text("## Principle P01 — Test\nAdvisory only.\n", encoding="utf-8")
    snapshot = load_operating_principles(principles)
    owner = identity()
    descriptor = GoalDescriptor(1, (ReadTaskState("g1", "task_A"),))
    return CurrentRequestContext(
        identity=owner,
        user_input="task_A",
        messages=({"role": "user", "content": "x" * 100_000},),
        step_count=0,
        tool_call_count=0,
        goal_descriptor=descriptor,
        completion_contract=CompletionContractCompiler().compile(descriptor),
        goal_outcomes={"g1": GoalOutcome("g1")},
        evidence=EvidenceState(owner),
        observations=(),
        budgets=BudgetState(RuntimeBudgets(max_context_tokens=max_context_tokens)),
        terminal_state=None,
        termination_reason=None,
        operating_principles_snapshot=snapshot,
        decision=None,
        final_candidate=None,
        gate_feedback=tuple(feedback),
        gate_passed=None,
        new_turn=False,
        continue_after_read_guard=False,
    )


def test_budget_measures_the_exact_final_provider_payload(tmp_path):
    current = _current(tmp_path, feedback=tuple(f"feedback-{index}-" + "x" * 100 for index in range(100)))
    context = ContextBuilder().build(current, ThreadHistory())
    payload = context.model_facing_payload()
    assert context.estimated_context_chars == len(repr(payload))
    assert context.estimated_context_chars <= current.budgets.limits.max_context_tokens * 4
    assert len(context.runtime_structured.gate_feedback) <= 8
    assert context.runtime_structured.gate_feedback == tuple(current.gate_feedback[:8])
    assert len(context.messages[0]["content"]) < 100_000


def test_tiny_budget_fails_closed_instead_of_dropping_critical_state(tmp_path):
    current = _current(tmp_path, max_context_tokens=1)
    with pytest.raises(ContextBudgetExceeded):
        ContextBuilder().build(current, ThreadHistory())
