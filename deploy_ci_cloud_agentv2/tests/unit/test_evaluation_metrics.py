from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from deploy_ci_cloud_agentv2.evaluation import EvaluationLabels, evaluate_audit, goal_state_macro_f1


def _event(kind, payload, *, request_id="r1", seconds=0):
    return SimpleNamespace(
        event_type=kind,
        payload=payload,
        request_id=request_id,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds),
    )


def test_goal_state_macro_f1_counts_missing_and_extra_predictions():
    assert goal_state_macro_f1({"g1": "SATISFIED"}, {"g1": "SATISFIED"}) == 1.0
    score = goal_state_macro_f1(
        {"g1": "SATISFIED", "g2": "REJECTED"},
        {"g1": "SATISFIED"},
    )
    assert score is not None and 0.0 < score < 1.0


def test_audit_metrics_are_downstream_and_cover_section_70_counters():
    events = [
        _event("AgentRunStarted", {}, seconds=0),
        _event("AgentDecisionMade", {
            "calls": [{"call_id": "a"}, {"call_id": "b"}],
            "context_projection_compressed": True,
            "estimated_context_chars": 401,
        }, seconds=1),
        _event("ToolCallStarted", {"attempt": 1}, seconds=2),
        _event("ToolCallStarted", {"attempt": 2}, seconds=3),
        _event("CompletionGateEvaluated", {"passed": True}, seconds=4),
        _event("GoalOutcomeUpdated", {"goal_id": "g1", "status": "SATISFIED"}, seconds=5),
        _event("ApprovalRequested", {"approval_request_id": "ar1", "transaction_id": "tx1"}, seconds=6),
        _event("ApprovalGranted", {"approval_request_id": "ar1", "transaction_id": "tx1"}, seconds=8),
        _event("MutationStarted", {"transaction_id": "tx1"}, seconds=9),
        _event("ActionVerificationRecorded", {"transaction_id": "tx1", "status": "VERIFIED"}, seconds=10),
        _event("AgentRunCompleted", {"status": "COMPLETED"}, seconds=11),
    ]
    metrics = evaluate_audit(
        events,
        labels=EvaluationLabels(
            request_success={"r1": True},
            goal_states={"g1": "SATISFIED"},
            expected_write_targets={"tx1": "A"},
        ),
    )
    assert metrics.resolved_at_1 == 1.0
    assert metrics.false_success_rate == 0.0
    assert metrics.human_approval_completion_rate == 1.0
    assert metrics.write_verification_success_rate == 1.0
    assert metrics.goal_state_macro_f1 == 1.0
    assert metrics.unapproved_write_execution_rate == 0.0
    assert metrics.duplicate_write_execution_rate == 0.0
    assert metrics.parallel_read_savings == 1
    assert metrics.runtime_retry_count == 1
    assert metrics.context_compression_count == 1
    assert metrics.estimated_tokens == 101
    assert metrics.approval_latency_seconds == pytest.approx(2.0)
    assert metrics.latency_seconds == pytest.approx(11.0)


def test_unapproved_duplicate_and_unknown_replay_metrics_fail_loudly():
    events = [
        _event("MutationStarted", {"transaction_id": "tx-no-approval"}, seconds=0),
        _event("ApprovalGranted", {"transaction_id": "tx2"}, seconds=1),
        _event("MutationStarted", {"transaction_id": "tx2"}, seconds=2),
        _event("MutationStarted", {"transaction_id": "tx2"}, seconds=3),
        _event("ReconciliationRequired", {"transaction_id": "tx3"}, seconds=4),
        _event("MutationStarted", {"transaction_id": "tx3"}, seconds=5),
    ]
    metrics = evaluate_audit(events)
    assert metrics.unapproved_write_execution_rate == pytest.approx(2 / 4)
    assert metrics.duplicate_write_execution_rate == pytest.approx(1 / 4)
    assert metrics.replay_after_unknown_outcome_rate == 1.0
