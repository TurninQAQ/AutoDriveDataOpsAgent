from deploy_ci_cloud_agentv2.agent.results import ResultStatus, normalize_read_result
from deploy_ci_cloud_agentv2.agent.goals import GoalDescriptor, SetTaskPriority
from deploy_ci_cloud_agentv2.safety.write_transaction import FrozenToolCall, MutationOutcome, MutationResult, WriteTransaction, WriteTransactionStatus
from deploy_ci_cloud_agentv2.verification.action import ActionVerifier
from deploy_ci_cloud_agentv2.verification.operational_goal import OperationalGoalVerifier


def _platform_task(priority):
    return {
        "task_name": "task_A",
        "exists": True,
        "state": "SUBMITTED",
        "priority": priority,
    }


def test_task_detail_normalizes_legacy_priority_detail_strictly():
    result = normalize_read_result(
        "get_task_detail",
        {"task_name": "task_A"},
        _platform_task({"task_type": "", "priority": 65, "priority_source": "explicit"}),
    )
    assert result.envelope.status is ResultStatus.SUCCESS
    assert result.validation_errors == ()
    assert result.priority == 65
    assert "priority" not in result.metadata


def test_present_malformed_legacy_priority_is_not_treated_as_absent():
    result = normalize_read_result(
        "get_task_detail", {"task_name": "task_A"}, _platform_task({"priority": "65"})
    )
    assert result.envelope.status is ResultStatus.SUCCESS
    assert result.priority is None
    assert result.validation_errors == ("priority.priority must be an integer; got str",)


class _PriorityFacade:
    def get_task_detail(self, task_name):
        return _platform_task({"priority": 65, "priority_source": "explicit"})


def _transaction():
    proposal = FrozenToolCall("call_1", "set_task_priority", {"task_name": "task_A", "priority": 65})
    return WriteTransaction(
        transaction_id="tx_1",
        proposal=proposal,
        fingerprint="fingerprint",
        bound_goal_ids=("g1",),
        goal_descriptor_version=1,
        completion_contract_fingerprint="contract",
        bound_goal_contract_fingerprint="goal-contract",
        status=WriteTransactionStatus.EXECUTED,
        approval_request_id="approval_1",
        affected_entities=("task_A",),
        mutation_result=MutationResult(MutationOutcome.CONFIRMED_SUCCESS, {"ok": True}),
    )


def test_priority_verifiers_accept_observed_legacy_priority_detail():
    transaction = _transaction()
    assert ActionVerifier(_PriorityFacade()).verify(transaction).status.value == "VERIFIED"
    assert OperationalGoalVerifier(_PriorityFacade()).verify(transaction).status.value == "VERIFIED"
