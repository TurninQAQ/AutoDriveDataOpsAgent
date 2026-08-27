from __future__ import annotations

from deploy_ci_cloud_agentv3.agent.final_guard import FinalGuard
from deploy_ci_cloud_agentv3.models.final_response import FinalCandidate
from deploy_ci_cloud_agentv3.models.write_result import WriteResult


def test_final_guard_requires_verified_result_for_success_status():
    failed = WriteResult(id="w1", action="set_task_priority", status="VERIFICATION_FAILED", verified=False, error="state unchanged")
    final = FinalGuard().build(FinalCandidate(status="write_verified", message="The priority was changed successfully."), failed.model_dump(mode="json"))
    assert final.status == "write_failed"
    assert "successfully" not in final.message.lower()


def test_final_guard_allows_verified_success_with_runtime_owned_wording():
    verified = WriteResult(id="w2", action="set_task_priority", status="VERIFIED", verified=True)
    final = FinalGuard().build(FinalCandidate(status="write_verified", message="anything"), verified.model_dump(mode="json"))
    assert final.status == "write_verified" and final.write_result_id == "w2"
    assert "verified write succeeded" in final.message.lower()


def test_unstructured_hallucinated_success_is_not_released_without_write_result():
    final = FinalGuard().build("Task task_a was deleted successfully.", None)
    assert final.status == "informational"
    assert "deleted successfully" not in final.message.lower()
    assert "no write-success claim" in final.message.lower()


def test_structured_write_verified_without_write_result_becomes_not_executed():
    final = FinalGuard().build(FinalCandidate(status="write_verified", message="Deleted."), None)
    assert final.status == "write_not_executed"
    assert final.message == "No platform write was executed in this run."
