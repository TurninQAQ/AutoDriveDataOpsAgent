from __future__ import annotations

import pytest

from deploy_ci_cloud_agentv3.models.pending_action import PendingAction, compute_pending_action_fingerprint
from deploy_ci_cloud_agentv3.services.write_service import WriteService


class UnknownRuntime:
    def __init__(self):
        self.priority = 3; self.write_calls = 0; self.precondition = {"v": 1}

    async def call_tool(self, name, args):
        if name == "capture_write_precondition": return dict(self.precondition)
        if name == "set_task_priority":
            self.write_calls += 1; self.priority = args["priority"]
            raise ConnectionResetError("response lost after request dispatch")
        if name == "get_action_verification_snapshot":
            return {"task_exists": True, "priority": self.priority, "errors": {}}
        raise AssertionError(name)


@pytest.mark.asyncio
async def test_unknown_outcome_reconciles_by_read_without_retrying_write():
    runtime = UnknownRuntime(); args = {"task_name": "task_a", "priority": 5}; pre = {"v": 1}
    fp = compute_pending_action_fingerprint(action="set_task_priority", args=args, artifact=None, precondition=pre)
    action = PendingAction(proposal_id="p", action="set_task_priority", args=args, reason="", expected_effect="", before={"priority": 3}, precondition=pre, fingerprint=fp)
    result = await WriteService(runtime).execute(action, fp)
    assert result.status == "VERIFIED" and result.verified is True
    assert runtime.write_calls == 1
