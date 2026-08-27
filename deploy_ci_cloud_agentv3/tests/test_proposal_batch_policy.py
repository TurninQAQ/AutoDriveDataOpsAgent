from deploy_ci_cloud_agentv3.agent.graph import _proposal_policy_rejection
from deploy_ci_cloud_agentv3.providers.base import ToolCall


def test_mixed_proposal_batch_returns_one_tool_message_per_tool_call():
    calls = [
        ToolCall(id="read-1", name="get_task_detail", arguments={"task_name": "task_a"}),
        ToolCall(id="proposal-1", name="propose_delete_task", arguments={"task_name": "task_a"}),
    ]
    errors, messages = _proposal_policy_rejection(calls)
    assert [item.call_id for item in errors] == ["read-1", "proposal-1"]
    assert [item.tool_name for item in errors] == ["get_task_detail", "propose_delete_task"]
    assert [item["tool_call_id"] for item in messages] == ["read-1", "proposal-1"]
    assert all(item["role"] == "tool" for item in messages)
    assert len(messages) == len(calls)
    assert all(item.kind == "TOOL_ERROR" for item in errors)
    assert all("only tool call" in (item.error or "") for item in errors)
