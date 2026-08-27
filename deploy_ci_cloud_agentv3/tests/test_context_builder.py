from __future__ import annotations

from deploy_ci_cloud_agentv3.agent.context_builder import ContextBuilder


def test_context_trimming_never_orphans_tool_messages():
    source = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "a", "type": "function", "function": {"name": "x", "arguments": "{}"}}, {"id": "b", "type": "function", "function": {"name": "y", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "a", "content": "A"},
        {"role": "tool", "tool_call_id": "b", "content": "B"},
        {"role": "assistant", "content": "after"},
        {"role": "user", "content": "new"},
    ]
    messages = ContextBuilder("sys", max_messages=4).build({"messages": source})[1:]
    for index, message in enumerate(messages):
        if message.get("role") == "tool":
            assert any(
                earlier.get("role") == "assistant" and any(call.get("id") == message.get("tool_call_id") for call in earlier.get("tool_calls") or [])
                for earlier in messages[:index]
            )
    # The whole two-tool group is either retained intact or dropped intact.
    ids = [m.get("tool_call_id") for m in messages if m.get("role") == "tool"]
    assert ids in ([], ["a", "b"])
