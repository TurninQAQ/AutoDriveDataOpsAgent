from deploy_ci_cloud_agentv2.agent.budgets import BudgetState, RuntimeBudgets
from deploy_ci_cloud_agentv2.agent.events import EventProvenance, EventStore


def test_event_store_is_readable_idempotent_and_has_provenance():
    store = EventStore()
    provenance = EventProvenance("m", "p", "catalog", "principles-v", "hash")
    first = store.append(
        event_type="AgentRunStarted",
        request_id="r",
        thread_id="t",
        payload={"ok": True},
        provenance=provenance,
        event_id="stable-event",
    )
    duplicate = store.append(
        event_type="AgentRunStarted",
        request_id="r",
        thread_id="t",
        payload={"changed": True},
        provenance=provenance,
        event_id="stable-event",
    )
    assert first == duplicate
    assert store.readable_trace("t")[0]["provenance"]["tool_catalog_hash"] == "catalog"


def test_budget_counters_are_explicit_and_bounded():
    state = BudgetState(RuntimeBudgets(max_agent_steps=1, max_read_tool_calls=2))
    assert state.has_agent_step()
    state = state.with_agent_step().with_read_calls(2)
    assert not state.has_agent_step()
    assert state.has_read_calls(0)
    assert not state.has_read_calls(1)
