from __future__ import annotations

import json

import platform_eval.aligned as aligned
from platform_agent.models import AgentIntent, AgentPlan, ToolCallSpec
from platform_eval.argument_contract import evaluate_argument_contract
from platform_eval.deepeval_adapter import _argument_requirements, _deterministic_tool_metrics


def _result(actual, *, expected=None, contract=None):
    return evaluate_argument_contract(
        actual,
        expected_arguments=expected,
        argument_contract=contract,
    )


def test_exact_matcher_passes_and_fails():
    rule = {"task_name": {"match": "exact", "value": "release_demo"}}
    assert _result([{"name": "diagnose_task", "arguments": {"task_name": "release_demo"}}], contract={"diagnose_task": rule})["ok"]
    failed = _result([{"name": "diagnose_task", "arguments": {"task_name": "release_test"}}], contract={"diagnose_task": rule})
    assert not failed["ok"]
    assert failed["wrong_exact_arguments"]


def test_subset_matcher_supports_nested_objects_and_extra_fields():
    result = _result(
        [{"name": "submit_task", "arguments": {"config": {"priority": 5, "extra": True}}}],
        contract={"submit_task": {"config": {"match": "subset", "value": {"priority": 5}}}},
    )
    assert result["ok"]


def test_presence_and_non_empty_matchers():
    contract = {"search_knowledge": {
        "query": {"match": "non_empty"},
        "top_k": {"match": "present"},
    }}
    assert _result([{"name": "search_knowledge", "arguments": {"query": "GPU reservation rules", "top_k": 5}}], contract=contract)["ok"]
    assert not _result([{"name": "search_knowledge", "arguments": {"query": "", "top_k": 5}}], contract=contract)["ok"]


def test_one_of_and_range_matchers():
    contract = {"tool": {
        "mode": {"match": "one_of", "values": ["safe", "read"]},
        "top_k": {"match": "range", "min": 1, "max": 10},
    }}
    assert _result([{"name": "tool", "arguments": {"mode": "read", "top_k": 5}}], contract=contract)["ok"]
    assert not _result([{"name": "tool", "arguments": {"mode": "write", "top_k": 20}}], contract=contract)["ok"]


def test_legacy_expected_arguments_remain_recursive_subset_compatible():
    result = _result(
        [{"name": "get_task_detail", "arguments": {"task_name": "release_demo", "extra": 1}}],
        expected={"get_task_detail": {"task_name": "release_demo"}},
    )
    assert result["ok"]
    assert result["total"] == 1
    assert result["exact_total"] == 0


def test_new_contract_overrides_legacy_exact_value_for_rewritten_search_query():
    result = _result(
        [{"name": "search_knowledge", "arguments": {"query": "GPU Reservation architecture purpose"}}],
        expected={"search_knowledge": {"query": "平台的 GPU Reservation 是什么？"}},
        contract={"search_knowledge": {"query": {"match": "non_empty"}}},
    )
    assert result["ok"]
    assert result["contract_accuracy"] == 1.0
    assert result["presence_coverage"] == 1.0
    assert result["exact_total"] == 0


def test_missing_argument_and_wrong_exact_argument_are_distinct():
    missing = _result([], contract={"diagnose_task": {"task_name": {"match": "exact", "value": "release_demo"}}})
    wrong = _result(
        [{"name": "diagnose_task", "arguments": {"task_name": "release_other"}}],
        contract={"diagnose_task": {"task_name": {"match": "exact", "value": "release_demo"}}},
    )
    assert missing["missing_arguments"]
    assert wrong["wrong_exact_arguments"]


def test_aligned_and_deepeval_deterministic_layers_share_contract_semantics():
    sample = {
        "case_id": "search",
        "category": "static_knowledge",
        "required_tools": ["search_knowledge"],
        "optional_tools": [],
        "forbidden_tools": [],
        "actual_tools": ["search_knowledge"],
        "actual_arguments": [{"name": "search_knowledge", "arguments": {"query": "GPU Reservation rules"}}],
        "tools_called": [{"name": "search_knowledge", "arguments": {"query": "GPU Reservation rules"}}],
        "expected_arguments": {"search_knowledge": {"query": "平台的 GPU Reservation 是什么？"}},
        "argument_contract": {"search_knowledge": {"query": {"match": "non_empty"}}},
    }
    hits, total, details = _argument_requirements(sample)
    metrics = _deterministic_tool_metrics([sample])
    assert (hits, total) == (1, 1)
    assert details[0]["matcher"] == "non_empty"
    assert metrics["argument_contract_accuracy"] == 1.0
    assert metrics["argument_presence_coverage"] == 1.0
    assert metrics["exact_argument_accuracy"] == 1.0


def test_aligned_evaluator_uses_same_rewritten_query_contract(monkeypatch, tmp_path):
    class FakeModel:
        async def plan(self, query, tool_descriptions, history):
            del query, tool_descriptions, history
            return AgentPlan(
                intent=AgentIntent.PLATFORM_KNOWLEDGE,
                tool_calls=[ToolCallSpec(
                    name="search_knowledge",
                    arguments={"query": "GPU Reservation architecture purpose"},
                )],
            )

    monkeypatch.setattr(aligned, "HeuristicReadOnlyModel", FakeModel)
    path = tmp_path / "case.jsonl"
    path.write_text(json.dumps({
        "id": "rewritten",
        "category": "static_knowledge",
        "query": "平台 GPU Reservation 为什么存在？",
        "expected_intent": "platform_knowledge",
        "required_tools": ["search_knowledge"],
        "optional_tools": [],
        "forbidden_tools": [],
        "expected_arguments": {"search_knowledge": {"query": "平台 GPU Reservation 为什么存在？"}},
        "argument_contract": {"search_knowledge": {"query": {"match": "non_empty"}}},
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    result = aligned.evaluate_agent_tool_contracts(path)
    assert result["argument_accuracy"] == 1.0
    assert result["argument_contract_accuracy"] == 1.0
