from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

import platform_eval.semantic as semantic
from platform_agent.models import AgentIntent, AgentPlan, ToolCallSpec
from platform_eval.deepeval_adapter import COLLECTION_INVALID, run_deepeval_tool_metrics


def _settings():
    return SimpleNamespace(
        provider="qwen",
        model="qwen3.7-flash",
        temperature=0.0,
        base_url=None,
    )


def _write_cases(path: Path, *queries: str) -> None:
    rows = []
    for index, query in enumerate(queries):
        rows.append(
            {
                "id": f"case_{index}",
                "query": query,
                "category": "diagnosis",
                "expected_intent": "gpu_diagnosis" if "GPU" in query else "platform_health",
                "required_tools": [
                    "get_gpu_pool" if "GPU" in query else "get_platform_health"
                ],
                "optional_tools": [],
                "forbidden_tools": ["delete_task"],
                "expected_arguments": {},
            }
        )
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


class _CatalogClient:
    calls = 0

    async def describe_tools(self):
        type(self).calls += 1
        return [
            {
                "name": name,
                "description": f"MCP contract for {name}",
                "input_schema": {"type": "object"},
            }
            for name in (
                "get_platform_health",
                "get_gpu_pool",
                "get_task_detail",
                "diagnose_task",
            )
        ]


class _CatalogAwareModel:
    seen_descriptions: list[dict] = []

    async def plan(self, query, tool_descriptions, history):
        del history
        type(self).seen_descriptions = list(tool_descriptions)
        name = "get_gpu_pool" if "GPU" in query else "get_platform_health"
        intent = AgentIntent.GPU_DIAGNOSIS if name == "get_gpu_pool" else AgentIntent.PLATFORM_HEALTH
        return AgentPlan(
            intent=intent,
            tool_calls=[ToolCallSpec(name=name, arguments={})],
            decision_summary="catalog-grounded plan",
        )


def _fake_deepeval(monkeypatch, *, tool_score: float = 0.0, argument_score: float = 0.0):
    class FakeToolCall:
        def __init__(self, name, input_parameters=None):
            self.name = name
            self.input_parameters = input_parameters or {}

    class FakeLLMTestCase:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeMetric:
        def __init__(self, *args, **kwargs):
            self.score = None

        def measure(self, case):
            del case
            self.score = self._score

    class FakeToolMetric(FakeMetric):
        _score = tool_score

    class FakeArgumentMetric(FakeMetric):
        _score = argument_score

    deepeval = types.ModuleType("deepeval")
    metrics = types.ModuleType("deepeval.metrics")
    metrics.ToolCorrectnessMetric = FakeToolMetric
    metrics.ArgumentCorrectnessMetric = FakeArgumentMetric
    test_case = types.ModuleType("deepeval.test_case")
    test_case.LLMTestCase = FakeLLMTestCase
    test_case.ToolCall = FakeToolCall
    deepeval.metrics = metrics
    deepeval.test_case = test_case
    monkeypatch.setitem(sys.modules, "deepeval", deepeval)
    monkeypatch.setitem(sys.modules, "deepeval.metrics", metrics)
    monkeypatch.setitem(sys.modules, "deepeval.test_case", test_case)


@pytest.mark.asyncio
async def test_deepeval_collector_provides_tool_descriptions(monkeypatch, tmp_path: Path):
    cases = tmp_path / "cases.jsonl"
    _write_cases(cases, "平台健康状态怎么样？")
    monkeypatch.setattr(_CatalogClient, "calls", 0)
    monkeypatch.setattr(semantic, "InMemoryMCPToolClient", _CatalogClient)
    monkeypatch.setattr(semantic, "build_model_from_env", lambda *args: _CatalogAwareModel())

    samples = await semantic._collect_tool_samples_async(cases, _settings())

    assert _CatalogClient.calls == 1
    assert _CatalogAwareModel.seen_descriptions
    names = {item["name"] for item in _CatalogAwareModel.seen_descriptions}
    assert {
        "get_platform_health",
        "get_gpu_pool",
        "get_task_detail",
        "diagnose_task",
    }.issubset(names)
    assert "delete_task" not in names
    assert samples[0]["collection_valid"] is True
    assert samples[0]["planner_valid"] is True
    assert samples[0]["actual_tools"] == ["get_platform_health"]


@pytest.mark.asyncio
async def test_deepeval_required_tool_case_produces_nonempty_tool_call(monkeypatch, tmp_path: Path):
    cases = tmp_path / "cases.jsonl"
    _write_cases(cases, "当前 GPU Reservation 和显存情况怎么样？")
    monkeypatch.setattr(semantic, "InMemoryMCPToolClient", _CatalogClient)
    monkeypatch.setattr(semantic, "build_model_from_env", lambda *args: _CatalogAwareModel())

    samples = await semantic._collect_tool_samples_async(cases, _settings())

    assert samples[0]["required_tools"] == ["get_gpu_pool"]
    assert samples[0]["actual_tools"] == ["get_gpu_pool"]
    assert samples[0]["model_tool_miss"] is False
    assert samples[0]["actual_arguments"] == [{"name": "get_gpu_pool", "arguments": {}}]


@pytest.mark.asyncio
async def test_collection_invalid_when_tool_catalog_loading_fails(monkeypatch, tmp_path: Path):
    cases = tmp_path / "cases.jsonl"
    _write_cases(cases, "当前 GPU Reservation 和显存情况怎么样？")

    class BrokenCatalog:
        async def describe_tools(self):
            raise RuntimeError("catalog unavailable")

    monkeypatch.setattr(semantic, "InMemoryMCPToolClient", BrokenCatalog)
    monkeypatch.setattr(semantic, "build_model_from_env", lambda *args: _CatalogAwareModel())

    samples = await semantic._collect_tool_samples_async(cases, _settings())
    result = run_deepeval_tool_metrics(samples)

    assert samples[0]["collection_valid"] is False
    assert samples[0]["collection_error"] == "tool_catalog_loading_failed"
    assert result["collection_status"] == COLLECTION_INVALID
    assert result["collection_invalid_count"] == 1


@pytest.mark.asyncio
async def test_collection_invalid_when_model_plan_raises(monkeypatch, tmp_path: Path):
    cases = tmp_path / "cases.jsonl"
    _write_cases(cases, "当前 GPU Reservation 和显存情况怎么样？")

    class BrokenModel:
        async def plan(self, query, tool_descriptions, history):
            del query, tool_descriptions, history
            raise RuntimeError("planner unavailable")

    monkeypatch.setattr(semantic, "InMemoryMCPToolClient", _CatalogClient)
    monkeypatch.setattr(semantic, "build_model_from_env", lambda *args: BrokenModel())

    samples = await semantic._collect_tool_samples_async(cases, _settings())
    result = run_deepeval_tool_metrics(samples)

    assert samples[0]["collection_valid"] is False
    assert samples[0]["planner_valid"] is False
    assert samples[0]["collection_error"] == "model_plan_failed"
    assert result["collection_status"] == COLLECTION_INVALID


def _empty_model_sample(**overrides):
    sample = {
        "case_id": "gpu_case",
        "id": "gpu_case",
        "input": "当前 GPU Reservation 和显存情况怎么样？",
        "query": "当前 GPU Reservation 和显存情况怎么样？",
        "actual_output": "没有选择工具",
        "required_tools": ["get_gpu_pool"],
        "optional_tools": [],
        "forbidden_tools": [],
        "expected_tools": [{"name": "get_gpu_pool", "arguments": {}}],
        "expected_arguments": {"get_gpu_pool": {}},
        "actual_tools": [],
        "actual_arguments": [],
        "tools_called": [],
        "collection_valid": True,
        "planner_valid": True,
        "model_tool_miss": True,
        "expected_intent": "gpu_diagnosis",
        "actual_intent": "gpu_diagnosis",
        "write_action": None,
    }
    sample.update(overrides)
    return sample


def test_empty_actual_tools_is_not_collection_invalid(monkeypatch):
    _fake_deepeval(monkeypatch)
    result = run_deepeval_tool_metrics([_empty_model_sample()])

    assert result["collection_status"] != COLLECTION_INVALID
    assert result["collection_valid_count"] == 1
    assert result["collection_invalid_count"] == 0
    assert result["cases"][0]["collection_valid"] is True
    assert result["cases"][0]["model_tool_miss"] is True


def test_missing_required_tool_has_zero_recall(monkeypatch):
    _fake_deepeval(monkeypatch)
    result = run_deepeval_tool_metrics([_empty_model_sample()])

    assert result["tool_recall"] == 0.0
    assert result["tool_f1"] == 0.0
    assert {"case_id": "gpu_case", "failure_type": "TOOL_MISSING"} in result["failures"]


def test_write_case_without_impact_reads_is_valid_failure(monkeypatch):
    _fake_deepeval(monkeypatch)
    sample = _empty_model_sample(
        case_id="tool_delete",
        id="tool_delete",
        input="删除 release_demo",
        query="删除 release_demo",
        required_tools=["get_task_detail", "get_queue_state"],
        expected_tools=[
            {"name": "get_task_detail", "arguments": {"task_name": "release_demo"}},
            {"name": "get_queue_state", "arguments": {}}
        ],
        expected_arguments={"get_task_detail": {"task_name": "release_demo"}},
        expected_intent="delete_task",
        actual_intent="delete_task",
        write_action={"task_name": "release_demo"},
        actual_tools=[],
        actual_arguments=[],
        tools_called=[],
        model_tool_miss=True,
    )
    result = run_deepeval_tool_metrics([sample])

    assert result["collection_status"] != COLLECTION_INVALID
    case = result["cases"][0]
    assert case["collection_valid"] is True
    assert case["write_action"] == {"task_name": "release_demo"}
    assert case["forbidden_tools_called"] == []
    assert case["model_tool_miss"] is True
    assert {"case_id": "tool_delete", "failure_type": "TOOL_MISSING"} in result["failures"]


def test_argument_requirement_coverage_is_deterministic(monkeypatch):
    _fake_deepeval(monkeypatch)
    sample = _empty_model_sample(
        actual_tools=["get_task_detail"],
        actual_arguments=[{"name": "get_task_detail", "arguments": {"task_name": "release_demo", "extra": 1}}],
        tools_called=[{"name": "get_task_detail", "arguments": {"task_name": "release_demo", "extra": 1}}],
        required_tools=["get_task_detail"],
        expected_tools=[{"name": "get_task_detail", "arguments": {"task_name": "release_demo"}}],
        expected_arguments={"get_task_detail": {"task_name": "release_demo"}},
        model_tool_miss=False,
    )
    result = run_deepeval_tool_metrics([sample])

    assert result["argument_requirement_coverage"] == 1.0
    assert result["cases"][0]["argument_requirement_coverage"] == 1.0
