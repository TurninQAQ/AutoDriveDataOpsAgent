from __future__ import annotations

import json
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


@pytest.mark.asyncio
async def test_deepeval_collector_provides_tool_descriptions(monkeypatch, tmp_path: Path):
    cases = tmp_path / "cases.jsonl"
    _write_cases(cases, "平台健康状态怎么样？")
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
    assert samples[0]["actual_tools"] == ["get_platform_health"]


@pytest.mark.asyncio
async def test_deepeval_required_tool_case_produces_nonempty_tool_call(monkeypatch, tmp_path: Path):
    cases = tmp_path / "cases.jsonl"
    _write_cases(cases, "当前 GPU Reservation 和显存情况怎么样？")
    monkeypatch.setattr(semantic, "InMemoryMCPToolClient", _CatalogClient)
    monkeypatch.setattr(semantic, "build_model_from_env", lambda *args: _CatalogAwareModel())

    samples = await semantic._collect_tool_samples_async(cases, _settings())

    assert samples[0]["required_tools"] == ["get_gpu_pool"]
    assert samples[0]["actual_tools"]
    assert "get_gpu_pool" in samples[0]["actual_tools"]
    assert samples[0]["actual_arguments"] == [{"name": "get_gpu_pool", "arguments": {}}]


def test_deepeval_marks_empty_required_tool_collection_invalid():
    result = run_deepeval_tool_metrics(
        [
            {
                "case_id": "gpu_case",
                "query": "当前 GPU Reservation 和显存情况怎么样？",
                "required_tools": ["get_gpu_pool"],
                "actual_tools": [],
                "actual_arguments": [],
                "tools_called": [],
                "expected_tools": [{"name": "get_gpu_pool", "arguments": {}}],
            }
        ]
    )

    assert result["status"] == COLLECTION_INVALID
    assert result["collection_status"] == COLLECTION_INVALID
    assert result["tool_correctness"] is None
    assert result["argument_correctness"] is None
    assert result["invalid_cases"][0]["case_id"] == "gpu_case"
    assert len(result["cases"]) == 1
    assert result["cases"][0]["collection_valid"] is False
    assert result["cases"][0]["actual_arguments"] == []
