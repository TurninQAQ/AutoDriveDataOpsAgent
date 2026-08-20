from __future__ import annotations

import json
from pathlib import Path

from platform_eval.aligned import load_jsonl
from platform_eval.deepeval_adapter import _contract_metrics, _deterministic_tool_metrics


ROOT = Path(__file__).resolve().parents[1]
V11 = ROOT / "eval" / "v1_1" / "agent_tool_cases.jsonl"
V132 = ROOT / "eval" / "v1_3_2" / "agent_tool_cases.jsonl"


def test_v132_contract_preserves_case_ids_queries_and_count():
    old = load_jsonl(V11)
    new = load_jsonl(V132)

    assert len(old) == len(new) == 21
    assert [(case["id"], case["query"]) for case in old] == [
        (case["id"], case["query"]) for case in new
    ]


def test_v132_contract_reclassifies_only_audited_requirements():
    old = {case["id"]: case for case in load_jsonl(V11)}
    new = {case["id"]: case for case in load_jsonl(V132)}

    assert old["tool_status"]["required_tools"] == ["get_task_detail", "get_queue_state"]
    assert new["tool_status"]["required_tools"] == ["get_task_detail"]
    assert new["tool_status"]["optional_tools"] == ["get_queue_state"]
    assert new["tool_stuck"]["required_tools"] == ["diagnose_task"]
    assert new["tool_stuck"]["optional_tools"] == ["get_task_detail"]
    assert new["tool_gpu"]["required_tools"] == old["tool_gpu"]["required_tools"]
    assert new["tool_oom"]["required_tools"] == ["diagnose_task", "get_stage_logs"]

    for case_id in (
        "tool_delete",
        "tool_stop",
        "tool_stop_clip",
        "tool_resume",
        "tool_priority",
        "tool_submit",
    ):
        assert new[case_id]["required_tools"] == []
        assert new[case_id]["expected_arguments"] == {}
        assert new[case_id]["forbidden_tools"] == old[case_id]["forbidden_tools"]

    assert old["tool_delete"]["required_tools"] == ["get_task_detail", "get_queue_state"]


def test_contract_metrics_split_read_and_write_contracts():
    read = {
        "case_id": "tool_status",
        "category": "task",
        "required_tools": ["get_task_detail"],
        "optional_tools": ["get_queue_state"],
        "forbidden_tools": [],
        "expected_arguments": {"get_task_detail": {"task_name": "release_demo"}},
        "actual_tools": ["get_task_detail", "get_queue_state"],
        "actual_arguments": [
            {"name": "get_task_detail", "arguments": {"task_name": "release_demo"}},
            {"name": "get_queue_state", "arguments": {}},
        ],
        "expected_intent": "task_status",
        "actual_intent": "task_status",
    }
    write = {
        "case_id": "tool_delete",
        "category": "write",
        "required_tools": [],
        "optional_tools": ["get_task_detail", "get_queue_state"],
        "forbidden_tools": ["delete_task"],
        "expected_arguments": {},
        "actual_tools": [],
        "actual_arguments": [],
        "expected_intent": "delete_task",
        "actual_intent": "delete_task",
    }
    read_metrics = _deterministic_tool_metrics([read])
    write_metrics = _deterministic_tool_metrics([write])
    result = _contract_metrics([read, write], read_metrics, write_metrics, [1.0], [1.0])

    assert result["read_metrics"]["tool_precision"] == 1.0
    assert result["read_metrics"]["tool_recall"] == 1.0
    assert result["read_metrics"]["argument_requirement_coverage"] == 1.0
    assert result["write_metrics"]["pre_action_observation_rate"] == 0.0
    assert result["write_metrics"]["write_action_accuracy"]["status"] == (
        "NOT_AVAILABLE_FROM_CURRENT_GOLDEN_SCHEMA"
    )
    assert result["write_metrics"]["forbidden_write_tool_rate"] == 0.0
    assert result["safety_metrics"]["hitl_enforcement"]["status"] == "COVERED_BY_DETERMINISTIC_TESTS"


def test_v11_contract_file_has_not_been_rewritten():
    # The v1_1 bytes are tracked history; only the new version may differ.
    assert json.loads(V11.read_text(encoding="utf-8").splitlines()[3])["required_tools"] == [
        "get_task_detail",
        "get_queue_state",
    ]
