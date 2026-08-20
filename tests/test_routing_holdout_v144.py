from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from platform_agent.prompt_contract import EVIDENCE_ROUTING_CONTRACT
from platform_mcp.server import WRITE_TOOL_NAMES
from platform_eval.aligned import load_jsonl
from platform_eval.argument_contract import validate_tool_cases


ROOT = Path(__file__).resolve().parents[1]
HOLDOUT = ROOT / "eval" / "v1_4_4" / "routing_holdout_cases.jsonl"


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans({"？": "?", "。": ".", "，": ",", "：": ":"}))
    return re.sub(r"\s+", " ", text)


def _prompt_examples() -> list[str]:
    return re.findall(r"['‘]([^'’]+)['’]", EVIDENCE_ROUTING_CONTRACT)


def _cases():
    return load_jsonl(HOLDOUT)


def test_v144_holdout_case_count_and_category_balance():
    cases = _cases()
    assert len(cases) == 48
    assert Counter(case["category"] for case in cases) == Counter({
        "static_knowledge": 8,
        "live_gpu_state": 6,
        "live_task_state": 6,
        "gpu_diagnosis": 4,
        "named_task_diagnosis": 6,
        "hybrid_live_knowledge": 5,
        "task_planning": 4,
        "write": 5,
        "no_tool": 4,
    })


def test_v144_schema_ids_and_tool_sets_are_valid():
    cases = _cases()
    validate_tool_cases(cases)
    assert len({case["id"] for case in cases}) == len(cases)


def test_v144_required_order_is_valid():
    for case in _cases():
        assert set(case.get("required_order", [])) <= set(case.get("required_tools", [])) | set(case.get("optional_tools", []))


def test_holdout_queries_do_not_copy_prompt_examples():
    examples = [_normalize(item) for item in _prompt_examples()]
    assert examples
    for case in _cases():
        query = _normalize(case["query"])
        assert query not in examples
        assert all(query not in example and example not in query for example in examples)


def test_v144_has_adversarial_overlap_cases():
    assert sum(bool(case.get("adversarial_overlap")) for case in _cases()) >= 10


def test_v144_hybrid_cases_require_both_evidence_sources():
    for case in _cases():
        if case["category"] != "hybrid_live_knowledge":
            continue
        required = set(case["required_tools"])
        assert "search_knowledge" in required
        assert required & {"get_gpu_pool", "get_task_detail", "diagnose_task"}


def test_v144_write_cases_forbid_direct_write_tools():
    for case in _cases():
        if case["category"] == "write":
            assert set(WRITE_TOOL_NAMES) <= set(case["forbidden_tools"])
            assert not set(WRITE_TOOL_NAMES) & set(case["required_tools"])
            assert not set(WRITE_TOOL_NAMES) & set(case["optional_tools"])
