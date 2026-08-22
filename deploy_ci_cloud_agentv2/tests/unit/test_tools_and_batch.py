import asyncio

import pytest

from deploy_ci_cloud_agentv2.agent.decisions import ReadToolBatch, ToolCall
from deploy_ci_cloud_agentv2.platform import InMemoryReadFacade
from deploy_ci_cloud_agentv2.tools.catalog import build_read_registry
from deploy_ci_cloud_agentv2.tools.metadata import ToolKind
from deploy_ci_cloud_agentv2.tools.runtime import ReadFailure, ReadToolRuntime


def test_phase_b_tool_specs_have_read_retry_and_parallel_metadata():
    registry = build_read_registry(InMemoryReadFacade())
    assert registry.spec("get_task_detail").parallel_safe is True
    assert registry.spec("get_gpu_pool").parallel_safe is True
    assert registry.spec("search_knowledge").parallel_safe is True
    assert registry.spec("get_queue_state").parallel_safe is True
    assert registry.spec("diagnose_task").parallel_safe is False
    assert all(spec.kind is ToolKind.READ for spec in registry.catalog())


def test_batch_rejects_non_parallel_safe_tool_and_dependency_reference():
    runtime = ReadToolRuntime(build_read_registry(InMemoryReadFacade()))
    with pytest.raises(ValueError, match="parallel-safe"):
        runtime.validate_batch(
            ReadToolBatch((ToolCall("c1", "diagnose_task", {"task_name": "task_A"}),)),
            3,
        )
    with pytest.raises(ValueError, match="concrete"):
        runtime.validate_batch(
            ReadToolBatch((ToolCall("c1", "get_task_detail", {"task_name": "$call.c0"}),)),
            3,
        )


def test_partial_batch_failure_preserves_successful_siblings():
    facade = InMemoryReadFacade(
        responses={
            "get_task_detail": {"task_name": "task_A", "state": "running"},
            "get_queue_state": {"task_name": "task_A", "position": 1},
            "get_gpu_pool": {"devices": []},
        },
        failures={
            "get_gpu_pool": [
                ReadFailure("READ_TIMEOUT", "timeout", retryable=True),
                ReadFailure("READ_TIMEOUT", "timeout", retryable=True),
                ReadFailure("READ_TIMEOUT", "timeout", retryable=True),
            ]
        },
    )
    runtime = ReadToolRuntime(build_read_registry(facade))
    result = asyncio.run(
        runtime.execute_batch(
            ReadToolBatch(
                (
                    ToolCall("task", "get_task_detail", {"task_name": "task_A"}),
                    ToolCall("queue", "get_queue_state", {"task_name": "task_A"}),
                    ToolCall("gpu", "get_gpu_pool", {}),
                )
            ),
            max_retries=2,
        )
    )
    assert [item.status for item in result.results] == [
        "SUCCESS",
        "SUCCESS",
        "READ_FAILURE",
    ]
    assert [item.data for item in result.results[:2]] == [
        {"task_name": "task_A", "state": "running"},
        {"task_name": "task_A", "position": 1},
    ]


def test_read_timeout_has_bounded_side_effect_free_retry():
    facade = InMemoryReadFacade(
        responses={"get_gpu_pool": {"devices": [{"gpu_id": "0"}]}},
        failures={
            "get_gpu_pool": [ReadFailure("READ_TIMEOUT", "timeout", retryable=True), None]
        },
    )
    runtime = ReadToolRuntime(build_read_registry(facade))
    result = asyncio.run(
        runtime.execute_single(
            ToolCall("gpu", "get_gpu_pool", {}), max_retries=2
        )
    )
    assert result.status == "SUCCESS"
    assert result.retry_count == 1
    assert [call[0] for call in facade.calls] == ["get_gpu_pool", "get_gpu_pool"]
