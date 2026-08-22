import pytest

from deploy_ci_cloud_agentv2.agent.field_contract import (
    FieldState,
    read_optional_bool,
    read_optional_int,
    read_optional_sequence,
    read_optional_string,
)
from deploy_ci_cloud_agentv2.agent.results import (
    DiagnosticResult,
    GpuPoolResult,
    KnowledgeResult,
    QueueResult,
    TaskDetailResult,
    normalize_read_result,
)
from deploy_ci_cloud_agentv2.agent.evidence import TransportStatus
from deploy_ci_cloud_agentv2.tools.runtime import classify_normalized_result


def _malformed(tool, payload, args=None):
    result = normalize_read_result(tool, args or {}, payload)
    assert result.validation_errors
    assert not result.is_valid
    assert classify_normalized_result(result, TransportStatus.SUCCESS).value == "MALFORMED"
    assert not result.qualifies_for_evidence()
    return result


def test_field_reader_preserves_absent_valid_and_invalid_states():
    assert read_optional_bool({}, "x").state is FieldState.ABSENT
    assert read_optional_bool({"x": True}, "x").state is FieldState.PRESENT_VALID
    assert read_optional_bool({"x": None}, "x").state is FieldState.PRESENT_INVALID
    assert read_optional_string({"x": " A "}, "x").value == "A"
    assert read_optional_string({"x": "   "}, "x").state is FieldState.PRESENT_INVALID
    assert read_optional_int({"x": True}, "x").state is FieldState.PRESENT_INVALID
    assert read_optional_sequence({"x": None}, "x").state is FieldState.PRESENT_INVALID


@pytest.mark.parametrize("bad_success", ["false", "true", 0, 1, None, [], {}])
def test_present_invalid_success_never_becomes_absent_or_success(bad_success):
    _malformed(
        "get_task_detail",
        {"success": bad_success, "task_name": "A", "state": "RUNNING"},
        {"task_name": "A"},
    )


@pytest.mark.parametrize("bad_status", [123, False, 0, None, [], {}, ""])
def test_present_invalid_status_fails_closed(bad_status):
    _malformed(
        "get_task_detail",
        {"status": bad_status, "task_name": "A", "state": "RUNNING"},
        {"task_name": "A"},
    )


@pytest.mark.parametrize("field", ["available", "not_found"])
@pytest.mark.parametrize("bad_value", ["false", 0, 1, None, [], {}])
def test_envelope_boolean_fields_are_exact(field, bad_value):
    payload = {field: bad_value, "task_name": "A", "state": "RUNNING"}
    _malformed("get_task_detail", payload, {"task_name": "A"})


@pytest.mark.parametrize("bad", [None, {}, "gpu0", 1, False])
def test_gpu_collections_are_strict(bad):
    _malformed("get_gpu_pool", {"devices": bad})
    _malformed("get_gpu_pool", {"reservations": bad})


@pytest.mark.parametrize("bad", [123, False, None, [], {}])
def test_knowledge_query_is_strict(bad):
    _malformed("search_knowledge", {"query": bad, "results": []}, {"query": "x"})


def test_knowledge_results_none_is_not_an_empty_success():
    _malformed("search_knowledge", {"query": "x", "results": None}, {"query": "x"})


@pytest.mark.parametrize("bad", [123, False, None, [], {}])
def test_diagnosis_identity_is_strict(bad):
    _malformed("diagnose_task", {"task_name": bad, "root_cause": "failure"}, {"task_name": "A"})


@pytest.mark.parametrize("bad", [123, [], {}, ""])
def test_queue_scope_cannot_be_inferred_from_invalid_scope(bad):
    result = _malformed("get_queue_state", {"scope": bad, "queue": []}, {"task_name": None})
    assert result.observed_scope.kind.value == "UNKNOWN"


@pytest.mark.parametrize("bad", [123, None, [], {}, "", "   "])
def test_queue_task_identity_cannot_be_dropped_to_platform(bad):
    result = _malformed("get_queue_state", {"task_name": bad, "queue": []}, {"task_name": None})
    assert result.observed_scope.kind.value == "UNKNOWN"


def test_queue_active_null_is_not_meaningful():
    _malformed("get_queue_state", {"active": None}, {"task_name": None})


def test_positive_typed_contracts_still_qualify():
    task = normalize_read_result("get_task_detail", {"task_name": "A"}, {"task_name": "A", "state": "RUNNING"})
    gpu = normalize_read_result("get_gpu_pool", {}, {"devices": [{"gpu_id": "0"}], "reservations": []})
    knowledge = normalize_read_result("search_knowledge", {"query": "x"}, {"query": "x", "results": [{"title": "x", "content": "meaning"}]})
    queue_platform = normalize_read_result("get_queue_state", {"task_name": None}, {"scope": "PLATFORM", "queue": []})
    queue_task = normalize_read_result("get_queue_state", {"task_name": "A"}, {"scope": "TASK", "task_name": "A", "position": 1})
    diagnosis = normalize_read_result("diagnose_task", {"task_name": "A"}, {"task_name": "A", "root_cause": "CUDA OOM"})
    assert isinstance(task, TaskDetailResult) and task.qualifies_for_evidence()
    assert isinstance(gpu, GpuPoolResult) and gpu.qualifies_for_evidence()
    assert isinstance(knowledge, KnowledgeResult) and knowledge.qualifies_for_evidence()
    assert isinstance(queue_platform, QueueResult) and queue_platform.qualifies_for_evidence()
    assert isinstance(queue_task, QueueResult) and queue_task.qualifies_for_evidence()
    assert isinstance(diagnosis, DiagnosticResult) and diagnosis.qualifies_for_evidence()


@pytest.mark.parametrize("field", ["success", "status", "available", "not_found", "error", "error_code"])
def test_known_envelope_field_wrong_type_is_not_ignored(field):
    values = {"success": "false", "status": 7, "available": "false", "not_found": 0, "error": [], "error_code": {}}
    payload = {"task_name": "A", "state": "RUNNING", field: values[field]}
    _malformed("get_task_detail", payload, {"task_name": "A"})
