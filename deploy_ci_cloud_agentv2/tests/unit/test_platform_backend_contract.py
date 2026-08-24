from deploy_ci_cloud_agentv2.agent.results import QueueResult, normalize_read_result
from deploy_ci_cloud_agentv2.platform_backend.client import InProcessPlatformClient


class _QueueFacade:
    def __init__(self, payload):
        self.payload = payload

    def get_queue_state(self, _task_name=None):
        return self.payload


def test_in_process_queue_empty_snapshot_matches_v2_read_contract():
    payload = InProcessPlatformClient(
        _QueueFacade({"version": 2, "active": None, "queue": []})
    ).call("get_queue_state", {})

    assert payload == {"version": 2, "scope": "PLATFORM", "queue": []}
    result = normalize_read_result("get_queue_state", {}, payload)
    assert isinstance(result, QueueResult)
    assert result.qualifies_for_evidence()


def test_in_process_queue_active_snapshot_becomes_a_platform_queue_entry():
    payload = InProcessPlatformClient(
        _QueueFacade(
            {
                "version": 2,
                "active": {"task_name": "task_A", "status": "active"},
                "queue": [{"task_name": "task_B", "status": "queued"}],
            }
        )
    ).call("get_queue_state", {})

    assert payload["scope"] == "PLATFORM"
    assert payload["queue"] == [
        {"task_name": "task_A", "status": "active", "position": 0, "state": "ACTIVE"},
        {"task_name": "task_B", "status": "queued", "position": 1, "state": "QUEUED"},
    ]
    result = normalize_read_result("get_queue_state", {}, payload)
    assert isinstance(result, QueueResult)
    assert result.qualifies_for_evidence()
