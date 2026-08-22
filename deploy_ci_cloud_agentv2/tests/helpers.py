from datetime import datetime, timezone

from deploy_ci_cloud_agentv2.agent.evidence import (
    ObservationDisposition,
    ToolObservation,
    TransportStatus,
)
from deploy_ci_cloud_agentv2.agent.identity import RequestIdentity
from deploy_ci_cloud_agentv2.agent.provenance import build_provenance
from deploy_ci_cloud_agentv2.agent.results import normalize_read_result
from deploy_ci_cloud_agentv2.agent.results import ResultStatus
from deploy_ci_cloud_agentv2.tools.runtime import classify_normalized_result


def identity(thread_id="thread", request_id="request", turn_id="turn"):
    return RequestIdentity(thread_id, request_id, turn_id)


def observation(
    tool,
    arguments,
    payload,
    *,
    owner=None,
    observation_id="obs-1",
    transport_status=TransportStatus.SUCCESS,
):
    owner = owner or identity()
    result = normalize_read_result(tool, arguments, payload) if transport_status is TransportStatus.SUCCESS else None
    disposition = classify_normalized_result(result, transport_status)
    return ToolObservation(
        observation_id=observation_id,
        call_id=f"call-{observation_id}",
        owner=owner,
        source=tool,
        target=(
            str(arguments.get("task_name"))
            if arguments.get("task_name") is not None
            else str(arguments.get("query", "platform"))
        ),
        transport_status=transport_status,
        disposition=disposition,
        data=payload,
        observed_at=datetime.now(timezone.utc),
        provenance=build_provenance(tool, arguments, result),
        result=result,
    )
