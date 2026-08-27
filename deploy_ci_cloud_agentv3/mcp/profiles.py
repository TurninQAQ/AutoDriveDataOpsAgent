from __future__ import annotations

AGENT_PROFILE = "agent"
RUNTIME_PROFILE = "runtime"

READ_TOOLS = {
    "get_task_detail",
    "get_gpu_pool",
    "get_queue_state",
    "diagnose_task",
    "search_knowledge",
}
PREPARE_TOOLS = {"prepare_task_spec"}
PROPOSAL_TOOLS = {
    "propose_set_task_priority",
    "propose_resume_task",
    "propose_stop_task",
    "propose_delete_task",
    "propose_submit_task",
}
WRITE_TOOLS = {
    "set_task_priority",
    "resume_task",
    "stop_task",
    "delete_task",
    "submit_task",
}
RUNTIME_INTERNAL_TOOLS = {"capture_write_precondition", "get_action_verification_snapshot", "get_prepared_artifact", "get_task_config_for_verification"}

AGENT_TOOLS = READ_TOOLS | PREPARE_TOOLS | PROPOSAL_TOOLS
RUNTIME_TOOLS = READ_TOOLS | WRITE_TOOLS | RUNTIME_INTERNAL_TOOLS
